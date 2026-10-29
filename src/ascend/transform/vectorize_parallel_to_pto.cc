// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file vectorize_parallel_to_pto.cc
 * \brief VectorizeParallelToPTO: rewrite verified T.Parallel loops inside
 * PTO SIMD_VF blocks into logical VMI code.
 *
 * Consumes the pto_parallel_loop_layout fragment written by
 * LayoutInference and verified by VerifyParallelToPTO. Per parallel unit:
 *   1. invert the lane fragment through the shared helper (the inverse
 *      layout's own Forward result is the single source of truth);
 *   2. build the chunk loop for q in [0, E/L), mapping the loop variable
 *      to q * L + lane;
 *   3. classify every access via the shared MemoryAccess analysis
 *      (continuous -> vload/vstore via tl.access_ptr(start, L, rw);
 *      lane-uniform -> scalar BufferLoad with lane bound to 0);
 *   4. convert value expressions (value roles tracked by result dtype);
 *      every VMI with a return value is bound through EmitBind before use;
 *   5. delete the consumed layout annotation, keep the SIMD_VF block
 *      annotations, and leave no unconverted Parallel behind.
 *
 * Supported shapes: 1D parallel units and two-layer adjacent Parallel nests
 * (2D — the nest converts as one unit with the layout
 * annotation on its outermost For); sequential units allowed; static
 * divisible extents; full masks. A third parallel layer or any other
 * unsupported structure is rejected by Verify.
 *
 * Non-divisible extents emit a tail chunk whose vload reads the full L
 * lanes from its chunk base: the ISA intentionally keeps load instructions
 * unmasked for performance, so the per-chunk tail mask constrains
 * computation and stores only, never the load. Reading the full L lanes
 * past the logical tail is still safe on device — the top 8K of UB is
 * reserved as stack space, so the extra read-only lanes cannot fault; they
 * carry undefined values that the compute and store masks keep out of the
 * valid outputs. Compact outputs of exactly E elements remain safe because
 * the masked store never writes past E.
 */

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "../../op/operator.h"
#include "../op/builtin.h"
#include "parallel_to_pto_utils.h"
#include "pto_slot_plan.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
namespace pto_analysis = ::tvm::tl::pto;

namespace {

DataType VectorDType(DataType elem, int64_t lanes) {
  return elem.with_lanes(static_cast<int>(lanes));
}

const Op &VmiOp(const std::string &tail) { return Op::Get("tl.vmi." + tail); }

// This version keeps local DCE in Vectorize so
// its directly emitted TIR/PTODSL is cleaner and easier to inspect.
// Cleanup may later move to PTOAS canonicalization/DCE to reduce
// TileLang-specific code and maintenance; clean-output checks would then
// target the optimized PTOAS IR. The emitted IR must be valid and preserve
// program semantics even before DCE.
//
// Dead vector-chain elimination on one converted chunk body. The
// conversion can leave Bind statements whose Var is never referenced:
// source binds consumed only through the address alias environment
// (bind_env_) never use their converted Var, and an unused source bind
// strands its whole value chain (vci/vbrc/vadd/vload).
//
// The traversal is a scope-aware *reverse liveness* pass — a
// dead Bind must not survive just because it was wrapped in an AttrStmt
// or appears as the only statement of a body. The rules are:
//   1. A SeqStmt is processed back to front; `live` always holds the
//      vars everything *after* the current statement needs.
//   2. A Bind's LHS var is a definition, never a use. When the result is
//      not live and the RHS is discardable, the Bind is deleted and its
//      RHS inputs do NOT become live (so the dead chain collapses in one
//      pass); otherwise it is kept and its RHS inputs join `live`.
//   3. An AttrStmt body is cleaned in its own nested scope with a fresh
//      liveness set; only the vars actually referenced inside propagate
//      outward. Attribute node/value references are always uses, and the
//      attribute structure and evaluation position are preserved.
//   4. A bare Bind follows the same rules; when everything dies the
//      remains are a legal no-op (Evaluate(0)).
// Nothing is merged, reordered or otherwise rescheduled;
// referenced loads stay exactly where they were (vecadd keeps its three
// vloads), and Bind-free (frontend-inlined) bodies are untouched.
class DeadBindElim {
public:
  Stmt operator()(const Stmt &stmt) {
    std::set<const VarNode *> live;
    Stmt out = Run(stmt, &live);
    return out.defined() ? out : NoOp();
  }

private:
  static Stmt NoOp() { return Evaluate(make_zero(DataType::Int(32))); }

  static void CollectExprVars(const PrimExpr &expr,
                              std::set<const VarNode *> *live) {
    PostOrderVisit(expr, [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        live->insert(v);
      }
    });
  }

  static void CollectStmtVars(const Stmt &stmt,
                              std::set<const VarNode *> *live) {
    PostOrderVisit(stmt, [&](const ObjectRef &n) {
      if (const auto *v = n.as<VarNode>()) {
        live->insert(v);
      }
    });
  }

  static bool DiscardableRhs(const PrimExpr &rhs) {
    // Positive "allowed to discard" set: a Call may disappear
    // only when it is one of the tl.vmi.* ops this pass actually emits as
    // a Bind RHS and that is pure/read-only by construction — vload
    // (read), vbrc, vci, vadd, vmul and vcvt (pure), enumerated from the
    // ConvertUnit/VmiOp call sites — or the pure tl.access_ptr address
    // descriptor nested inside a vload operand (already whitelisted as a
    // pure index intrinsic by IsSupportedAddressExpr). vstore never
    // appears as a Bind RHS; any other Call — call_extern, other tl.* or
    // an unknown op — keeps its Bind. Plain UB BufferLoads are
    // discardable: they are ordinary reads with no write effect (the
    // scalar "sload" binds). That "ordinary" premise is enforced by
    // VerifyParallelToPTO, not assumed here: accesses to
    // tirx.volatile-annotated allocations and volatile_scope attributes
    // are rejected there, and opaque calls are already excluded
    // from addresses.
    static const std::set<std::string> kDiscardable = {
        "tl.vmi.vload",
        "tl.vmi.vbrc",
        "tl.vmi.vci",
        "tl.vmi.vadd",
        "tl.vmi.vsub",
        "tl.vmi.vmul",
        "tl.vmi.vdiv",
        "tl.vmi.vabs",
        "tl.vmi.vexp",
        "tl.vmi.vln",
        "tl.vmi.vsqrt",
        "tl.vmi.vcvt",
        "tl.vmi.vinterpret_cast",
        // Control-flow ops (vcmp/vcmps produce masks, vsel/vand/vor/vnot
        // combine them); an unused result is discardable like the arithmetic
        // ops.
        "tl.vmi.vcmp",
        "tl.vmi.vcmps",
        "tl.vmi.vsel",
        "tl.vmi.vand",
        "tl.vmi.vor",
        "tl.vmi.vnot",
        "tl.access_ptr",
    };
    bool discardable = true;
    PostOrderVisit(rhs, [&](const ObjectRef &n) {
      if (const auto *call = n.as<CallNode>()) {
        const auto *op = call->op.as<OpNode>();
        const std::string name = op != nullptr ? op->name : "";
        if (kDiscardable.count(name) == 0) {
          discardable = false;
        }
      }
    });
    return discardable;
  }

  // Returns the kept statement, or an *undefined* Stmt when the statement
  // was deleted (only a Bind can be deleted). `live` is threaded from
  // later statements back to earlier ones.
  Stmt Run(const Stmt &stmt, std::set<const VarNode *> *live) {
    if (const auto *bind = stmt.as<BindNode>()) {
      if (live->count(bind->var.get()) == 0 && DiscardableRhs(bind->value)) {
        // Dead and safe to drop: the RHS inputs stay out of `live`, so a
        // chain feeding only this Bind collapses in the same pass.
        return Stmt();
      }
      // Kept: classic liveness — subtract the definition, add the uses.
      live->erase(bind->var.get());
      CollectExprVars(bind->value, live);
      return stmt;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      Array<Stmt> kept;
      for (auto it = seq->seq.rbegin(); it != seq->seq.rend(); ++it) {
        Stmt sub = Run(*it, live);
        if (sub.defined()) {
          kept.push_back(sub);
        }
      }
      if (kept.empty()) {
        return NoOp();
      }
      // std::reverse does not accept the ffi Array iterators on this
      // toolchain; re-append in reverse instead.
      Array<Stmt> forward;
      for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
        forward.push_back(*it);
      }
      return SeqStmt::Flatten(forward);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      // Nested scope: clean the body with a fresh liveness set, then
      // propagate only the references that survive to its entry (external
      // uses; vars defined and used only inside never escape because the
      // converted Vars are unique objects). Attribute node/value
      // references are always uses and keep the structure.
      std::set<const VarNode *> body_live;
      Stmt body = Run(attr->body, &body_live);
      if (!body.defined()) {
        body = NoOp();
      }
      live->insert(body_live.begin(), body_live.end());
      if (auto ne = attr->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          CollectExprVars(GetRef<PrimExpr>(pe), live);
        }
      }
      CollectExprVars(attr->value, live);
      if (body.same_as(attr->body)) {
        return stmt;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    }
    // Any other statement is always kept; all its Var occurrences are
    // uses.
    CollectStmtVars(stmt, live);
    return stmt;
  }
};

// Scalar fold of a binary operation over the converted lane-uniform
// operands: rebuild the TIR semantics of the source node.
using ScalarFoldFn = PrimExpr (*)(const PrimExpr &, const PrimExpr &);

class VectorizeParallelToPTOImpl : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    VectorizeParallelToPTOImpl mutator;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = mutator.VisitStmt(f->body);
    return GetRef<PrimFunc>(fptr);
  }

private:
  // Kernel/outer serial loops in scope at the current position (outside
  // any region): bound here so a fragment entry depending on them
  // (f[b, j]) proves its interval — the region-side walk cannot see
  // them, and Verify bound the same ranges when it accepted the program.
  Stmt VisitStmt_(const ForNode *op) final {
    // Any non-parallel loop with constant bounds contributes its range
    // (T.unroll included): Verify's kernel-level visitor binds every
    // non-parallel loop, so Vectorize must mirror that set or a
    // fragment entry under T.unroll fails the interval proof only on
    // the Vectorize side with a misleading "can be negative" message.
    if (op->kind != ForKind::kParallel && op->min.as<IntImmNode>() != nullptr &&
        op->extent.as<IntImmNode>() != nullptr) {
      Map<Var, Range> saved = outer_serial_ranges_;
      outer_serial_ranges_.Set(
          op->loop_var, Range::FromMinExtent(op->min, op->extent));
      Stmt result = StmtExprMutator::VisitStmt_(op);
      outer_serial_ranges_ = std::move(saved);
      return result;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // ------------------------------------------------------------------
  // Unified unit discovery (shared rules with Verify).
  // A SIMD_VF region contains zero or more *sequential* parallel units.
  // Each unit is the outermost parallel For of a 1D loop or of a
  // two-layer adjacent Parallel nest (2D); a third parallel layer (or a
  // Parallel under an unsupported wrapper) is rejected by Verify and
  // defensively rejected here too.
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // SIMD_VF block entry
  // ------------------------------------------------------------------
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != pto_analysis::kSimdVfBlockName) {
      block_stack_.push_back(op);
      Stmt result = StmtExprMutator::VisitStmt_(op);
      block_stack_.pop_back();
      // Fragments allocated at kernel scope were consumed by a child
      // region's conversion: drop the dead allocation and its layout-map
      // entry here so no stale reference survives into codegen (the
      // region blocks clean their own entries; this covers the outer
      // owners).
      return DropConsumedFragmentsFromBlock(op, result);
    }
    // Shared region scan: the same implementation Verify uses, so
    // skip/reject/convert decisions cannot drift apart.
    const Stmt &region = op->body;
    pto_analysis::PtoRegionScan scan = pto_analysis::ScanPtoRegion(region);
    if (!scan.has_any_parallel) {
      // Pure hand-written VMI region (possibly with control flow): skipped
      // by both passes; no lanes annotation needed.
      block_stack_.push_back(op);
      Stmt result = StmtExprMutator::VisitStmt_(op);
      block_stack_.pop_back();
      return result;
    }
    ICHECK(!scan.has_unsupported_wrapper)
        << "[VectorizeParallelToPTO] internal error: unsupported wrapper "
           "structure passed Verify: "
        << scan.issue;
    ICHECK(!scan.units.empty())
        << "[VectorizeParallelToPTO] internal error: parallel units lost "
           "after scan";

    // Track the block on the annotation chain for the slot-state lookups
    // in ConvertRegionBody (fragment layouts and plans).
    block_stack_.push_back(op);
    // The region is always converted as a whole and its loop structure is
    // kept. A serial loop that carries a UB buffer between iterations stays
    // inside one vecscope, so the ordering comes from the backend: ptoas
    // analyses intra-vecscope dependencies (loop-carried included) and inserts
    // pto.mem_bar. When the value can stay in a register (`T.alloc_var`),
    // the per-lane vreg promotion below avoids the round trip altogether.
    //
    // The block itself is rebuilt around the converted body: its alloc_buffers
    // declare the region-local UB buffers, and fragment allocations consumed
    // by the conversion are dropped.
    Stmt body = ConvertRegionBody(region);
    Stmt rebuilt = RebuildBlockWithoutConsumedFragments(op, std::move(body));
    block_stack_.pop_back();
    return rebuilt;
  }

  // Convert one SIMD_VF region body: mask plan, slot state, carry promotion,
  // units.
  Stmt ConvertRegionBody(const Stmt &region) {
    // Reset the region-level slot state: every local.fragment buffer of
    // this converting region maps to one reserved backing (shared by all
    // sequential units — e.g. an accumulator kept across units).
    BuildSlotState();
    pto_analysis::PtoRegionScan scan = pto_analysis::ScanPtoRegion(region);
    ICHECK(!scan.units.empty())
        << "[VectorizeParallelToPTO] internal error: parallel units lost "
           "after scan";
    // One inverse per unit: the same mapping feeds the lanes read, the
    // shared-mask decision and the unit conversion below, so a region
    // with n units inverts n times instead of 2n+1. The inversion is
    // the expensive part (layout reshape + analyzer proofs); the
    // mappings are small value structs.
    std::vector<pto_analysis::PtoInverseMapping> mappings;
    mappings.reserve(scan.units.size());
    for (const For &unit : scan.units) {
      mappings.push_back(pto_analysis::InvertPtoLaneLayout(unit));
    }
    // Lanes come from the first unit's layout; all units in one VF
    // share the same L (Verify enforces the annotation consistency).
    int64_t lanes = mappings[0].lanes;
    // One declare_fragment_buffer per plan whose fragment this region
    // accesses, bound once before the first unit and shared by every
    // sequential unit of the VF (e.g. an accumulator kept across loops).
    // Each region occurrence binds a fresh handle Var (named after the
    // plan's pre-created fragment_mem_var); its definition dominates all
    // uses because it is inserted at the top of the vecscope.
    Stmt new_region = EmitSlotDeclarations(region, lanes);
    // A full mask binding per VF, shared by the divisible units. A
    // region whose units all have non-divisible extents never uses it
    // (every unit builds its own per-chunk dynamic mask), so the
    // binding is only emitted when at least one divisible unit needs
    // it.
    bool any_divisible = false;
    for (const pto_analysis::PtoInverseMapping &mapping : mappings) {
      if (mapping.tail_lanes == 0) {
        any_divisible = true;
        break;
      }
    }
    if (any_divisible) {
      shared_mask_ = Var("mask", DataType::Bool(static_cast<int>(lanes)));
      PrimExpr mask_value = Call(
          DataType::Bool(static_cast<int>(lanes)), VmiOp("create_mask"),
          {IntImm(DataType::Int(32), static_cast<int>(lanes))},
          {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes))}});
      // Define the shared mask inside the simdvf scope attr when the
      // region is wrapped by one: VPTO treats vecscope-region data as
      // region-local and rejects pset values crossing the boundary.
      const auto *scope_attr = new_region.as<AttrStmtNode>();
      if (scope_attr != nullptr &&
          scope_attr->attr_key == "tl.simdvf_scope") {
        new_region = AttrStmt(scope_attr->node, scope_attr->attr_key,
                              scope_attr->value,
                              SeqStmt::Flatten(Bind(shared_mask_, mask_value),
                                               scope_attr->body));
      } else {
        new_region =
            SeqStmt::Flatten(Bind(shared_mask_, mask_value), new_region);
      }
    }

    // Convert each sequential unit in order; unit state (bind
    // maps, counters) is reset per unit; the shared full mask, when
    // emitted, is visible to every unit.
    for (size_t i = 0; i < scan.units.size(); ++i) {
      Stmt converted = ConvertUnit(scan.units[i], mappings[i], region);
      // No TileLang-side backing mem_bar: ordering of backing accesses
      // (cross-unit RAW/WAR, in-shell cross-iteration RAW) is the
      // toolchain's responsibility — PR307's automatic vecscope barriers
      // model the affine-address, constant-bound forms this pass emits.
      // The toolchain floor is the pr274-pr307-merge lineage; validated
      // by the device suites.
      new_region = ReplaceStmt(new_region, scan.units[i], converted);
    }
    if (region_carry_.active) {
      new_region = SeqStmt::Flatten(AllocBuffer(region_carry_.vreg),
                                    new_region);
      region_carry_ = RegionCarry();
    }
    CheckNoResidualFragmentAccess(new_region);
    CheckSlotDeclarations(new_region);
    return new_region;
  }

  // ------------------------------------------------------------------
  // PTO slot storage state (region level)
  // ------------------------------------------------------------------
  struct SlotState {
    PtoSlotPlan plan;
    Fragment layout;
    // Handle bound by this region's declare_fragment_buffer. A fresh Var
    // per region occurrence: a vecscope split makes several regions out of
    // one plan, and each part defines its own handle (identical backing,
    // count and lanes; never two Binds of the same Var). The plan's
    // pre-created fragment_mem_var provides the name and the planning
    // identity.
    Var handle;
  };

  // Load the slot plans and fragment layouts recorded on the SBlock chain
  // (innermost first — the converting block itself and its ancestors; both
  // PlanPtoSlotStorage and the layout writeback place them there).
  void BuildSlotState() {
    slot_map_.clear();
    converted_fragments_.clear();
    for (auto it = block_stack_.rbegin(); it != block_stack_.rend(); ++it) {
      auto plans_value = (*it)->annotations.Get(kPtoSlotPlan);
      if (!plans_value.has_value()) {
        continue;
      }
      auto plans = plans_value.value().try_cast<Array<PtoSlotPlan>>();
      ICHECK(plans.has_value())
          << "[VectorizeParallelToPTO] internal error: annotation "
          << kPtoSlotPlan << " must be an Array<PtoSlotPlan>";
      for (const PtoSlotPlan &plan : plans.value()) {
        if (slot_map_.count(plan->fragment) != 0) {
          continue;
        }
        const Fragment layout = FindFragmentLayout(plan->fragment);
        ICHECK(layout.defined())
            << "[VectorizeParallelToPTO] internal error: fragment buffer `"
            << plan->fragment->name
            << "` has a slot plan but no layout_map entry";
        slot_map_.emplace(plan->fragment, SlotState{plan, layout, Var()});
      }
    }
  }

  Fragment FindFragmentLayout(const Buffer &buffer) const {
    for (auto it = block_stack_.rbegin(); it != block_stack_.rend(); ++it) {
      auto value = (*it)->annotations.Get(attr::kLayoutMap);
      if (!value.has_value()) {
        continue;
      }
      auto map = value.value().try_cast<LayoutMap>();
      if (!map.has_value()) {
        continue;
      }
      if (map.value().count(buffer) != 0) {
        auto frag = map.value()[buffer].as<Fragment>();
        ICHECK(frag.has_value())
            << "[VectorizeParallelToPTO] internal error: layout_map entry "
               "for fragment buffer `"
            << buffer->name << "` must be a tl.Fragment";
        return frag.value();
      }
    }
    return Fragment();
  }

  // Base address operand of one slot's declaration: a standard
  // rewritable access pointer over the whole backing (count * lanes
  // elements at element offset 0), so the address pass chain
  // (LowerAccessPtr / FlattenBuffer / MergeUBAllocations) rewrites it
  // exactly like an ordinary UB access and the declaration ends up
  // carrying the backing's final arena byte offset.
  PrimExpr MakeBackingBasePtr(const SlotState &state) const {
    const int64_t total = state.plan->count * state.plan->vector_lanes;
    return Call(DataType::Handle(), tl::access_ptr(),
                {BufferLoad(state.plan->backing,
                            {IntImm(DataType::Int(64), 0)}),
                 IntImm(DataType::Int(32), static_cast<int>(total)),
                 IntImm(DataType::Int(32), 3)});
  }

  // Fragment buffers of this region that map to a slot, in document
  // order of first access; drives one declaration per plan per VF.
  std::vector<Buffer> CollectAccessedSlotFragments(const Stmt &region) const {
    class AccessCollector : public StmtExprVisitor {
    public:
      static std::vector<Buffer> Run(const Stmt &body,
                                     const std::unordered_map<
                                         Buffer, SlotState, ObjectPtrHash,
                                         ObjectPtrEqual> &slot_map) {
        AccessCollector collector(slot_map);
        collector(body);
        return std::move(collector.fragments_);
      }

    private:
      explicit AccessCollector(
          const std::unordered_map<Buffer, SlotState, ObjectPtrHash,
                                   ObjectPtrEqual> &slot_map)
          : slot_map_(slot_map) {}
      void Record(const Buffer &buffer) {
        if (slot_map_.count(buffer) == 0) {
          return;
        }
        for (const Buffer &seen : fragments_) {
          if (seen.same_as(buffer)) {
            return;
          }
        }
        fragments_.push_back(buffer);
      }
      void VisitExpr_(const BufferLoadNode *op) final {
        Record(op->buffer);
        StmtExprVisitor::VisitExpr_(op);
      }
      void VisitStmt_(const BufferStoreNode *op) final {
        Record(op->buffer);
        // The base visitor may not descend into BufferStore predicates;
        // record fragment reads appearing only there explicitly (a
        // split part whose only use is a store predicate would otherwise
        // lose its declaration).
        if (op->predicate.defined()) {
          VisitExpr(op->predicate.value());
        }
        StmtExprVisitor::VisitStmt_(op);
      }
      const std::unordered_map<Buffer, SlotState, ObjectPtrHash,
                               ObjectPtrEqual> &slot_map_;
      std::vector<Buffer> fragments_;
    };
    return AccessCollector::Run(region, slot_map_);
  }

  // Bind one declare_fragment_buffer per accessed plan at the top of the
  // region, inside the tl.simdvf_scope attr when the region is wrapped by
  // one (VPTO treats vecscope data as region-local).
  // Strip alloc_buffers / layout_map entries of fragments consumed by
  // conversion from a rebuilt outer block. No-op unless a converting
  // region below this block consumed a kernel-scope fragment.
  Stmt DropConsumedFragmentsFromBlock(const SBlockNode *original,
                                      const Stmt &rebuilt) {
    if (converted_fragments_.empty()) {
      return rebuilt;
    }
    Array<Buffer> allocs;
    bool changed = false;
    for (const Buffer &buffer : original->alloc_buffers) {
      if (converted_fragments_.count(buffer->data.get()) != 0) {
        changed = true;
        continue;
      }
      allocs.push_back(buffer);
    }
    Map<Var, Layout> layouts;
    bool layout_changed = false;
    if (auto anno = original->annotations.Get(attr::kLayoutMap);
        anno.has_value()) {
      if (auto map = anno.value().try_cast<Map<Var, Layout>>();
          map.has_value()) {
        for (const auto &[var, layout] : map.value()) {
          if (converted_fragments_.count(var.get()) != 0) {
            layout_changed = true;
            continue;
          }
          layouts.Set(var, layout);
        }
      }
    }
    if (!changed && !layout_changed) {
      return rebuilt;
    }
    const auto *block = rebuilt.as<SBlockNode>();
    ICHECK(block != nullptr)
        << "[VectorizeParallelToPTO] internal error: a non-SBlock "
           "statement carries fragment allocations";
    Map<String, Any> annotations = block->annotations;
    if (layout_changed) {
      annotations.Set(attr::kLayoutMap, layouts);
    }
    return SBlock(block->iter_vars, block->reads, block->writes,
                  block->name_hint, block->body, block->init, allocs,
                  block->match_buffers, annotations);
  }

  Stmt EmitSlotDeclarations(const Stmt &region, int64_t lanes) {
    declared_handles_.clear();
    handle_plans_.clear();
    const std::vector<Buffer> accessed = CollectAccessedSlotFragments(region);
    if (accessed.empty()) {
      return region;
    }
    Array<Stmt> binds;
    for (const Buffer &fragment : accessed) {
      auto it = slot_map_.find(fragment);
      ICHECK(it != slot_map_.end())
          << "[VectorizeParallelToPTO] internal error: fragment buffer `"
          << fragment->name
          << "` has accesses but no slot plan in this region's scope";
      SlotState &state = it->second;
      ICHECK_EQ(state.plan->vector_lanes, lanes)
          << "[VectorizeParallelToPTO] internal error: slot plan for buffer `"
          << fragment->name << "` has VL " << state.plan->vector_lanes
          << " but the region converts with lanes " << lanes;
      PrimExpr decl = Call(
          DataType::Handle(), VmiOp("declare_fragment_buffer"),
          {MakeBackingBasePtr(state),
           IntImm(DataType::Int(64), state.plan->count),
           IntImm(DataType::Int(64), state.plan->vector_lanes)});
      state.handle =
          Var(state.plan->fragment_mem_var->name_hint, DataType::Handle());
      binds.push_back(Bind(state.handle, decl));
      declared_handles_.insert(state.handle.get());
      handle_plans_.emplace(state.handle.get(), state.plan);
    }
    const auto *scope_attr = region.as<AttrStmtNode>();
    if (scope_attr != nullptr && scope_attr->attr_key == "tl.simdvf_scope") {
      return AttrStmt(scope_attr->node, scope_attr->attr_key,
                      scope_attr->value,
                      SeqStmt::Flatten(std::move(binds), scope_attr->body));
    }
    return SeqStmt::Flatten(std::move(binds), region);
  }

  // Post-check: every local.fragment access of the region went through the
  // slot path. A survivor means Verify and Vectorize disagree.
  void CheckNoResidualFragmentAccess(const Stmt &body) const {
    class ResidualFinder : public StmtExprVisitor {
    public:
      bool found = false;
      std::string name;
      void VisitExpr_(const BufferLoadNode *op) final {
        if (op->buffer.scope() == "local.fragment") {
          found = true;
          name = op->buffer->name;
        }
        StmtExprVisitor::VisitExpr_(op);
      }
      void VisitStmt_(const BufferStoreNode *op) final {
        if (op->buffer.scope() == "local.fragment") {
          found = true;
          name = op->buffer->name;
        }
        if (op->predicate.defined()) {
          VisitExpr(op->predicate.value());
        }
        StmtExprVisitor::VisitStmt_(op);
      }
    };
    ResidualFinder finder;
    finder(body);
    ICHECK(!finder.found)
        << "[VectorizeParallelToPTO] internal error: fragment buffer `"
        << finder.name << "` access survived conversion";
  }

  // Post-check for the emitted declarations: recover each handle's
  // backing / count / lanes from its declare_fragment_buffer Bind, then
  // verify that every use of the handle is a vload/vstore base whose
  // element offset addresses whole entries within [0, count * lanes) and
  // whose value dtype matches the declared element type. Any other use of
  // a handle is a conversion bug: handles are memory-object identities,
  // not generic pointers.
  void CheckSlotDeclarations(const Stmt &body) const {
    class DeclarationChecker : public StmtExprVisitor {
    public:
      DeclarationChecker(
          const std::unordered_map<const VarNode *, PtoSlotPlan> &handle_plans)
          : handle_plans_(handle_plans) {}

      bool found_declaration(const VarNode *var) const {
        return declared_.count(var) != 0;
      }

      void VisitStmt_(const BindNode *op) final {
        const auto *call = op->value.as<CallNode>();
        if (call != nullptr &&
            call->op.same_as(tl::vmi_declare_fragment_buffer())) {
          auto plan = handle_plans_.find(op->var.get());
          ICHECK(plan != handle_plans_.end())
              << "[VectorizeParallelToPTO] internal error: "
                 "declare_fragment_buffer binds the unexpected Var `"
              << op->var->name_hint << "`";
          const auto *count = call->args[1].as<IntImmNode>();
          const auto *lanes = call->args[2].as<IntImmNode>();
          ICHECK(count != nullptr && lanes != nullptr &&
                 count->value == plan->second->count &&
                 lanes->value == plan->second->vector_lanes)
              << "[VectorizeParallelToPTO] internal error: declaration of `"
              << op->var->name_hint << "` disagrees with its plan ("
              << (count ? std::to_string(count->value) : "?") << "x"
              << (lanes ? std::to_string(lanes->value) : "?") << " vs plan "
              << plan->second->count << "x" << plan->second->vector_lanes
              << ")";
          // The declared base must address the plan's own backing: the
          // address is access_ptr(BufferLoad(backing, 0), total, 3).
          const auto *base_call = call->args[0].as<CallNode>();
          const BufferLoadNode *base_load = nullptr;
          if (base_call != nullptr) {
            base_load = base_call->args[0].as<BufferLoadNode>();
          }
          ICHECK(base_load != nullptr &&
                 base_load->buffer->data.get() ==
                     plan->second->backing->data.get())
              << "[VectorizeParallelToPTO] internal error: declaration of `"
              << op->var->name_hint << "` does not address its plan's "
                 "backing";
          declared_.emplace(op->var.get(), plan->second);
        }
        StmtExprVisitor::VisitStmt_(op);
      }

      void VisitExpr_(const CallNode *op) final {
        const auto *op_node = op->op.as<OpNode>();
        const bool is_vload =
            op_node != nullptr && op_node->name == "tl.vmi.vload";
        const bool is_vstore =
            op_node != nullptr && op_node->name == "tl.vmi.vstore";
        if (is_vload || is_vstore) {
          const size_t base_i = is_vload ? 0 : 1;
          const size_t offset_i = base_i + 1;
          const auto *base = op->args[base_i].as<VarNode>();
          if (base != nullptr && declared_.count(base) != 0) {
            const PtoSlotPlan &plan = declared_.at(base);
            // Whole-entry accesses: the value lanes equal VL. vload carries
            // the vector in its result dtype, vstore in its value operand.
            const DataType value_dtype =
                is_vload ? op->dtype : op->args[0].dtype();
            ICHECK_EQ(value_dtype.lanes(), plan->vector_lanes)
                << "[VectorizeParallelToPTO] internal error: access on `"
                << base->name_hint << "` carries " << value_dtype.lanes()
                << " lanes, declaration has " << plan->vector_lanes;
            ICHECK_EQ(value_dtype.element_of(), plan->fragment->dtype)
                << "[VectorizeParallelToPTO] internal error: access on `"
                << base->name_hint << "` carries dtype "
                << value_dtype.element_of() << ", plan declares "
                << plan->fragment->dtype;
            CheckEntryOffset(op->args[offset_i], value_dtype.lanes(), plan,
                             base);
            // The remaining arguments are a plain offset/mask/value; only
            // the base is handle-special.
            for (size_t i = 0; i < op->args.size(); ++i) {
              if (i != base_i) {
                StmtExprVisitor::VisitExpr(op->args[i]);
              }
            }
            return;
          }
        }
        StmtExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const VarNode *op) final {
        ICHECK(declared_.count(op) == 0)
            << "[VectorizeParallelToPTO] internal error: fragment handle `"
            << op->name_hint
            << "` is used outside a vload/vstore base position";
      }

    private:
      // Whole-entry discipline: prove the element offset is a multiple of
      // the vector lanes. Simplification distributes the chunk geometry
      // into the product ((i*Q+q)*L canonicalizes to i*(Q*L)+q*L), so
      // matching the exact "x*L" shape rejects equivalent forms; prove
      // divisibility of the simplified expression instead.
      static bool IsWholeEntryMultiple(const PrimExpr &expr, int64_t lanes) {
        if (const int64_t *constant = as_const_int(expr)) {
          return *constant % lanes == 0;
        }
        if (const auto *mul = expr.as<MulNode>()) {
          const int64_t *lhs = as_const_int(mul->a);
          const int64_t *rhs = as_const_int(mul->b);
          if ((lhs != nullptr && *lhs % lanes == 0) ||
              (rhs != nullptr && *rhs % lanes == 0)) {
            return true;
          }
          return IsWholeEntryMultiple(mul->a, lanes) &&
                 IsWholeEntryMultiple(mul->b, lanes);
        }
        if (const auto *add = expr.as<AddNode>()) {
          return IsWholeEntryMultiple(add->a, lanes) &&
                 IsWholeEntryMultiple(add->b, lanes);
        }
        if (const auto *sub = expr.as<SubNode>()) {
          return IsWholeEntryMultiple(sub->a, lanes) &&
                 IsWholeEntryMultiple(sub->b, lanes);
        }
        return false;
      }

      void CheckEntryOffset(const PrimExpr &offset, int64_t size,
                            const PtoSlotPlan &plan, const VarNode *base) {
        const int64_t total = plan->count * plan->vector_lanes;
        if (const int64_t *constant = as_const_int(offset)) {
          // A constant must satisfy both bounds and the whole-entry
          // granularity — a non-multiple constant would tear an entry just
          // like a non-multiple dynamic offset.
          ICHECK(*constant >= 0 && *constant + size <= total &&
                 *constant % plan->vector_lanes == 0)
              << "[VectorizeParallelToPTO] internal error: access on `"
              << base->name_hint << "` at element offset " << *constant
              << "+size" << size << " leaves the declared [0, " << total
              << ") range or is not entry * lanes granularity";
          return;
        }
        // Dynamic offsets are entry-driven: entry * lanes, possibly folded
        // into sums whose terms keep the same granularity. The range itself
        // was proven by Verify's interval proof; here only the granularity
        // matters (a non-multiple remainder would tear an entry).
        ICHECK(IsWholeEntryMultiple(offset, plan->vector_lanes))
            << "[VectorizeParallelToPTO] internal error: access on `"
            << base->name_hint << "` uses offset " << offset
            << ", which is not entry * lanes granularity";
      }

      const std::unordered_map<const VarNode *, PtoSlotPlan> &handle_plans_;
      std::unordered_map<const VarNode *, PtoSlotPlan> declared_;
    };

    DeclarationChecker checker(handle_plans_);
    checker(body);
    for (const VarNode *handle : declared_handles_) {
      ICHECK(checker.found_declaration(handle))
          << "[VectorizeParallelToPTO] internal error: handle `"
          << handle->name_hint << "` lost its declaration";
    }
  }

  // Drop the consumed fragment allocations, their layout_map entries, the
  // consumed slot plans and their read/write regions from the rebuilt
  // block. The backing allocations stay: the emitted backing loads/stores
  // reference them, and the address pass chain rewrites their base.
  Stmt RebuildBlockWithoutConsumedFragments(const SBlockNode *op,
                                            const Stmt &new_body) {
    if (converted_fragments_.empty()) {
      return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint,
                    new_body, op->init, op->alloc_buffers, op->match_buffers,
                    op->annotations);
    }
    Array<Buffer> alloc_buffers;
    for (const Buffer &buffer : op->alloc_buffers) {
      if (converted_fragments_.count(buffer->data.get()) != 0) {
        continue;
      }
      alloc_buffers.push_back(buffer);
    }
    Array<BufferRegion> reads, writes;
    for (const BufferRegion &region : op->reads) {
      if (converted_fragments_.count(region->buffer->data.get()) == 0) {
        reads.push_back(region);
      }
    }
    for (const BufferRegion &region : op->writes) {
      if (converted_fragments_.count(region->buffer->data.get()) == 0) {
        writes.push_back(region);
      }
    }
    Map<String, Any> annotations;
    {
      // Copy through the block's annotation map, dropping the consumed
      // entries below.
      for (const auto &[key, value] : op->annotations) {
        annotations.Set(key, value);
      }
    }
    if (auto value = annotations.Get(attr::kLayoutMap)) {
      auto map = value.value().try_cast<LayoutMap>();
      if (map.has_value()) {
        LayoutMap kept;
        for (const auto &[buffer, layout] : map.value()) {
          if (converted_fragments_.count(buffer->data.get()) == 0) {
            kept.Set(buffer, layout);
          }
        }
        if (kept.empty()) {
          annotations.erase(attr::kLayoutMap);
        } else {
          annotations.Set(attr::kLayoutMap, kept);
        }
      }
    }
    if (auto value = annotations.Get(kPtoSlotPlan)) {
      auto plans = value.value().try_cast<Array<PtoSlotPlan>>();
      if (plans.has_value()) {
        Array<PtoSlotPlan> kept;
        for (const PtoSlotPlan &plan : plans.value()) {
          if (converted_fragments_.count(plan->fragment->data.get()) == 0) {
            kept.push_back(plan);
          }
        }
        if (kept.empty()) {
          annotations.erase(kPtoSlotPlan);
        } else {
          annotations.Set(kPtoSlotPlan, kept);
        }
      }
    }
    return SBlock(op->iter_vars, reads, writes, op->name_hint, new_body,
                  op->init, alloc_buffers, op->match_buffers, annotations);
  }

  // Replace one statement (the unit's For) with its conversion, recursing
  // through the same wrapper kinds DiscoverUnits understands.
  Stmt ReplaceStmt(const Stmt &body, const For &target,
                   const Stmt &replacement) {
    if (const auto *for_node = body.as<ForNode>()) {
      if (for_node == target.get()) {
        return replacement;
      }
      Stmt nb = ReplaceStmt(for_node->body, target, replacement);
      if (nb.same_as(for_node->body)) {
        return body;
      }
      For nf = GetRef<For>(for_node);
      nf.CopyOnWrite()->body = nb;
      return nf;
    }
    if (const auto *seq = body.as<SeqStmtNode>()) {
      Array<Stmt> new_seq;
      bool changed = false;
      for (const auto &s : seq->seq) {
        Stmt ns = ReplaceStmt(s, target, replacement);
        changed |= !ns.same_as(s);
        new_seq.push_back(ns);
      }
      return changed ? SeqStmt::Flatten(new_seq) : body;
    }
    if (const auto *attr = body.as<AttrStmtNode>()) {
      Stmt nb = ReplaceStmt(attr->body, target, replacement);
      if (nb.same_as(attr->body)) {
        return body;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, nb);
    }
    if (const auto *block = body.as<SBlockNode>()) {
      Stmt nb = ReplaceStmt(block->body, target, replacement);
      if (nb.same_as(block->body)) {
        return body;
      }
      return SBlock(block->iter_vars, block->reads, block->writes,
                    block->name_hint, nb, block->init, block->alloc_buffers,
                    block->match_buffers, block->annotations);
    }
    if (const auto *realize = body.as<SBlockRealizeNode>()) {
      // nb is the replaced *body* of the block (a SeqStmt/For/etc.), not an
      // SBlock: rebuild the block with the new body first.
      SBlock block = realize->block;
      Stmt nb = ReplaceStmt(block->body, target, replacement);
      if (nb.same_as(block->body)) {
        return body;
      }
      SBlock new_block =
          SBlock(block->iter_vars, block->reads, block->writes,
                 block->name_hint, nb, block->init, block->alloc_buffers,
                 block->match_buffers, block->annotations);
      return SBlockRealize(realize->iter_values, realize->predicate, new_block);
    }
    return body;
  }

  // ------------------------------------------------------------------
  // Loop shells (constant-bound T.serial inside T.Parallel, admitted by
  // Verify). AnalyzeLoopShell runs per unit before conversion. The shell
  // itself is always preserved as a scalar loop; when the canonical
  // carry pattern matches, the carried value is promoted to a local.var
  // vreg buffer (the UB round-trip is not order-safe on the current
  // PTOAS: no automatic mem_bar between the carried vstore and the next
  // iteration's vload). A shell whose body both reads and writes a
  // buffer through UB without matching the pattern would silently read
  // stale values, so it is rejected instead.
  // ------------------------------------------------------------------
  static void FlattenStmts(const Stmt &stmt, std::vector<Stmt> *out) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        FlattenStmts(s, out);
      }
      return;
    }
    out->push_back(stmt);
  }

  class ShellAccessCollector : public StmtExprVisitor {
  public:
    struct Access {
      const BufferNode *buf;
      Array<PrimExpr> indices;
      bool is_store;
    };
    std::vector<Access> accesses;
    std::vector<const VarNode *> loop_vars;
    void VisitStmt_(const BufferStoreNode *op) final {
      accesses.push_back({op->buffer.get(), op->indices, true});
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      accesses.push_back({op->buffer.get(), op->indices, false});
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const ForNode *op) final {
      loop_vars.push_back(op->loop_var.get());
      StmtExprVisitor::VisitStmt_(op);
    }
  };

  class BufferAccessCounter : public StmtExprVisitor {
  public:
    explicit BufferAccessCounter(const BufferNode *buf) : buf_(buf) {}
    void VisitStmt_(const BufferStoreNode *op) final {
      if (op->buffer.get() == buf_) {
        ++count_;
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) final {
      if (op->buffer.get() == buf_) {
        ++count_;
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    int count_ = 0;

  private:
    const BufferNode *buf_;
  };

  [[noreturn]] void ShellFail(const std::string &message) {
    LOG(FATAL) << "[VectorizeParallelToPTO] loop shell: " << message;
  }

  bool IsCarryAccess(const Buffer &buf, const Array<PrimExpr> &indices) const {
    if (!carry_src_.defined() || !buf->data.same_as(carry_src_->data)) {
      return false;
    }
    if (carry_lane_var_match_) {
      // Region-level promotion: every unit has its own Parallel loop var, so
      // the carried access is "this buffer, indexed by a single lane Var".
      return indices.size() == 1 && indices[0].as<VarNode>() != nullptr;
    }
    if (indices.size() != carry_indices_.size()) {
      return false;
    }
    for (size_t i = 0; i < indices.size(); ++i) {
      if (!ffi::StructuralEqual()(indices[i], carry_indices_[i])) {
        return false;
      }
    }
    return true;
  }

  void AnalyzeLoopShell(const Stmt &unit_body, const Stmt &region_body) {
    carry_seed_ = nullptr;
    carry_loop_ = nullptr;
    carry_src_ = tirx::Buffer();
    carry_vreg_ = tirx::Buffer();
    carry_indices_ = Array<PrimExpr>();
    carry_active_ = false;

    std::vector<Stmt> stmts;
    FlattenStmts(unit_body, &stmts);
    const ForNode *shell = nullptr;
    size_t shell_pos = 0;
    for (size_t i = 0; i < stmts.size(); ++i) {
      if (const auto *f = stmts[i].as<ForNode>()) {
        if (f->kind == ForKind::kSerial) {
          if (shell != nullptr) {
            ShellFail("only one T.serial loop per T.Parallel unit is "
                      "supported at this stage");
          }
          shell = f;
          shell_pos = i;
        }
      }
    }
    if (shell == nullptr) {
      return; // no shell in this unit
    }

    ShellAccessCollector collector;
    collector(shell->body);
    // Buffers the shell both reads and writes through memory — the
    // cross-iteration dependencies.
    std::unordered_map<const BufferNode *, std::pair<int, int>> counts;
    for (const auto &acc : collector.accesses) {
      // A per-lane scalar is a register, not a memory carry: its in-shell
      // update is promoted to the vreg by ConvertBody, so it never needs a
      // UB round-trip promotion (and must not be counted as the carried
      // buffer, whose post-shell isolation check would then reject the
      // legitimate reads of the final value).
      if (pto_analysis::IsLaneScalarBuffer(GetRef<Buffer>(acc.buf))) {
        continue;
      }
      // Fragment buffers follow the same rule as the region-level carry
      // analysis and the vecscope split: their in-shell read-modify-write
      // goes through the slot backing (toolchain-ordered), and promoting
      // them to a carry vreg would both split the slot read/write path and
      // leave the writeback storing a never-written vreg into the
      // fragment through the plain UB path (B2).
      if (GetRef<Buffer>(acc.buf).scope() == "local.fragment") {
        continue;
      }
      auto &c = counts[acc.buf];
      if (acc.is_store) {
        ++c.second;
      } else {
        ++c.first;
      }
    }
    const BufferNode *carried = nullptr;
    for (const auto &kv : counts) {
      if (kv.second.first > 0 && kv.second.second > 0) {
        if (carried != nullptr) {
          ShellFail("the T.serial loop reads and writes more than one "
                    "buffer; only a single carried buffer is supported at "
                    "this stage");
        }
        carried = kv.first;
      }
    }
    if (carried == nullptr) {
      // Shell without a memory-carried value: plain preservation has no
      // cross-iteration write->read through UB to order.
      return;
    }

    // Every access of the carried buffer inside the shell must use one
    // index set (structurally equal, independent of every loop
    // variable), with exactly one store.
    std::vector<const ShellAccessCollector::Access *> carried_accesses;
    for (const auto &acc : collector.accesses) {
      if (acc.buf == carried) {
        carried_accesses.push_back(&acc);
      }
    }
    int carried_stores = 0;
    for (const auto *acc : carried_accesses) {
      if (acc->is_store) {
        ++carried_stores;
      }
    }
    if (carried_stores != 1) {
      ShellFail("the carried buffer must have exactly one store inside the "
                "T.serial loop");
    }
    const Array<PrimExpr> &ref = carried_accesses[0]->indices;
    for (const auto *acc : carried_accesses) {
      if (acc->indices.size() != ref.size()) {
        ShellFail("all carried accesses must use the same index");
      }
      for (size_t i = 0; i < ref.size(); ++i) {
        if (!ffi::StructuralEqual()(acc->indices[i], ref[i])) {
          ShellFail("all carried accesses must use the same index");
        }
      }
    }
    std::vector<const VarNode *> loop_vars = collector.loop_vars;
    loop_vars.push_back(shell->loop_var.get());
    for (const auto &idx : ref) {
      for (const VarNode *lv : loop_vars) {
        if (tirx::UsesVar(idx, [lv](const VarNode *v) { return v == lv; })) {
          ShellFail("the carried index must not depend on the loop "
                    "variable");
        }
      }
    }

    // The seed: a store to the carried buffer, immediately before the
    // shell, whose value does not read the carried buffer.
    if (shell_pos == 0) {
      ShellFail("the loop carry needs a seed store to the carried buffer "
                "immediately before the T.serial loop");
    }
    const auto *seed = stmts[shell_pos - 1].as<BufferStoreNode>();
    if (seed != nullptr && seed->predicate.defined()) {
      ShellFail("the seed store of the carried buffer must be "
                "unconditional; a conditional seed cannot be promoted to a "
                "register and has no ordered UB form inside a vecscope");
    }
    if (seed == nullptr || seed->buffer.get() != carried ||
        seed->indices.size() != ref.size()) {
      ShellFail("the loop carry needs a seed store to the carried buffer "
                "immediately before the T.serial loop");
    }
    for (size_t i = 0; i < ref.size(); ++i) {
      if (!ffi::StructuralEqual()(seed->indices[i], ref[i])) {
        ShellFail("the seed store must use the same index as the carried "
                  "accesses");
      }
    }
    class SeedReadsCarried : public StmtExprVisitor {
    public:
      explicit SeedReadsCarried(const BufferNode *buf) : buf_(buf) {}
      void VisitExpr_(const BufferLoadNode *op) final {
        if (op->buffer.get() == buf_) {
          reads_ = true;
        }
        StmtExprVisitor::VisitExpr_(op);
      }
      bool reads_ = false;

    private:
      const BufferNode *buf_;
    };
    SeedReadsCarried seed_check(carried);
    seed_check(seed->value);
    if (seed_check.reads_) {
      ShellFail("the seed value must not read the carried buffer");
    }

    // Isolation: the carried buffer must not be accessed anywhere else
    // in the region (every other access would observe the eliminated
    // round-trip). The pattern's own accesses are the seed plus the
    // in-shell ones; the post-shell writeback is emitted, not counted.
    BufferAccessCounter region_count(carried);
    region_count(region_body);
    if (region_count.count_ !=
        static_cast<int>(carried_accesses.size()) + 1) {
      ShellFail("the carried buffer must only be accessed by the seed "
                "store and the T.serial loop in this SIMD_VF region");
    }

    carry_seed_ = seed;
    carry_loop_ = shell;
    carry_src_ = GetRef<Buffer>(carried);
    carry_indices_ = ref;
    carry_vreg_ = tirx::decl_buffer(
        {IntImm(DataType::Int(32), 1)},
        VectorDType(carried->dtype, lanes_),
        std::string(carried->name) + "_carry", "local.var");
  }

  Stmt WrapCarryAlloc(Stmt s) {
    if (carry_vreg_.defined()) {
      return SeqStmt::Flatten(AllocBuffer(carry_vreg_), std::move(s));
    }
    return s;
  }

  // ------------------------------------------------------------------
  // Per-lane scalar promotion (`T.alloc_var` in a Parallel unit)
  // ------------------------------------------------------------------
  // `T.alloc_var(dtype)` is a single-element `local.var` buffer. Every lane
  // owns its own copy, so the buffer is a per-lane register rather than a
  // lane-addressed memory object; Vectorize reads and writes it as a
  // `local.var` vector (vreg) so the value stays in RF.
  // Create (once per unit) the vreg backing a per-lane scalar.
  tirx::Buffer LaneScalarVreg(const Buffer &buffer) {
    auto it = lane_scalar_vregs_.find(buffer.get());
    if (it != lane_scalar_vregs_.end()) {
      return it->second;
    }
    tirx::Buffer vreg = tirx::decl_buffer(
        {IntImm(DataType::Int(32), 1)}, VectorDType(buffer->dtype, lanes_),
        std::string(buffer->name) + "_lane", "local.var");
    lane_scalar_vregs_[buffer.get()] = vreg;
    lane_scalar_allocs_.push_back(vreg);
    return vreg;
  }

  // Declare the per-lane vregs created for this unit inside it but outside
  // its shell, so a carried update binds to one SSA name. Reversed so the
  // first created vreg ends up outermost.
  Stmt DeclareLaneScalarVregs(Stmt s) {
    for (auto it = lane_scalar_allocs_.rbegin(); it != lane_scalar_allocs_.rend();
         ++it) {
      s = SeqStmt::Flatten(AllocBuffer(*it), std::move(s));
    }
    lane_scalar_allocs_.clear();
    if (unit_carry_from_region_) {
      carry_active_ = false;
      return s;
    }
    return WrapCarryAlloc(std::move(s));
  }

  // The shell's own stores were eliminated by promotion; publish the
  // final carry value back to the source buffer once, after the shell.
  Stmt EmitCarryWriteback() {
    ICHECK(carry_src_.scope() != "local.fragment")
        << "[VectorizeParallelToPTO] internal error: a local.fragment "
           "buffer must never be a carry source (see AnalyzeLoopShell)";
    auto access = pto_analysis::AnalyzeBufferAccess(
        carry_src_, carry_indices_, /*is_write=*/true, loop_var_,
        CurrentMapping(), &analyzer_, bind_env_);
    ICHECK(access.pattern == pto_analysis::AccessPattern::kContinuous)
        << "[VectorizeParallelToPTO] internal error: the carry writeback "
           "address is not continuous";
    Array<PrimExpr> start_idx;
    for (const auto &idx : access.start_indices) {
      start_idx.push_back(
          analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(idx))));
    }
    PrimExpr ptr =
        Call(DataType::Handle(), tl::access_ptr(),
             {BufferLoad(carry_src_, start_idx),
              IntImm(DataType::Int(32), static_cast<int>(lanes_)),
              IntImm(DataType::Int(32), 2)});
    return Evaluate(Call(
        DataType::Void(), VmiOp("vstore"),
        {BufferLoad(carry_vreg_, {make_zero(DataType::Int(32))}), ptr,
         IntImm(DataType::Int(32), 0), mask_var_}));
  }

  // ------------------------------------------------------------------
  // Region-level carry promotion (serial loop WRAPPING Parallel units).
  //
  // Inside one vecscope a UB read-modify-write chain has no ordering
  // point: every sync operation delimits a vecscope, so none can appear
  // between two iterations. An accumulator updated that way keeps reading
  // its seed or a partially updated value, and the outcome depends on the
  // trip count and on backend scheduling. Promoting the accumulator to a
  // local.var vreg makes the loop carry an SSA value (scf.for iter_args),
  // so no memory ordering is needed at all.
  //
  // Recognised shape (deliberately narrow, mirroring AnalyzeLoopShell):
  //   <unit S>: B[lane] = seed            (right before the loop, no B read)
  //   for t in serial(G):
  //     <unit U>: ... = B[lane] ... B[lane] = f(...)   (one store, same index)
  // with B a 1-D shared buffer of exactly one lane vector (lanes elements),
  // and no other unit in the region touching B.
  // ------------------------------------------------------------------
  static bool StmtHasFor(const Stmt &s, const ForNode *target) {
    bool found = false;
    PostOrderVisit(s, [&](const ObjectRef &n) {
      if (n.get() == target) {
        found = true;
      }
    });
    return found;
  }

  // Flatten the *top level* of a region body, looking through the AttrStmt
  // (simdvf scope) and SBlock wrappers that the region may carry.
  static void FlattenTopStmts(const Stmt &stmt, std::vector<Stmt> *out) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        FlattenTopStmts(s, out);
      }
      return;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      FlattenTopStmts(attr->body, out);
      return;
    }
    if (const auto *blk = stmt.as<SBlockNode>()) {
      FlattenTopStmts(blk->body, out);
      return;
    }
    out->push_back(stmt);
  }

  struct RegionCarry {
    bool active = false;
    const ForNode *loop = nullptr;
    const ForNode *seed = nullptr;
    tirx::Buffer src{};
    Array<PrimExpr> indices{};
    tirx::Buffer vreg{};
    PrimExpr seed_value{};
  };

  // Try to promote one candidate loop; returns true when the canonical
  // carried-buffer pattern matches.
  bool TryRegionCarry(const ForNode *loop, const std::vector<Stmt> &stmts,
                      const pto_analysis::PtoRegionScan &scan, int64_t lanes,
                      RegionCarry *out) {
    const ForNode *first = nullptr;
    for (const For &u : scan.units) {
      if (StmtHasFor(loop->body, u.get())) {
        first = u.as<ForNode>();
        break;
      }
    }
    if (first == nullptr) {
      return false;
    }

    // Carried buffer: the single shared buffer that the first in-loop unit
    // both reads and writes, with exactly one store.
    ShellAccessCollector acc;
    acc(static_cast<const Stmt &>(first->body));
    const BufferNode *carried = nullptr;
    Array<PrimExpr> indices;
    for (const auto &a : acc.accesses) {
      if (!a.is_store) {
        continue;
      }
      // Fragment buffers never participate in region carry promotion:
      // their value lives in the slot backing and is ordered by the
      // backing (toolchain-ordered), not by the carry vreg. Promoting a fragment
      // into a vreg while loads still go through the backing would
      // split the read/write path (B2).
      if (GetRef<Buffer>(a.buf).scope() == "local.fragment") {
        continue;
      }
      bool also_read = false;
      for (const auto &b : acc.accesses) {
        if (!b.is_store && b.buf == a.buf &&
            ffi::StructuralEqual()(b.indices, a.indices)) {
          also_read = true;
          break;
        }
      }
      if (!also_read) {
        continue;
      }
      if (carried != nullptr && carried != a.buf) {
        return false;  // more than one candidate
      }
      carried = a.buf;
      indices = a.indices;
    }
    if (carried == nullptr || indices.size() != 1 ||
        indices[0].as<VarNode>() == nullptr) {
      return false;
    }
    // Full lane width, single chunk.
    if (carried->shape.size() != 1) {
      return false;
    }
    const auto *extent = carried->shape[0].as<IntImmNode>();
    if (extent == nullptr || extent->value != lanes) {
      return false;
    }

    // Every in-loop unit updates the same buffer at a single lane index.
    for (const For &u : scan.units) {
      if (!StmtHasFor(loop->body, u.get())) {
        continue;
      }
      ShellAccessCollector ua;
      ua(static_cast<const Stmt &>(u.as<ForNode>()->body));
      // The carried index must be this unit's own lane variable. A bare Var is
      // not enough: `m_ub[t]` (t an enclosing serial variable) is also a
      // single-Var index and passes Verify as a lane-uniform load, but
      // redirecting it to the vreg would turn one lane's value into a
      // whole-vector read — a silent semantic error.
      std::set<const VarNode *> unit_vars;
      unit_vars.insert(u->loop_var.get());
      if (const auto *inner_par = u->body.as<ForNode>()) {
        if (inner_par->kind == ForKind::kParallel) {
          unit_vars.insert(inner_par->loop_var.get());
        }
      }
      int stores = 0;
      int loads = 0;
      for (const auto &a : ua.accesses) {
        if (a.buf != carried) {
          continue;
        }
        const auto *iv = a.indices.size() == 1 ? a.indices[0].as<VarNode>()
                                               : nullptr;
        if (iv == nullptr || unit_vars.count(iv) == 0) {
          return false;
        }
        a.is_store ? ++stores : ++loads;
      }
      if (stores != 1 || loads < 1) {
        return false;
      }
    }

    // Seed: the statement right before the loop writes B once, never reads it.
    size_t loop_pos = 0;
    bool found_loop = false;
    for (size_t i = 0; i < stmts.size(); ++i) {
      if (stmts[i].get() == loop) {
        loop_pos = i;
        found_loop = true;
        break;
      }
    }
    if (!found_loop || loop_pos == 0) {
      return false;
    }
    const auto *seed = stmts[loop_pos - 1].as<ForNode>();
    if (seed == nullptr) {
      return false;
    }
    ShellAccessCollector sa;
    sa(static_cast<const Stmt &>(seed->body));
    int seed_stores = 0;
    int seed_loads = 0;
    PrimExpr seed_value;
    for (const auto &a : sa.accesses) {
      if (a.buf != carried) {
        continue;
      }
      if (a.indices.size() != 1 ||
          a.indices[0].as<VarNode>() == nullptr) {
        return false;
      }
      if (a.is_store) {
        ++seed_stores;
        if (const auto *st = seed->body.as<BufferStoreNode>()) {
          seed_value = st->value;
        }
      } else {
        ++seed_loads;
      }
    }
    if (seed_stores != 1 || seed_loads != 0 || !seed_value.defined()) {
      return false;
    }
    // The seed is broadcast at region level, so it must be lane-invariant.
    bool seed_has_var = false;
    PostOrderVisit(seed_value, [&](const ObjectRef &n) {
      if (n.as<VarNode>() != nullptr) {
        seed_has_var = true;
      }
    });
    if (seed_has_var) {
      return false;
    }

    // Isolation: no other unit in the region may touch the carried buffer.
    for (const For &u : scan.units) {
      if (StmtHasFor(loop->body, u.get()) || u.get() == seed) {
        continue;
      }
      BufferAccessCounter counter(carried);
      counter(static_cast<const Stmt &>(u.as<ForNode>()->body));
      if (counter.count_ != 0) {
        return false;
      }
    }

    out->active = true;
    out->loop = loop;
    out->seed = seed;
    out->src = GetRef<Buffer>(carried);
    out->indices = indices;
    out->seed_value = seed_value;
    out->vreg = tirx::decl_buffer(
        {IntImm(DataType::Int(32), 1)},
        VectorDType(GetRef<Buffer>(carried)->dtype, lanes),
        std::string(carried->name) + "_rcarry", "local.var");
    return true;
  }

  void AnalyzeRegionCarry(const Stmt &region,
                          const pto_analysis::PtoRegionScan &scan,
                          int64_t lanes) {
    region_carry_ = RegionCarry();
    if (lanes <= 0) {
      return;
    }
    std::vector<Stmt> stmts;
    FlattenTopStmts(region, &stmts);
    // Several serial loops may wrap parallel units (e.g. a fill loop
    // followed by an accumulate loop): exactly one of them must match the
    // carried-buffer pattern, otherwise the region is left unpromoted.
    std::vector<const ForNode *> candidates;
    for (const Stmt &s : stmts) {
      const auto *f = s.as<ForNode>();
      if (f == nullptr || f->kind != ForKind::kSerial) {
        continue;
      }
      for (const For &u : scan.units) {
        if (StmtHasFor(f->body, u.get())) {
          candidates.push_back(f);
          break;
        }
      }
    }
    for (const ForNode *cand : candidates) {
      RegionCarry rc;
      if (!TryRegionCarry(cand, stmts, scan, lanes, &rc)) {
        continue;
      }
      if (region_carry_.active) {
        region_carry_ = RegionCarry();  // ambiguous: promote nothing
        return;
      }
      region_carry_ = rc;
    }
    if (region_carry_.active) {
      VLOG(1) << "[rc] ACCEPT carry buffer " << region_carry_.src->name;
    }
  }

  Stmt EmitRegionCarryWriteback(int64_t lanes) {
    PrimExpr ptr = Call(
        DataType::Handle(), tl::access_ptr(),
        {BufferLoad(region_carry_.src, {make_zero(DataType::Int(32))}),
         IntImm(DataType::Int(32), static_cast<int>(lanes)),
         IntImm(DataType::Int(32), 2)});
    return Evaluate(Call(
        DataType::Void(), VmiOp("vstore"),
        {BufferLoad(region_carry_.vreg, {make_zero(DataType::Int(32))}), ptr,
         IntImm(DataType::Int(32), 0), shared_mask_}));
  }

  // ------------------------------------------------------------------
  // Per-unit conversion
  // ------------------------------------------------------------------
  // Ranges of the serial loops enclosing a unit inside the region, for
  // the fragment entry-interval proof: an entry like f[k, j] with k a
  // region-level serial var is only provable when k's range is bound, and
  // Verify bound exactly these ranges when it accepted the program — so
  // the same set must be reproducible here or Vectorize would reject a
  // verified access. Only constant-bound loops bind; non-constant ones
  // were already rejected for fragment accesses by the failing proof in
  // Verify. The shared helper is also what the fragment layout solver
  // binds, keeping both proofs on the same domain.
  static Map<Var, Range> EnclosingSerialRanges(const Stmt &stmt,
                                               const ForNode *unit) {
    return pto_analysis::CollectEnclosingSerialRanges(stmt, unit);
  }

  Stmt ConvertUnit(const For &loop,
                   const pto_analysis::PtoInverseMapping &mapping,
                   const Stmt &region_body) {
    // The inverse mapping comes from the region entry: it is the shared
    // source of truth for lanes, the mask plan and the body conversion
    // (see InvertPtoLaneLayout).
    // Reset per-unit conversion state.
    bind_env_.clear();
    value_map_.clear();
    tmp_counter_ = 0;
    lane_scalar_vregs_.clear();
    lane_scalar_allocs_.clear();
    outer_loop_ranges_ = EnclosingSerialRanges(region_body, loop.get());
    // Serial loops in scope outside the region (kernel level): the region
    // body does not contain them, but fragment entries may still depend on
    // their variables (f[b, j]), and Verify proved the interval against the
    // original tree — with these ranges bound. Seed them so Vectorize
    // reproduces the same proof.
    for (const auto &kv : outer_serial_ranges_) {
      outer_loop_ranges_.Set(kv.first, kv.second);
    }
    // Keep the complete mapping (2D fields included) for the
    // per-access analysis calls in ConvertBody.
    unit_mapping_ = mapping;

    // For a 2D unit the vectorized coordinate is the selected
    // dimension; the other one stays a serial loop var. The converted
    // body is the innermost compute body.
    const ForNode *inner = nullptr;
    if (const auto *body_for = loop->body.as<ForNode>()) {
      if (body_for->kind == ForKind::kParallel) {
        inner = body_for;
      }
    }
    const Var vector_var =
        mapping.is_2d
            ? (mapping.select_inner ? mapping.inner_var : mapping.outer_var)
            : loop->loop_var;
    loop_var_ = vector_var;
    lanes_ = mapping.lanes;
    // The chunk loop covers the padded chunk count Q = P / L
    // (mapping.chunk_count), which includes the tail chunk of a
    // non-divisible extent.
    nchunks_ = mapping.chunk_count;
    chunk_var_ = Var("q", vector_var.dtype());
    lane_var_ = mapping.lane_var; // analysis placeholder; never escapes
    chunk_analysis_ = mapping.chunk_var;

    // The substitution loop_var -> chunk_analysis_ * L + lane_var_ drives
    // all address/index rewriting (same expression the shared helper
    // uses).
    lane_indexed_expr_ = mapping.index_expr;

    // A divisible extent uses the caller's shared full mask (constant
    // across chunks). A non-divisible extent gets a per-unit scalar
    // `remaining` that starts at the logical extent E and decreases by L
    // per chunk; each chunk binds a dynamic create_mask(remaining, L)
    // before the body and updates remaining at the chunk end. The tail
    // chunk's mask zeroes the lanes past E, which protects the store
    // (loads still read the full padded range — the caller guarantees
    // its readability; the store mask never widens a write past E).
    // The mask var must exist before the body conversion: the
    // converted vstore/vadd reference it while the body is built.
    const bool has_tail = mapping.tail_lanes != 0;
    if (!has_tail) {
      mask_var_ = shared_mask_;
    } else {
      // decl_buffer shapes the Var with a local.var pointer type; the
      // codegen reads the scope from there and emits the scalar
      // local-variable form the PTODSL surface expects.
      remaining_buffer_ =
          tirx::decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Int(32),
                            "remaining", "local.var");
      // The per-chunk mask bind is emitted inside the chunk loop below;
      // this var is what it binds and what the body references.
      mask_var_ = Var("mask", DataType::Bool(static_cast<int>(lanes_)));
    }

    // Convert the unit body statements in order. For a 2D unit this is
    // the innermost compute body; the outer (kept) coordinate is
    // re-emitted as a serial loop around the chunk loop.
    Array<Stmt> unit_stmts;
    ConvertBody(inner != nullptr ? static_cast<const Stmt &>(inner->body)
                                 : static_cast<const Stmt &>(loop->body),
                &unit_stmts);

    // Drop binds whose value is never used (address-only aliases and
    // unused source binds strand vci/vbrc/vadd/vload chains) before the
    // chunk loop is built.
    Stmt eliminated = DeadBindElim()(SeqStmt::Flatten(unit_stmts));
    Array<Stmt> chunk_seq;
    if (!has_tail) {
      chunk_seq.push_back(std::move(eliminated));
    } else {
      // Per-chunk mask bind + body + remaining update, in that order.
      // mask_var_ was created before the body conversion.
      PrimExpr remaining_load =
          BufferLoad(remaining_buffer_, {make_zero(DataType::Int(32))});
      PrimExpr mask_value =
          Call(DataType::Bool(static_cast<int>(lanes_)), VmiOp("create_mask"),
               {remaining_load},
               {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes_))}});
      chunk_seq.push_back(Bind(mask_var_, mask_value));
      chunk_seq.push_back(std::move(eliminated));
      PrimExpr next_remaining =
          remaining_load -
          IntImm(DataType::Int(32), static_cast<int64_t>(lanes_));
      chunk_seq.push_back(BufferStore(remaining_buffer_, next_remaining,
                                      {make_zero(DataType::Int(32))}));
    }
    Stmt chunk_body = SeqStmt::Flatten(chunk_seq);
    Stmt chunk_loop =
        For(chunk_var_, make_zero(chunk_var_->dtype),
            IntImm(chunk_var_->dtype, static_cast<int64_t>(nchunks_)),
            ForKind::kSerial, chunk_body);
    if (inner == nullptr) {
      if (!has_tail) {
        return DeclareLaneScalarVregs(chunk_loop);
      }
      // remaining lives across the whole chunk loop: declare it, set it
      // to E once before the loop.
      Stmt init =
          BufferStore(remaining_buffer_,
                      IntImm(DataType::Int(32),
                             static_cast<int64_t>(mapping.logical_extent)),
                      {make_zero(DataType::Int(32))});
      return DeclareLaneScalarVregs(SeqStmt::Flatten(
          AllocBuffer(remaining_buffer_), SeqStmt::Flatten(init, chunk_loop)));
    }
    // 2D: keep the non-vectorized coordinate as the outer
    // serial loop; the vectorized dimension becomes the chunk loop.
    const Var kept_var =
        mapping.select_inner ? loop->loop_var : inner->loop_var;
    const int64_t kept_extent =
        mapping.select_inner ? mapping.outer_extent : mapping.inner_extent;
    Stmt kept_body = chunk_loop;
    if (has_tail) {
      // remaining resets to E inside the kept loop: every row starts a
      // fresh recursion, so rows are independent and the tail chunk of
      // each row masks exactly that row's tail lanes.
      Stmt init =
          BufferStore(remaining_buffer_,
                      IntImm(DataType::Int(32),
                             static_cast<int64_t>(mapping.logical_extent)),
                      {make_zero(DataType::Int(32))});
      kept_body = SeqStmt::Flatten(init, chunk_loop);
    }
    Stmt kept_loop =
        For(kept_var, make_zero(kept_var->dtype),
            IntImm(kept_var->dtype, static_cast<int64_t>(kept_extent)),
            ForKind::kSerial, kept_body);
    Stmt result =
        has_tail ? SeqStmt::Flatten(AllocBuffer(remaining_buffer_), kept_loop)
                 : kept_loop;
    return DeclareLaneScalarVregs(std::move(result));
  }

  // Substitute the analysis chunk var with the real loop var q (and keep
  // the lane placeholder for lane-binding steps).
  PrimExpr BindChunkToQ(const PrimExpr &expr) const {
    class Replacer : public ExprMutator {
    public:
      Replacer(const VarNode *from, const Var &to) : from_(from), to_(to) {}
      PrimExpr VisitExpr_(const VarNode *op) final {
        if (op == from_) {
          return to_;
        }
        return ffi::GetRef<PrimExpr>(op);
      }

    private:
      const VarNode *from_;
      Var to_;
    };
    return Replacer(chunk_analysis_.get(), chunk_var_)(expr);
  }

  // Rewrite Var references inside metadata objects through value_map_
  // Follows containers (Array/Map) structurally; PrimExprs get a
  // var substitution.
  ObjectRef RemapMetadataVars(const ObjectRef &obj) const {
    if (!obj.defined()) {
      return obj;
    }
    if (const auto *expr = obj.as<PrimExprNode>()) {
      class RemapVars : public ExprMutator {
      public:
        RemapVars(const Map<Var, Var> &vmap) : vmap_(vmap) {}
        PrimExpr VisitExpr_(const VarNode *op) final {
          auto it = vmap_.find(GetRef<Var>(op));
          if (it != vmap_.end()) {
            return (*it).second;
          }
          return GetRef<PrimExpr>(op);
        }

      private:
        const Map<Var, Var> &vmap_;
      };
      return RemapVars(value_map_)(GetRef<PrimExpr>(expr));
    }
    // Containers are rejected by Verify (metadata must be a PrimExpr);
    // reaching here with one is an internal error.
    ICHECK(obj.as<PrimExprNode>() == nullptr)
        << "[VectorizeParallelToPTO] internal error: container metadata "
           "passed Verify";
    return obj;
  }

  PrimExpr SubstituteLaneZero(const PrimExpr &expr) const {
    class ZeroLane : public ExprMutator {
    public:
      ZeroLane(const Var &lane) : lane_(lane) {}
      PrimExpr VisitExpr_(const VarNode *op) final {
        if (op == lane_.get()) {
          return make_zero(lane_->dtype);
        }
        return ffi::GetRef<PrimExpr>(op);
      }

    private:
      Var lane_;
    };
    return ZeroLane(lane_var_)(expr);
  }

  // ------------------------------------------------------------------
  // Statement conversion (pre-binds first, then the statement)
  // ------------------------------------------------------------------
  void ConvertBody(const Stmt &body, Array<Stmt> *out) {
    if (const auto *seq = body.as<SeqStmtNode>()) {
      for (const auto &s : seq->seq) {
        ConvertBody(s, out);
      }
      return;
    }
    if (const auto *bind = body.as<BindNode>()) {
      // Whitelist is Verify's job; defensive check here.
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(bind->value, &pre);
      // Pre-binds first, then the Bind itself.
      for (const auto &s : pre) {
        out->push_back(s);
      }
      Var new_var(bind->var->name_hint, value.dtype());
      bind_env_.Set(bind->var, bind->value); // analysis environment
      value_map_.Set(bind->var, new_var);    // value environment
      out->push_back(Bind(new_var, value));
      return;
    }
    if (const auto *store = body.as<BufferStoreNode>()) {
      // A one-sided conditional write never reaches the vectorizer as a
      // predicated store: LegalizeParallelToPTO turns it into a value choice
      // against the destination's old value, and Verify rejects a store that
      // still carries a predicate.
      ICHECK(!store->predicate.defined())
          << "[VectorizeParallelToPTO] internal error: predicated store passed "
             "Verify; predicated stores must be rejected by "
             "VerifyParallelToPTO";
      // Per-lane scalar (T.alloc_var): the whole-buffer store becomes a write
      // to the vreg. A scalar value is broadcast over the lanes; a vector
      // value (a lane-varying load) is stored as-is.
      if (pto_analysis::IsLaneScalarBuffer(store->buffer)) {
        Array<Stmt> pre;
        PrimExpr value = ConvertExpr(store->value, &pre);
        if (value.dtype().lanes() == 1) {
          value = BroadcastToLanes("vbrc_lane", value, &pre);
        }
        tirx::Buffer vreg = LaneScalarVreg(store->buffer);
        for (const auto &s : pre) {
          out->push_back(s);
        }
        out->push_back(BufferStore(vreg, value,
                                   {make_zero(DataType::Int(32))}));
        return;
      }
      if (carry_seed_ != nullptr && store == carry_seed_ &&
          store->buffer.scope() != "local.fragment") {
        // Seed of the promoted carry: forward the value into the vreg
        // buffer instead of storing to UB, so the shell's loads never
        // see a round-trip.
        Array<Stmt> pre;
        PrimExpr value = ConvertExpr(store->value, &pre);
        if (value.dtype().lanes() == 1) {
          value = BroadcastToLanes("vbrc_val", value, &pre);
        }
        for (const auto &s : pre) {
          out->push_back(s);
        }
        out->push_back(BufferStore(carry_vreg_, value,
                                   {make_zero(DataType::Int(32))}));
        return;
      }
      if (carry_active_ && store->buffer.scope() != "local.fragment" &&
          IsCarryAccess(store->buffer, store->indices)) {
        // Carry update inside the shell: write the vreg buffer instead
        // of a vstore. Unwrap a trailing pure Bind so the emitted
        // assignment keeps the direct `acc = f(acc, ...)` form the
        // PTODSL rewriter turns into scf.for iter_args.
        Array<Stmt> pre;
        PrimExpr value = ConvertExpr(store->value, &pre);
        if (const auto *var = value.as<VarNode>()) {
          if (!pre.empty()) {
            const Stmt &last = pre.back();
            if (const auto *b = last.as<BindNode>()) {
              if (b->var.get() == var) {
                Array<Stmt> kept;
                for (size_t i = 0; i + 1 < pre.size(); ++i) {
                  kept.push_back(pre[i]);
                }
                pre = kept;
                value = b->value;
              }
            }
          }
        }
        if (value.dtype().lanes() == 1) {
          // Region-level promotion: the seed unit stores a scalar constant
          // into the vector carrier, so broadcast it first.
          value = BroadcastToLanes("vbrc_val", value, &pre);
        }
        for (const auto &s : pre) {
          out->push_back(s);
        }
        out->push_back(BufferStore(carry_vreg_, value,
                                   {make_zero(DataType::Int(32))}));
        return;
      }
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(store->value, &pre);
      // One-sided conditional writes never arrive here as predicates:
      // LegalizeParallelToPTO lowers them to a value choice against the
      // destination's old value, and Verify rejects a store that still
      // carries one. The chunk's valid mask is the whole store mask.
      PrimExpr store_mask = mask_var_;
      if (store->buffer.scope() == "local.fragment") {
        ConvertFragmentStore(store, value, pre, out, store_mask);
        return;
      }
      auto access = pto_analysis::AnalyzeBufferAccess(
          store->buffer, store->indices, /*is_write=*/true, loop_var_,
          CurrentMapping(), &analyzer_, bind_env_);
      if (access.pattern == pto_analysis::AccessPattern::kContinuous) {
        if (value.dtype().lanes() == 1) {
          value = BroadcastToLanes("vbrc_val", value, &pre);
        }
        Array<PrimExpr> start_idx;
        for (const auto &idx : access.start_indices) {
          start_idx.push_back(
              analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(idx))));
        }
        PrimExpr ptr =
            Call(DataType::Handle(), tl::access_ptr(),
                 {BufferLoad(store->buffer, start_idx),
                  IntImm(DataType::Int(32), static_cast<int>(lanes_)),
                  IntImm(DataType::Int(32), 2)});
        for (const auto &s : pre) {
          out->push_back(s);
        }
        out->push_back(Evaluate(
            Call(DataType::Void(), VmiOp("vstore"),
                 {value, ptr, IntImm(DataType::Int(32), 0), store_mask})));
      } else {
        // Verify rejects lane-uniform stores (the current
        // support boundary), so this branch is unreachable. Treat any
        // survivor as an internal error, same style as the wrapper
        // checks above.
        ICHECK(false)
            << "[VectorizeParallelToPTO] internal error: lane-uniform store `"
            << store->buffer->name
            << "` passed Verify; lane-uniform stores must be rejected by "
               "VerifyParallelToPTO";
      }
      return;
    }
    if (const auto *attr = body.as<AttrStmtNode>()) {
      // Verify guarantees the metadata has no lane-dependent semantics
      // (direct or via bind aliases). Rewrite its Var references through
      // value_map_ so already-converted binds are not left dangling
      // The body is converted underneath.
      Array<Stmt> stmts;
      // The body is a nested statement scope — keep the
      // source-bind and value maps scoped with it, so a bind defined
      // inside cannot remap a later sibling's reference (Verify rejects
      // that input; this keeps the transformation state consistent with
      // the unit visitor's scope handling).
      Map<Var, PrimExpr> saved_env = bind_env_;
      Map<Var, Var> saved_map = value_map_;
      ConvertBody(attr->body, &stmts);
      bind_env_ = std::move(saved_env);
      value_map_ = std::move(saved_map);
      Stmt inner = SeqStmt::Flatten(stmts);
      // Default to the *original* node. Only PrimExpr metadata is
      // remapped; a non-object Any (int/str atom metadata) must forward
      // unchanged — the old code wrote back a default-constructed (empty)
      // node when the try_cast failed.
      ffi::Any node = attr->node;
      if (auto ne = attr->node.try_cast<ObjectRef>()) {
        if (const auto *pe = ne.value().as<PrimExprNode>()) {
          node = RemapMetadataVars(GetRef<PrimExpr>(pe));
        }
      }
      PrimExpr value = attr->value;
      if (const auto *ve = attr->value.as<PrimExprNode>()) {
        value = RemapMetadataVars(GetRef<PrimExpr>(ve)).as<PrimExpr>().value();
      }
      out->push_back(AttrStmt(node, attr->attr_key, value, inner));
      return;
    }
    if (const auto *eval = body.as<EvaluateNode>()) {
      // Convert the expression with the same machinery so no stale loop
      // variable survives; the whitelist is Verify's job.
      Array<Stmt> pre;
      PrimExpr value = ConvertExpr(eval->value, &pre);
      for (const auto &s : pre) {
        out->push_back(s);
      }
      out->push_back(Evaluate(value));
      return;
    }
    if (const auto *loop = body.as<ForNode>()) {
      ICHECK(loop->kind == ForKind::kSerial)
          << "[VectorizeParallelToPTO] internal error: non-serial loop "
             "passed Verify";
      // Loop-shell preservation: a serial loop admitted by Verify is
      // copied through as a scalar shell; the body is converted with the
      // same statement machinery. Binds defined inside stay scoped to
      // the shell (same discipline as the AttrStmt branch). Verify has
      // pinned min/extent to constants and the step to 1, so the shell
      // is rebuilt with default step semantics.
      Array<Stmt> inner;
      Map<Var, PrimExpr> saved_env = bind_env_;
      Map<Var, Var> saved_map = value_map_;
      bool saved_carry = carry_active_;
      Map<Var, Range> saved_ranges = outer_loop_ranges_;
      if (carry_loop_ != nullptr && loop == carry_loop_) {
        carry_active_ = true;
      }
      // Verify admitted the shell with constant bounds; bind them so the
      // fragment entry-interval proof sees the same ranges it did.
      if (loop->min.as<IntImmNode>() != nullptr &&
          loop->extent.as<IntImmNode>() != nullptr) {
        outer_loop_ranges_.Set(
            loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      }
      // In-shell backing accesses are ordered by the toolchain's
      // automatic vecscope barriers (same policy as the unit level).
      ConvertBody(loop->body, &inner);
      outer_loop_ranges_ = std::move(saved_ranges);
      carry_active_ = saved_carry;
      bind_env_ = std::move(saved_env);
      value_map_ = std::move(saved_map);
      out->push_back(For(loop->loop_var, loop->min, loop->extent,
                         ForKind::kSerial, SeqStmt::Flatten(inner)));
      return;
    }
    LOG(FATAL) << "[VectorizeParallelToPTO] internal error: unsupported "
                  "statement passed Verify: "
               << body->GetTypeKey();
  }

  pto_analysis::PtoInverseMapping CurrentMapping() const {
    // Return the *complete* inverse mapping saved for this unit.
    // Rebuilding a 1D-only mapping here dropped is_2d, the two loop vars,
    // their extents and select_inner, so the shared access proof lost the
    // kept coordinate's range and Vectorize rejected units Verify had
    // accepted. The analysis placeholders (chunk/lane vars) are exactly
    // the ones saved from InvertPtoLaneLayout; only the emitted chunk
    // loop var q is separate (see BindChunkToQ).
    return unit_mapping_;
  }

  // ------------------------------------------------------------------
  // PTO slot storage conversion (fragment buffer -> vload/vstore on the
  // declaration handle). The entry index comes from the shared fragment
  // analysis: lane bound to 0 (entries are whole-vector), chunk bound to
  // the emitted q. Reads/writes use the handle declared once per VF as the
  // base and element_offset = entry * L; the declaration address already
  // carries the backing offset, so the access offset must not re-add it.
  // The physical residency (register forwarding vs. actual UB traffic) is
  // PTOAS's mem-to-value decision; TileLang owns the reservation, and
  // the ordering is the toolchain's via PR307's automatic vecscope
  // barriers (no TileLang-side mem_bar is emitted; the coverage
  // envelope of PR307's modeling is the documented toolchain floor).
  // ------------------------------------------------------------------
  const SlotState &RequireSlotState(const Buffer &buffer) {
    auto it = slot_map_.find(buffer);
    ICHECK(it != slot_map_.end())
        << "[VectorizeParallelToPTO] internal error: fragment buffer `"
        << buffer->name
        << "` passed Verify without a slot plan in Vectorize's scope";
    ICHECK(it->second.handle.defined())
        << "[VectorizeParallelToPTO] internal error: fragment buffer `"
        << buffer->name
        << "` is accessed before its declare_fragment_buffer was emitted";
    converted_fragments_.insert(buffer->data.get());
    return it->second;
  }

  PrimExpr BackingEntryOffset(const pto_analysis::FragmentAccess &access) {
    PrimExpr entry =
        analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(access.entry_expr)));
    return analyzer_.Simplify(entry *
                              IntImm(entry.dtype(), static_cast<int>(lanes_)));
  }

  PrimExpr ConvertFragmentLoad(const BufferLoadNode *load, Array<Stmt> *pre) {
    const SlotState &state = RequireSlotState(load->buffer);
    auto access = pto_analysis::AnalyzeFragmentAccess(
        load->buffer, load->indices, /*is_write=*/false, loop_var_,
        CurrentMapping(), state.layout, &analyzer_, bind_env_,
        outer_loop_ranges_);
    // No CSE and no cross-access reuse: every textual fragment read is its
    // own vload on the handle (memory semantics; merging and the
    // store->load forwarding are PTOAS's decisions).
    return EmitBind(
        "vload",
        Call(VectorDType(load->dtype, lanes_), VmiOp("vload"),
             {state.handle, BackingEntryOffset(access)},
             {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
        pre);
  }

  void ConvertFragmentStore(const BufferStoreNode *store, PrimExpr value,
                            Array<Stmt> &pre, Array<Stmt> *out,
                            const PrimExpr &store_mask) {
    const SlotState &state = RequireSlotState(store->buffer);
    auto access = pto_analysis::AnalyzeFragmentAccess(
        store->buffer, store->indices, /*is_write=*/true, loop_var_,
        CurrentMapping(), state.layout, &analyzer_, bind_env_,
        outer_loop_ranges_);
    // Backing writes are ordinary masked stores: the active lanes of the
    // mask (chunk mask, optionally narrowed by a recovered store
    // predicate) are updated and every other lane keeps the backing's
    // previous contents (the whole-entry slot semantics). A scalar RHS
    // broadcasts first, same rule as the continuous path.
    if (value.dtype().lanes() == 1) {
      value = EmitBind(
          "vbrc_val",
          Call(VectorDType(value.dtype(), lanes_), VmiOp("vbrc"), {value},
               {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
          &pre);
    }
    for (const auto &s : pre) {
      out->push_back(s);
    }
    out->push_back(Evaluate(Call(
        DataType::Void(), VmiOp("vstore"),
        {value, state.handle, BackingEntryOffset(access), store_mask})));
  }

  // ------------------------------------------------------------------
  // Expression conversion (index vars resolve through bind_env_)
  // ------------------------------------------------------------------
  PrimExpr ConvertExpr(const PrimExpr &expr, Array<Stmt> *pre) {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      // Per-lane scalar (T.alloc_var): read the vreg that carries its value.
      if (pto_analysis::IsLaneScalarBuffer(load->buffer)) {
        return BufferLoad(LaneScalarVreg(load->buffer),
                          {make_zero(DataType::Int(32))});
      }
      if (load->buffer.scope() == "local.fragment") {
        return ConvertFragmentLoad(load, pre);
      }
      if (carry_active_ && load->buffer.scope() != "local.fragment" &&
          IsCarryAccess(load->buffer, load->indices)) {
        // Promoted carry read: the value lives in the vreg buffer.
        return BufferLoad(carry_vreg_, {make_zero(DataType::Int(32))});
      }
      auto access = pto_analysis::AnalyzeBufferAccess(
          load->buffer, load->indices, /*is_write=*/false, loop_var_,
          CurrentMapping(), &analyzer_, bind_env_);
      if (access.pattern == pto_analysis::AccessPattern::kContinuous) {
        Array<PrimExpr> start_idx;
        for (const auto &idx : access.start_indices) {
          start_idx.push_back(
              analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(idx))));
        }
        PrimExpr ptr =
            Call(DataType::Handle(), tl::access_ptr(),
                 {BufferLoad(load->buffer, start_idx),
                  IntImm(DataType::Int(32), static_cast<int>(lanes_)),
                  IntImm(DataType::Int(32), 1)});
        // VMI with a return value goes through EmitBind.
        return EmitBind("vload",
                        Call(VectorDType(load->dtype, lanes_), VmiOp("vload"),
                             {ptr, IntImm(DataType::Int(32), 0)},
                             {{"size", IntImm(DataType::Int(32),
                                              static_cast<int>(lanes_))}}),
                        pre);
      }
      // lane-uniform scalar load: keep the BufferLoad (lane bound to 0,
      // chunk bound to q — indices resolve through the bind env
      // inside the shared helper already) and bind it to a scalar Var in
      // the current chunk; later uses (vbrc broadcast, scalar arithmetic,
      // store values) reference that Var. Each textual occurrence binds
      // separately — no cross-access load reuse. The Var dtype is the
      // buffer's own scalar dtype.
      Array<PrimExpr> indices;
      for (const auto &idx : access.indices_after_inverse) {
        indices.push_back(
            analyzer_.Simplify(BindChunkToQ(SubstituteLaneZero(idx))));
      }
      return EmitBind("sload", BufferLoad(load->buffer, indices), pre);
    }
    if (const auto *var = expr.as<VarNode>()) {
      auto it = value_map_.find(GetRef<Var>(var));
      if (it != value_map_.end()) {
        return (*it).second;
      }
      if (var == loop_var_.get()) {
        // lane_indexed used as a value: explicit vci vector.
        PrimExpr base = analyzer_.Simplify(
            BindChunkToQ(SubstituteLaneZero(lane_indexed_expr_)));
        return EmitBind(
            "vci",
            Call(VectorDType(var->dtype, lanes_), VmiOp("vci"), {base},
                 {{"size",
                   IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
            pre);
      }
      return GetRef<PrimExpr>(var);
    }
    if (expr.as<IntImmNode>() || expr.as<FloatImmNode>()) {
      return expr;
    }
    if (const auto *add = expr.as<AddNode>()) {
      return ConvertBinary(add->a, add->b, "vadd", pre);
    }
    if (const auto *sub = expr.as<SubNode>()) {
      return ConvertBinary(sub->a, sub->b, "vsub", pre);
    }
    if (const auto *mul = expr.as<MulNode>()) {
      return ConvertBinary(mul->a, mul->b, "vmul", pre);
    }
    if (const auto *div = expr.as<DivNode>()) {
      return ConvertBinary(div->a, div->b, "vdiv", pre);
    }
    if (const auto *call = expr.as<CallNode>()) {
      // Lane-mask boolean combinations come first. They are not unary
      // math, so the strict unary-math mapping below must not see them
      // (it only covers fabs/exp/log/sqrt). The frontend infinity()
      // constant never arrives here: Legalize rewrites it to FloatImm.
      std::string name;
      if (const auto *op_node = call->op.as<OpNode>()) {
        name = op_node->name;
      }
      if (const char *mask_op = pto_analysis::MaskLogicVmiOpName(name)) {
        const bool is_not = std::string_view(mask_op) == "vnot";
        ICHECK_EQ(call->args.size(), is_not ? 1 : 2);
        if (is_not) {
          PrimExpr a = ConvertExpr(call->args[0], pre);
          DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
          return EmitBind("mnot", Call(mask_ty, VmiOp("vnot"), {a}), pre);
        }
        return ConvertMaskLogic(call->args[0], call->args[1], mask_op, pre);
      }
      // Hardware-backed unary math (fabs/exp/log/sqrt); the dtype matrix
      // mirrors the ISA documentation (see IsSupportedVectorUnaryOp).
      auto call_op = call->op.as<Op>();
      ICHECK(call_op.has_value())
          << "[VectorizeParallelToPTO] internal error: non-Op call passed "
             "Verify: "
          << expr;
      const char *unary = pto_analysis::VectorUnaryVmiOpName(
          call_op.value()->name);
      ICHECK(unary != nullptr)
          << "[VectorizeParallelToPTO] internal error: call passed Verify "
             "without a VMI mapping: "
          << call_op.value()->name;
      PrimExpr a = ConvertExpr(call->args[0], pre);
      if (a.dtype().lanes() == 1) {
        return Call(expr.dtype(), call->op, {a});
      }
      return EmitBind(unary,
                      Call(VectorDType(expr.dtype().element_of(), lanes_),
                           VmiOp(unary), {a, mask_var_}),
                      pre);
    }
    if (const auto *cast = expr.as<CastNode>()) {
      // Convert the input exactly once, then plan from the explicit shape of
      // that converted input (the original Cast carries the element types).
      PrimExpr value = ConvertExpr(cast->value, pre);
      pto_analysis::CastContext context;
      if (value.dtype().lanes() == 1) {
        context.value_kind = pto_analysis::CastValueKind::kScalar;
        context.logical_lanes = 1;
      } else {
        ICHECK_EQ(value.dtype().lanes(), lanes_)
            << "[VectorizeParallelToPTO] internal error: Cast input has "
               "unexpected lanes "
            << value.dtype().lanes() << " (current VL " << lanes_ << ")";
        context.value_kind = pto_analysis::CastValueKind::kVectorVmi;
        context.logical_lanes = lanes_;
      }
      ICHECK_EQ(value.dtype().element_of(), cast->value.dtype())
          << "[VectorizeParallelToPTO] internal error: converted Cast input "
             "element type "
          << value.dtype().element_of() << " does not match the original "
          << cast->value.dtype();
      std::string plan_why;
      auto plan = pto_analysis::PlanValueCast(cast, context, &plan_why);
      ICHECK(plan.has_value())
          << "[VectorizeParallelToPTO] internal error: Cast passed Verify "
             "without a conversion plan: "
          << plan_why;
      if (const auto *scalar =
              std::get_if<pto_analysis::ScalarCastPlan>(&plan.value())) {
        ICHECK_EQ(value.dtype().lanes(), 1)
            << "[VectorizeParallelToPTO] internal error: scalar Cast plan "
               "for a vector input";
        if (scalar->action == pto_analysis::ScalarCastAction::kKeepInput) {
          return value;
        }
        // Rebuild the ordinary scalar Cast with the validated annotations and
        // the original span; the scalar lowering itself is unchanged.
        return Cast(cast->dtype, value, scalar->annotations, cast->span);
      }
      const auto &vector = std::get<pto_analysis::VectorCastPlan>(plan.value());
      PrimExpr current = value;
      for (const auto &step : vector.steps) {
        ICHECK_EQ(current.dtype().element_of(), step.from)
            << "[VectorizeParallelToPTO] internal error: Cast step expects "
            << step.from << ", got " << current.dtype().element_of();
        Map<String, ObjectRef> attrs;
        attrs.Set("to_dtype", StringImm(TvmDtypeName(step.to)));
        PrimExpr call;
        if (step.op == pto_analysis::VectorCastOp::kIntegerReinterpret) {
          call = Call(VectorDType(step.to, lanes_), VmiOp("vinterpret_cast"),
                      {current}, attrs);
        } else {
          if (!step.saturate.empty()) {
            attrs.Set("saturate", StringImm(step.saturate));
          }
          if (!step.rounding.empty()) {
            attrs.Set("rounding", StringImm(step.rounding));
          }
          call = Call(VectorDType(step.to, lanes_), VmiOp("vcvt"), {current},
                      attrs);
        }
        current = EmitBind("cast", call, pre);
      }
      // Zero steps is the proven identity; a final type check keeps every
      // multi-step chain honest (source convert once, dtype/VL continuous).
      ICHECK_EQ(current.dtype(), VectorDType(cast->dtype, lanes_))
          << "[VectorizeParallelToPTO] internal error: Cast plan ended at "
          << current.dtype() << ", expected "
          << VectorDType(cast->dtype, lanes_);
      return current;
    }
    if (const auto *andn = expr.as<AndNode>()) {
      return ConvertMaskLogic(andn->a, andn->b, "vand", pre);
    }
    if (const auto *orn = expr.as<OrNode>()) {
      return ConvertMaskLogic(orn->a, orn->b, "vor", pre);
    }
    if (const auto *notn = expr.as<NotNode>()) {
      PrimExpr a = AsLaneMask(ConvertExpr(notn->a, pre), pre);
      DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
      ICHECK(a.dtype().lanes() > 1)
          << "[VectorizeParallelToPTO] vnot expects a lane mask";
      return EmitBind("mnot", Call(mask_ty, VmiOp("vnot"), {a}), pre);
    }
    if (const auto *sel = expr.as<SelectNode>()) {

      // A Select inside a Parallel body may test a lane-uniform condition
      // (it depends only on the non-vectorized coordinates, e.g. a row
      // index). PTODSL has no scalar predicate, so the
      // compare is redone in the vector domain and the result is a mask
      // that is either all-true or all-false; a bare scalar bool has no
      // operands to compare, so it is widened and compared against zero.
      PrimExpr cond;
      if (RelationalCmpName(sel->condition) != nullptr) {
        cond = ConvertRelationalToMask(sel->condition, pre);
      } else {
        // Convert first: a `Bind` var may be declared bool while it holds a
        // lane mask, so the source dtype is not the deciding fact.
        cond = ConvertExpr(sel->condition, pre);
        if (IsScalarBool(cond)) {
          cond = ScalarBoolToMask(cond, pre);
        }
      }
      PrimExpr tval = ConvertExpr(sel->true_value, pre);
      PrimExpr fval = ConvertExpr(sel->false_value, pre);
      if (tval.dtype().lanes() == 1) {
        tval = BroadcastToLanes("brc", tval, pre);
      }
      if (fval.dtype().lanes() == 1) {
        fval = BroadcastToLanes("brc", fval, pre);
      }
      ICHECK(cond.dtype().lanes() > 1)
          << "[VectorizeParallelToPTO] unsupported Select condition: expected "
             "a lane mask, a lane-uniform comparison, or a scalar bool; got a "
             "scalar `"
          << sel->condition->GetTypeKey()
          << "`. Rewrite the test as a comparison, or make it lane-varying.";
      return EmitBind(
          "vsel", Call(tval.dtype(), VmiOp("vsel"), {cond, tval, fval}), pre);
    }
    PrimExpr rel_a;
    PrimExpr rel_b;
    if (const pto::RelationalKind *rel =
        pto::MatchRelational(expr, &rel_a, &rel_b)) {
      return ConvertCompare(rel_a, rel_b, rel->cmp, pre);
    }
    LOG(FATAL) << "[VectorizeParallelToPTO] internal error: unsupported "
                  "expression passed Verify: "
               << expr->GetTypeKey();
  }

  // Relational node -> PTODSL compare name, or nullptr for other nodes.
  static const char *RelationalCmpName(const PrimExpr &e) {
    const pto::RelationalKind *rel = pto::MatchRelational(e, nullptr, nullptr);
    return rel != nullptr ? rel->cmp : nullptr;
  }

  // A bool with no lanes: a kernel parameter, a value read from a 1-element
  // buffer, or the result of scalar-only arithmetic. PTODSL has no scalar
  // predicate, so such a value cannot be lowered as-is.
  static bool IsScalarBool(const PrimExpr &e) {
    return e.dtype().is_bool() && e.dtype().lanes() == 1;
  }

  // Materialise a scalar bool as a lane mask that is either all-true or
  // all-false, which is exactly the semantics of applying one scalar
  // predicate to a whole vector.  A VMI mask must be produced by
  // pto.vmi.create_mask, so the flag becomes the active-lane count: all lanes
  // when it is true, none when it is false.  The mask therefore stays in the
  // VMI domain from the start, with no vector operand and no carrier dtype.
  PrimExpr ScalarBoolToMask(const PrimExpr &flag, Array<Stmt> *pre) {
    DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
    PrimExpr active = EmitBind(
        "pcount",
        Select(flag, IntImm(DataType::Int(32), static_cast<int>(lanes_)),
               IntImm(DataType::Int(32), 0)),
        pre);
    return EmitBind(
        "cmask", Call(mask_ty, VmiOp("create_mask"), {active},
                      {{"size",
                        IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
        pre);
  }

  // Turn a value that may still be a scalar bool into a lane mask. Anything
  // that is already a lane vector (a real mask) passes through untouched.
  PrimExpr AsLaneMask(const PrimExpr &value, Array<Stmt> *pre) {
    if (value.dtype().lanes() > 1) {
      return value;
    }
    if (IsScalarBool(value)) {
      return ScalarBoolToMask(value, pre);
    }
    return value;
  }

  // Materialise a relational condition as a lane mask even when both sides
  // are lane-uniform: broadcast the scalars and compare vector-wise. The
  // resulting mask is all-true or all-false, which is exactly the semantics
  // of a scalar predicate applied to a whole vector.
  PrimExpr ConvertRelationalToMask(const PrimExpr &cond, Array<Stmt> *pre) {
    PrimExpr lhs;
    PrimExpr rhs;
    const pto::RelationalKind *rel = pto::MatchRelational(cond, &lhs, &rhs);
    ICHECK(rel != nullptr) << "not a relational condition";
    const char *cmp = rel->cmp;
    Array<Stmt> local;
    PrimExpr a = ConvertExpr(lhs, &local);
    PrimExpr b = ConvertExpr(rhs, &local);
    for (const auto &s : local) {
      pre->push_back(s);
    }
    if (a.dtype().lanes() == 1) {
      a = BroadcastToLanes("brc", a, pre);
    }
    if (b.dtype().lanes() == 1) {
      b = BroadcastToLanes("brc", b, pre);
    }
    return EmitBind(
        "vcmp",
        Call(DataType::Bool(static_cast<int>(lanes_)), VmiOp("vcmp"),
             {a, b, mask_var_, StringImm(cmp)}),
        pre);
  }

  PrimExpr ConvertCompare(const PrimExpr &lhs, const PrimExpr &rhs,
                          const char *cmp, Array<Stmt> *pre) {
    PrimExpr a = ConvertExpr(lhs, pre);
    PrimExpr b = ConvertExpr(rhs, pre);
    bool a_vec = a.dtype().lanes() > 1;
    bool b_vec = b.dtype().lanes() > 1;
    DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
    if (!a_vec && !b_vec) {
      // Lane-uniform compare: keep scalar relational (Select of scalars is
      // rare inside Parallel; Verify still allows it). The scalar semantics
      // come from the shared relational table, not a second copy of it.
      const pto::RelationalKind *kind = pto::RelationalByName(cmp);
      ICHECK(kind != nullptr)
          << "[VectorizeParallelToPTO] internal error: unknown cmp " << cmp;
      return kind->fold(a, b);
    }
    if (a_vec && !b_vec) {
      return EmitBind(
          "vcmps",
          Call(mask_ty, VmiOp("vcmps"), {a, b, mask_var_, StringImm(cmp)}),
          pre);
    }
    if (!a_vec && b_vec) {
      a = BroadcastToLanes("brc", a, pre);
    }
    return EmitBind(
        "vcmp", Call(mask_ty, VmiOp("vcmp"), {a, b, mask_var_, StringImm(cmp)}),
        pre);
  }

  PrimExpr ConvertMaskLogic(const PrimExpr &lhs, const PrimExpr &rhs,
                            const char *op_name, Array<Stmt> *pre) {
    PrimExpr a = AsLaneMask(ConvertExpr(lhs, pre), pre);
    PrimExpr b = AsLaneMask(ConvertExpr(rhs, pre), pre);
    DataType mask_ty = DataType::Bool(static_cast<int>(lanes_));
    ICHECK(a.dtype().lanes() > 1 && b.dtype().lanes() > 1)
        << "[VectorizeParallelToPTO] " << op_name
        << " expects lane masks (a non-bool scalar operand has no lane mask)";
    return EmitBind(op_name, Call(mask_ty, VmiOp(op_name), {a, b}), pre);
  }

  PrimExpr ConvertBinary(const PrimExpr &lhs, const PrimExpr &rhs,
                         const std::string &vmi_name, Array<Stmt> *pre) {
    PrimExpr a = ConvertExpr(lhs, pre);
    PrimExpr b = ConvertExpr(rhs, pre);
    bool a_vec = a.dtype().lanes() > 1;
    bool b_vec = b.dtype().lanes() > 1;
    if (!a_vec && !b_vec) {
      // Lane-uniform folds stay scalar with the TIR semantics of the node.
      static const std::unordered_map<std::string, ScalarFoldFn>
          kScalarBinaryFold = {
              {"vadd",
               [](const PrimExpr &a, const PrimExpr &b) { return a + b; }},
              {"vsub",
               [](const PrimExpr &a, const PrimExpr &b) { return a - b; }},
              {"vmul",
               [](const PrimExpr &a, const PrimExpr &b) { return a * b; }},
              {"vdiv",
               [](const PrimExpr &a, const PrimExpr &b) { return div(a, b); }},
          };
      auto it = kScalarBinaryFold.find(vmi_name);
      if (it != kScalarBinaryFold.end()) {
        return it->second(a, b);
      }
      ICHECK(false) << "[VectorizeParallelToPTO] internal error: unknown "
                       "binary op "
                    << vmi_name;
    }
    if (a_vec && !b_vec) {
      b = BroadcastToLanes("brc", b, pre);
    } else if (!a_vec && b_vec) {
      a = BroadcastToLanes("brc", a, pre);
    }
    return EmitBind(vmi_name,
                    Call(VectorDType(a.dtype().element_of(), lanes_),
                         VmiOp(vmi_name), {a, b, mask_var_}),
                    pre);
  }

  // TVM dtype name identical to Python str(dtype) (vmi.py ABI):
  // float32 -> "float32", bfloat16 -> "bfloat16", int8 -> "int8", ...
  static std::string TvmDtypeName(DataType dtype) {
    std::ostringstream os;
    if (dtype.is_bfloat16()) {
      return "bfloat16";
    }
    // The FP8 spellings the PTO dtype parser recognizes; both are the TIR
    // dtype names, not the PTO target names.
    if (dtype.is_float8_e4m3fn()) {
      return "float8_e4m3fn";
    }
    if (dtype.is_float8_e5m2()) {
      return "float8_e5m2";
    }
    if (dtype.is_float()) {
      os << "float" << dtype.bits();
    } else if (dtype.is_int()) {
      os << "int" << dtype.bits();
    } else if (dtype.is_uint()) {
      os << "uint" << dtype.bits();
    } else if (dtype.is_bool()) {
      return "bool";
    } else {
      os << dtype;
    }
    return os.str();
  }

  // Broadcast a lane-uniform scalar to the full vector width: bind a
  // vbrc of the scalar at the current VL and return the bound Var.
  PrimExpr BroadcastToLanes(const char *tag, const PrimExpr &scalar,
                            Array<Stmt> *pre) {
    return EmitBind(
        tag,
        Call(VectorDType(scalar.dtype(), lanes_), VmiOp("vbrc"), {scalar},
             {{"size", IntImm(DataType::Int(32), static_cast<int>(lanes_))}}),
        pre);
  }

  Var EmitBind(const std::string &name, const PrimExpr &value,
               Array<Stmt> *pre) {
    Var result(name + "_" + std::to_string(tmp_counter_++), value.dtype());
    pre->push_back(Bind(result, value));
    return result;
  }

  arith::Analyzer analyzer_;
  Var shared_mask_; // one full mask per VF, shared by all units

  // SBlocks in lexical scope (innermost last), for annotation lookups.
  std::vector<const SBlockNode *> block_stack_;
  // PTO slot storage (region level): fragment buffer -> {plan, layout}.
  std::unordered_map<Buffer, SlotState, ObjectPtrHash, ObjectPtrEqual>
      slot_map_;
  // Data vars of fragments whose accesses were converted in this region.
  std::set<const VarNode *> converted_fragments_;
  // Handles declared in the region being converted (the fresh Vars bound
  // by a declare_fragment_buffer Bind; the plan's pre-created
  // fragment_mem_var provides their names). Uses of these handles outside
  // a vload/vstore base are a conversion bug.
  std::set<const VarNode *> declared_handles_;
  // Region-level map from the declared handle Var to its plan, backing
  // the post-conversion declaration check.
  std::unordered_map<const VarNode *, PtoSlotPlan> handle_plans_;

  // Per-unit state.
  Var loop_var_;
  Var chunk_var_;      // real emitted loop var (q)
  Var chunk_analysis_; // analysis placeholder from the inverse mapping
  Var lane_var_;       // analysis placeholder; never escapes
  PrimExpr lane_indexed_expr_;
  /*! The complete inverse mapping of the unit being converted
   * (2D fields included), consumed by CurrentMapping(). */
  pto_analysis::PtoInverseMapping unit_mapping_;
  int64_t lanes_ = 0;
  int64_t nchunks_ = 0;
  Var mask_var_;
  /*! Per-unit scalar buffer holding `remaining` for a non-divisible
   * extent (invalid for divisible units). */
  tirx::Buffer remaining_buffer_{};
  // Loop-shell carry promotion (AnalyzeLoopShell): when a unit's
  // preserved serial shell matches the canonical carry pattern, the
  // carried buffer's in-shell accesses are rewritten to this local.var
  // vreg buffer, so the loop never round-trips through UB — the
  // in-shell cross-iteration RAW is exactly the shape PR307's
  // auto-bar coverage does not model for deep nests (PTOAS
  // hw-native-sys/PTOAS#1679), so keeping it in registers is the
  // safe shape.
  const BufferStoreNode *carry_seed_ = nullptr;
  const ForNode *carry_loop_ = nullptr;
  tirx::Buffer carry_src_{};
  tirx::Buffer carry_vreg_{};
  Array<PrimExpr> carry_indices_{};
  bool carry_active_ = false;
  // Ranges of serial loops in scope at the fragment access being converted
  // (region-level wrappers from EnclosingSerialRanges plus in-unit shells
  // bound by ConvertBody). Passed to AnalyzeFragmentAccess so the entry
  // interval is provable for entries that depend on outer loops.
  Map<Var, Range> outer_loop_ranges_;
  // Serial loops in scope at the current mutation position outside any
  // region (kernel level), maintained by the mutator's For visitor.
  Map<Var, Range> outer_serial_ranges_;
  // Region-level carry promotion state (AnalyzeRegionCarry). When a unit is
  // the seed or an in-loop unit of the promoted region, its B accesses are
  // redirected to this vreg and the allocation happens once at region level
  // (unit_carry_from_region_ suppresses WrapCarryAlloc).
  RegionCarry region_carry_{};
  bool unit_carry_from_region_ = false;
  // Region-level carry matches any single lane Var index (each unit has its
  // own Parallel loop var), not one fixed expression.
  bool carry_lane_var_match_ = false;
  // Per-lane scalar promotion (`T.alloc_var` inside a Parallel unit). The
  // single-element `local.var` buffer is a private register per lane, so each
  // whole-buffer access becomes a `local.var` vector (vreg) access: the seed
  // store broadcasts, later loads/stores reference the same vreg. Keyed by the
  // source buffer node; the vregs are declared at unit scope by
  // DeclareLaneScalarVregs.
  std::unordered_map<const BufferNode *, tirx::Buffer> lane_scalar_vregs_;
  std::vector<tirx::Buffer> lane_scalar_allocs_;
  int tmp_counter_ = 0;
  Map<Var, PrimExpr> bind_env_; // analysis env: original var -> def expr
  Map<Var, Var> value_map_;     // value env: original var -> converted var
};

} // namespace

namespace transform {

tvm::transform::Pass VectorizeParallelToPTO() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return f;
    }
    return VectorizeParallelToPTOImpl::Substitute(std::move(f));
  };
  return tvm::tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.VectorizeParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VectorizeParallelToPTO",
                        VectorizeParallelToPTO);
}

} // namespace transform

} // namespace tl
} // namespace tvm

