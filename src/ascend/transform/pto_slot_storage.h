/*!
 * \file pto_slot_storage.h
 * \brief Region-level state for PTO slot (fragment) storage conversion.
 *
 * Owns the slot plans and fragment layouts loaded from the SBlock chain,
 * the declaration emission (one declare_fragment_buffer per plan per VF),
 * and the post-conversion checks (no residual fragment access, every
 * declared handle used as a vload/vstore base). The mutator in
 * vectorize_parallel_to_pto.cc holds one instance per function and calls
 * into it at region boundaries; expression-level conversion of individual
 * fragment accesses (ConvertFragmentLoad/Store) stays in the mutator
 * because it needs the full expression-conversion environment.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_PTO_SLOT_STORAGE_H_
#define TVM_TL_ASCEND_TRANSFORM_PTO_SLOT_STORAGE_H_

#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

#include <set>
#include <unordered_map>
#include <vector>

#include "layout/layout.h"
#include "pto_slot_plan.h"

namespace tvm {
namespace tl {
namespace pto {

/*! \brief Per-region slot storage state and block-rebuild helpers.
 *
 * One instance lives for the whole function (the mutator's lifetime);
 * \c Build resets and repopulates it at each converting region entry
 * from the SBlock chain (innermost-first: the converting block and its
 * ancestors carry the plan and layout annotations).
 */
class PtoSlotStorageModule {
public:
  struct SlotState {
    PtoSlotPlan plan;
    Fragment layout;
    /*! Handle bound by this region's declare_fragment_buffer. A fresh
     * Var per region occurrence; the plan's pre-created
     * fragment_mem_var provides the name and the planning identity. */
    Var handle;
  };

  /*! Load the slot plans and fragment layouts from the SBlock chain
   * (innermost-first). Also resets the per-region tracking state. */
  void Build(const std::vector<const tirx::SBlockNode *> &block_stack);

  /*! Look up the slot state for a fragment buffer that passed Verify.
   * Marks the fragment as converted (drives the block rebuild below). */
  const SlotState &RequireSlotState(const tirx::Buffer &buffer);

  /*! Find the fragment layout for \p buffer on the block chain; empty
   * when the buffer has no layout_map entry in scope. */
  Fragment FindFragmentLayout(
      const tirx::Buffer &buffer,
      const std::vector<const tirx::SBlockNode *> &block_stack) const;

  /*! Fragment buffers of this region that map to a slot, in document
   * order of first access; drives one declaration per plan per VF. */
  std::vector<tirx::Buffer> CollectAccessedSlotFragments(
      const tirx::Stmt &region) const;

  /*! Emit the declare_fragment_buffer Bind for every accessed slot,
   * prepended to \p region. Also records handle→plan for the checks. */
  tirx::Stmt EmitSlotDeclarations(const tirx::Stmt &region, int64_t lanes);

  /*! Post-conversion: no local.fragment access may survive. */
  void CheckNoResidualFragmentAccess(const tirx::Stmt &body) const;

  /*! Post-conversion: every declared handle is used only as a
   * vload/vstore base with entry-granular offsets. */
  void CheckSlotDeclarations(const tirx::Stmt &body) const;

  /*! Drop the consumed fragment allocations, their layout_map entries,
   * the consumed slot plans and their read/write regions from the
   * rebuilt block. The backing allocations stay: the emitted backing
   * loads/stores reference them. */
  tirx::Stmt RebuildBlockWithoutConsumedFragments(
      const tirx::SBlockNode *op, const tirx::Stmt &new_body);

  /*! Drop kernel-scope fragment allocations consumed by child regions. */
  tirx::Stmt DropConsumedFragmentsFromBlock(
      const tirx::SBlockNode *original, const tirx::Stmt &rebuilt);

  bool IsConverted(const tirx::VarNode *data) const {
    return converted_fragments_.count(data) != 0;
  }

  bool IsDeclared(const tirx::VarNode *handle) const {
    return declared_handles_.count(handle) != 0;
  }

  const std::unordered_map<tirx::Buffer, SlotState, ffi::ObjectPtrHash,
                           ffi::ObjectPtrEqual> &
  slot_map() const {
    return slot_map_;
  }

private:
  /*! Base address operand of one slot's declaration: a standard
   * rewritable access pointer over the whole backing. */
  PrimExpr MakeBackingBasePtr(const SlotState &state) const;

  std::unordered_map<tirx::Buffer, SlotState, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      slot_map_;
  std::set<const tirx::VarNode *> converted_fragments_;
  std::set<const tirx::VarNode *> declared_handles_;
  std::unordered_map<const tirx::VarNode *, PtoSlotPlan> handle_plans_;
};

} // namespace pto
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_PTO_SLOT_STORAGE_H_
