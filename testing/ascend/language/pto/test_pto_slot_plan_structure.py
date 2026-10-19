"""PTO slot storage structure tests.

Hardware independent: each case lowers TileLang IR for the PTO target with
:func:`tilelang.engine.lower.lower` (TileLang IR only, no PTOAS/Bisheng device
compilation and no device execution) and inspects the modules produced by
``PlanPtoSlotStorage`` and ``VectorizeParallelToPTO`` through capture
wrappers.

Covered contract:

* PlanPtoSlotStorage declares one hidden ``shared.dyn`` backing per accessed
  ``local.fragment`` buffer and records a ``tl.pto_slot_plan`` annotation
  (entry count from the layout, vector lanes, aligned byte size, element
  dtype name) on the region's SBlock; the fragment allocation and its
  accesses are untouched at this stage;
* the entry count follows the layout (2D ``acc[M, VL]`` -> ``M`` entries;
  1D ``scores[E]`` -> ``ceil(E / VL)`` entries — never ``elements / VL``);
* Vectorize binds one ``slot_alloc(backing, count, vector_type)`` per plan
  inside the ``tl.simdvf_scope`` attr, converts fragment loads/stores into
  ``slot_load`` / ``slot_store`` with the entry index expressed over the
  chunk loop variable, keeps two textual reads as two ``slot_load`` calls,
  and leaves no ``local.fragment`` access behind;
* region-level fragment access, ``T.fill``/``T.copy`` on a fragment, a
  lane-shifted fragment index (``B[i, j+1]``) and a fragment crossing two
  SIMD_VF regions are rejected by Verify;
* a slot reservation exceeding the unified-buffer capacity is rejected by
  the AutoSchedule budget.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
import tvm


# ---------------------------------------------------------------------------
# Capture helpers (same style as the existing PTO structure suites)
# ---------------------------------------------------------------------------


@contextmanager
def _capture_pass(pass_name: str):
    captured = {}
    original = getattr(ascend_transform, pass_name)

    def capturing_factory():
        inner = original()

        def run(mod):
            result = inner(mod)
            captured["module"] = result
            return result

        return run

    setattr(ascend_transform, pass_name, capturing_factory)
    try:
        yield captured
    finally:
        setattr(ascend_transform, pass_name, original)


def _lower_capture(func, pass_name: str, *, expect_error=False):
    target = normalize_pto_target("pto")
    error = None
    with _capture_pass(pass_name) as captured:
        try:
            with target:
                _lower(func, target=target)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            error = exc
    if error is not None and not expect_error:
        raise error
    return captured.get("module"), error


def _body_of(mod):
    assert len(mod.functions) == 1, f"expected a single function, got {list(mod.functions)}"
    return next(iter(mod.functions.values())).body


def _visit(mod, fn):
    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), fn)


def _vmi_calls(mod, name: str):
    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            op_name = getattr(getattr(node, "op", None), "name", None)
            if op_name == f"tl.vmi.{name}":
                calls.append(node)

    _visit(mod, visit)
    return calls


def _sblocks(mod):
    blocks = []

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock):
            blocks.append(node)

    _visit(mod, visit)
    return blocks


def _simdvf_block(mod):
    for block in _sblocks(mod):
        if block.name_hint == "SIMD_VF":
            return block
    return None


def _slot_plans(mod):
    plans = []
    for block in _sblocks(mod):
        for key, value in block.annotations.items():
            if key == "tl.pto_slot_plan":
                for plan in value:
                    plans.append((block, plan))
    return plans


def _collect_stores(mod, scope: str):
    stores = []

    def visit(node):
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.scope() == scope:
            stores.append(node)

    _visit(mod, visit)
    return stores


def _collect_loads(mod, scope: str):
    loads = []

    def visit(node):
        if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.scope() == scope:
            loads.append(node)

    _visit(mod, visit)
    return loads


def _backing_vmi_calls(mod, vmi_name: str, backing_name: str):
    """VMI calls of *vmi_name* whose access-ptr reads the named backing."""

    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            op_name = getattr(getattr(node, "op", None), "name", None)
            if op_name == f"tl.vmi.{vmi_name}":
                text = str(node)
                if backing_name in text:
                    calls.append(node)

    _visit(mod, visit)
    return calls


def _mem_bars(mod):
    bars = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            op_name = getattr(getattr(node, "op", None), "name", None)
            if op_name == "tl.simd.mem_bar":
                bars.append(node)

    _visit(mod, visit)
    return bars


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _k_sv3_keep(CM: int, lanes: int, K: int = 3, dtype="float32"):
    """GEMV partial-sum keep: acc[CM, VL] lives across a serial K loop.

    ``xk`` is inlined as a lane-uniform UB scalar load (a ``local.var``
    accumulator is not accepted by the current input contract)."""

    @T.prim_func
    def main(
        A: T.Tensor((K, CM * lanes), dtype),
        X: T.Tensor((K,), dtype),
        Out: T.Tensor((CM * lanes,), dtype),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((K, CM * lanes), dtype)
            x_ub = T.alloc_shared((K,), dtype)
            out_ub = T.alloc_shared((CM * lanes,), dtype)
            T.copy(A, a_ub)
            T.copy(X, x_ub)
            with T.SimdVF(lanes=lanes):
                acc = T.alloc_fragment((CM, lanes), dtype)
                for i, j in T.Parallel(CM, lanes):
                    acc[i, j] = T.float32(0.0)
                for k in T.serial(K):
                    for i, j in T.Parallel(CM, lanes):
                        acc[i, j] = acc[i, j] + T.Cast(dtype, a_ub[k, i * lanes + j]) * T.Cast(dtype, x_ub[k])
                for i, j in T.Parallel(CM, lanes):
                    out_ub[i * lanes + j] = acc[i, j]
            T.copy(out_ub, Out)

    return main


def _k_scores_1d(E: int, lanes: int, dtype="float32"):
    """cf1-shaped keep: a 1D fragment over E elements, read back in full."""

    @T.prim_func
    def main(
        S: T.Tensor((E,), dtype),
        Out: T.Tensor((E,), dtype),
    ):
        with T.Kernel(1):
            s_ub = T.alloc_shared((E,), dtype)
            out_ub = T.alloc_shared((E,), dtype)
            T.copy(S, s_ub)
            with T.SimdVF(lanes=lanes):
                scores = T.alloc_fragment((E,), dtype)
                for i in T.Parallel(E):
                    scores[i] = s_ub[i]
                for i in T.Parallel(E):
                    out_ub[i] = scores[i]
            T.copy(out_ub, Out)

    return main


# ---------------------------------------------------------------------------
# Plan structure
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_plan_declares_backing_and_annotation():
    CM, lanes = 24, 128
    mod, _ = _lower_capture(_k_sv3_keep(CM, lanes), "PlanPtoSlotStorage")
    plans = _slot_plans(mod)
    assert len(plans) == 1, f"expected one slot plan, got {len(plans)}"
    block, plan = plans[0]
    assert block.name_hint == "SIMD_VF"
    assert plan.fragment.name == "acc"
    assert plan.count == CM
    assert plan.vector_lanes == lanes
    assert plan.elem_dtype_name == "float32"
    # 24 entries * 128 lanes * 4 bytes = 12288, already 32B aligned.
    assert plan.bytes == 12288
    backing = plan.backing
    assert backing.scope() == "shared.dyn"
    assert int(backing.shape[0]) == CM * lanes
    # The fragment allocation is untouched at this stage and the region still
    # carries the scalar accesses.
    simdvf = _simdvf_block(mod)
    alloc_names = [buf.name for buf in simdvf.alloc_buffers]
    assert "acc" in alloc_names
    assert backing.name in alloc_names
    assert _collect_stores(mod, "local.fragment"), "fragment stores must survive Plan"


@pytest.mark.pto
def test_plan_count_1d_is_ceil_division():
    E, lanes = 200, 128  # non-divisible: 2 entries, not ceil(200*?/128)
    mod, _ = _lower_capture(_k_scores_1d(E, lanes), "PlanPtoSlotStorage")
    plans = _slot_plans(mod)
    assert len(plans) == 1
    plan = plans[0][1]
    assert plan.count == (E + lanes - 1) // lanes
    assert plan.bytes == 2 * lanes * 4


# ---------------------------------------------------------------------------
# Vectorize structure
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_vectorize_slot_roundtrip_2d():
    CM, lanes, K = 8, 128, 3
    mod, _ = _lower_capture(_k_sv3_keep(CM, lanes, K), "VectorizeParallelToPTO")

    backing = "acc_slot_backing"
    stores = _backing_vmi_calls(mod, "vstore", backing)
    loads = _backing_vmi_calls(mod, "vload", backing)
    # Structural statement counts (TIR keeps loops, no unrolling): the
    # zero-init round contributes one backing store (broadcast value), the
    # accumulation round one backing load + one backing store inside its
    # k/i loops, and the write-back round one backing load.
    assert len(stores) == 2, [str(s) for s in stores]
    assert len(loads) == 2, [str(s) for s in loads]
    # Entry q lives at element offset q * lanes (identity layout).
    for call in loads + stores:
        assert f"i * {lanes}" in str(call), str(call)

    # A vecscope mem_bar orders every backing-writing unit before the next
    # unit's backing reads (two writing units here: zero-init and the
    # accumulation loop).
    bars = _mem_bars(mod)
    assert len(bars) == 2, [str(b) for b in bars]
    assert all(str(b) == 'T.tl.simd.mem_bar("VST_VLD")' for b in bars)

    # No fragment access survives conversion; the fragment allocation is
    # consumed while the backing stays.
    assert not _collect_stores(mod, "local.fragment")
    assert not _collect_loads(mod, "local.fragment")
    simdvf = _simdvf_block(mod)
    alloc_names = [buf.name for buf in simdvf.alloc_buffers]
    assert "acc" not in alloc_names, "consumed fragment allocation must be removed"
    assert any(n.endswith("_slot_backing") for n in alloc_names)


@pytest.mark.pto
def test_vectorize_two_reads_stay_two_loads():
    """Two textual reads of the same fragment element are two backing
    vloads (no cross-access reuse at the TileLang level)."""
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    keep[i] = a_ub[i]
                for i in T.Parallel(E):
                    b_ub[i] = keep[i] * keep[i]
            T.copy(b_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    # 1 store round + 2 loads in the consumer round == 2 backing vloads.
    assert len(_backing_vmi_calls(mod, "vload", "keep_slot_backing")) == 2


# ---------------------------------------------------------------------------
# Verify rejections
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_region_level_fragment_access_rejected():
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((1,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    keep[i] = a_ub[i]
                b_ub[0] = keep[0]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "VectorizeParallelToPTO", expect_error=True)
    assert error is not None
    text = str(error)
    # The bare region-level statement is wrapped by the frontend into an
    # implicit extent-1 unit, so it may hit either the region-level
    # fragment rule or the pre-existing lane-uniform store rejection — both
    # are correct rejections of the single-element fragment extract.
    assert "outside the converting T.Parallel bodies" in text or "not yet supported" in text, text


@pytest.mark.pto
def test_fragment_fill_rejected():
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((E,), "float32")
                T.fill(keep, T.float32(0.0))
                for i in T.Parallel(E):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "VectorizeParallelToPTO", expect_error=True)
    assert error is not None
    assert "on fragment buffers is not supported yet" in str(error)


@pytest.mark.pto
def test_lane_shifted_fragment_index_rejected():
    lanes = 128
    M, N = 4, lanes

    @T.prim_func
    def main(A: T.Tensor((M * N,), "float32"), B: T.Tensor((M * N,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((M * N,), "float32")
            b_ub = T.alloc_shared((M * N,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((M, N), "float32")
                for i, j in T.Parallel(M, N):
                    keep[i, j] = a_ub[i * N + j]
                for i, j in T.Parallel(M, N):
                    # j+1 shifts the element's storage lane by one — the
                    # whole-entry slot form cannot express it.
                    b_ub[i * N + j] = keep[i, j + 1]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "VectorizeParallelToPTO", expect_error=True)
    assert error is not None
    text = str(error)
    assert "storage lane" in text or "slot entry index" in text, text


@pytest.mark.pto
def test_fragment_crossing_two_regions_rejected():
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            # Allocated at kernel scope so both regions see it lexically;
            # the plan is recorded on the first region's SBlock only, so the
            # second region must reject the cross-region use.
            keep = T.alloc_fragment((E,), "float32")
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    keep[i] = a_ub[i]
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    b_ub[i] = keep[i]
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "VectorizeParallelToPTO", expect_error=True)
    assert error is not None
    text = str(error)
    assert "no tl.pto_slot_plan record in scope" in text or "no layout_map entry" in text, text


@pytest.mark.pto
def test_conditional_fragment_store_legalizes_to_select():
    """cf1 shape: a single-sided conditional fragment store. Legalize
    rewrites it to Select(new, old) with the old value read from the same
    buffer, which Vectorize turns into a backing vload + vsel + one
    masked backing vstore (inactive lanes keep the previous contents)."""
    lanes = 128
    E = lanes

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            out_ub = T.alloc_shared((E,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF(lanes=lanes):
                scores = T.alloc_fragment((E,), "float32")
                for i in T.Parallel(E):
                    scores[i] = a_ub[i]
                for i in T.Parallel(E):
                    if a_ub[i] > T.float32(0.5):
                        scores[i] = T.float32(-1.0)
                for i in T.Parallel(E):
                    out_ub[i] = scores[i]
            T.copy(out_ub, B)

    mod, _ = _lower_capture(main, "VectorizeParallelToPTO")
    backing = "scores_slot_backing"
    # The conditional round reads the old value once (Select else-arm),
    # vsel-merges, and writes the entry once.
    loads = _backing_vmi_calls(mod, "vload", backing)
    stores = _backing_vmi_calls(mod, "vstore", backing)
    vsels = _vmi_calls(mod, "vsel")
    # init round: 1 store; conditional round: 1 load (old value) + 1
    # store; write-back round: 1 load.
    assert len(stores) == 2, [str(s) for s in stores]
    assert len(loads) == 2, [str(s) for s in loads]
    assert len(vsels) == 1, "the conditional update must merge via vsel"


# ---------------------------------------------------------------------------
# Capacity budget
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_slot_reservation_over_capacity_rejected():
    lanes = 128
    # 1024 entries * 128 lanes * 4 B = 512 KiB > the 248 KiB unified buffer.
    # The kernel keeps every other UB allocation tiny so the slot deduction
    # is unambiguously the failing check.
    CM = 1024

    @T.prim_func
    def main(B: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            b_ub = T.alloc_shared((1,), "float32")
            with T.SimdVF(lanes=lanes):
                keep = T.alloc_fragment((CM, lanes), "float32")
                for i, j in T.Parallel(CM, lanes):
                    keep[i, j] = T.float32(1.0)
            b_ub[0] = T.float32(0.0)
            T.copy(b_ub, B)

    _, error = _lower_capture(main, "PlanPtoSlotStorage", expect_error=True)
    assert error is not None
    assert "PTO slot storage reserves" in str(error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
