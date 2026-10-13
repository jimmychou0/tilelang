"""End-to-end NPU tests for the Parallel->VMI path.

These kernels contain ``T.Parallel`` inside ``T.SimdVF(lanes=...)`` and run
through the real ``tilelang.compile(target="pto")`` pipeline, which routes
them to ``VerifyParallelToPTO`` + ``VectorizeParallelToPTO`` instead of
``AscendSimdVFLowerParallel``. Auto-schedule stays enabled: the non-autosched
pipeline has no InsertSync (MTE sync), so numeric results would be garbage.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import tilelang
import tilelang.ascend.language as T

from tilelang import tvm
from tilelang.engine.lower import lower


def _require_npu_runtime():
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu is unavailable in the current environment")

    try:
        _ = torch.randn(1, device=torch.device("npu"))
        torch.npu.synchronize()
    except Exception as err:
        pytest.skip(f"NPU runtime is unavailable in the current environment: {err}")


def _vecadd(n, lanes):
    """C = A * (A + B) over a 1D Parallel, multi-chunk when n > lanes."""

    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32"), C: T.Buffer((n,), "float32")):
        with T.Kernel(1):
            t1 = T.alloc_shared((n,), "float32")
            t2 = T.alloc_shared((n,), "float32")
            t3 = T.alloc_shared((n,), "float32")
            T.copy(A, t1)
            T.copy(B, t2)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    t3[i] = t1[i] * (t1[i] + t2[i])
            T.copy(t3, C)

    return main


def _i32_index(n, lanes):
    """C[i] = A[i] + i: int32 loop index feeds the value stream, exercising the
    integer VCI signless bridge (vinterpret_cast) on the new path."""

    @T.prim_func
    def main(A: T.Buffer((n,), "int32"), C: T.Buffer((n,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "int32")
            c = T.alloc_shared((n,), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = a[i] + i
            T.copy(c, C)

    return main


def _row_kernel(m, n, lanes):
    """2D j-continuous: c[i, j] = a[i, j] + i + j over T.Parallel(m, n)."""

    @T.prim_func
    def main(A: T.Buffer((m, n), "int32"), C: T.Buffer((m, n), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((m, n), "int32")
            c = T.alloc_shared((m, n), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(m, n):
                    c[i, j] = a[i, j] + i + j
            T.copy(c, C)

    return main


def _column_kernel(m, n, lanes):
    """2D i-continuous: buffers are (n, m) and accessed [j, i]."""

    @T.prim_func
    def main(A: T.Buffer((n, m), "int32"), C: T.Buffer((n, m), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((n, m), "int32")
            c = T.alloc_shared((n, m), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(m, n):
                    c[j, i] = a[j, i] + i + j
            T.copy(c, C)

    return main


def _cast_f16_muladd_1d(n, lanes):
    """1D widening cast: c[i] = f32(a[i]) * (f32(a[i]) + f32(b[i]))."""

    @T.prim_func
    def main(A: T.Buffer((n,), "float16"), B: T.Buffer((n,), "float16"), C: T.Buffer((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float16")
            b = T.alloc_shared((n,), "float16")
            c = T.alloc_shared((n,), "float32")
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.Cast("float32", a[i]) * (T.Cast("float32", a[i]) + T.Cast("float32", b[i]))
            T.copy(c, C)

    return main


def _cast_bf16_add_1d(n, lanes):
    """1D bf16 -> f32: c[i] = f32(a[i]) + f32(a[i])."""

    @T.prim_func
    def main(A: T.Buffer((n,), "bfloat16"), C: T.Buffer((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "bfloat16")
            c = T.alloc_shared((n,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.Cast("float32", a[i]) + T.Cast("float32", a[i])
            T.copy(c, C)

    return main


def _cast_i16_square_1d(n, lanes):
    """1D int16 -> int32 widening: c[i] = i32(a[i]) * i32(a[i])."""

    @T.prim_func
    def main(A: T.Buffer((n,), "int16"), C: T.Buffer((n,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "int16")
            c = T.alloc_shared((n,), "int32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.Cast("int32", a[i]) * T.Cast("int32", a[i])
            T.copy(c, C)

    return main


def _cast_f16_add_2d(m, n, lanes):
    """2D j-continuous widening cast: c[i, j] = f32(a[i, j]) + f32(b[i, j])."""

    @T.prim_func
    def main(A: T.Buffer((m, n), "float16"), B: T.Buffer((m, n), "float16"), C: T.Buffer((m, n), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((m, n), "float16")
            b = T.alloc_shared((m, n), "float16")
            c = T.alloc_shared((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(m, n):
                    c[i, j] = T.Cast("float32", a[i, j]) + T.Cast("float32", b[i, j])
            T.copy(c, C)

    return main


def _fp8_cast_1d(n, lanes):
    """Cast to fp8: c[i] = f8(a[i]). Outside the supported dtype set."""

    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), C: T.Buffer((n,), "float8_e4m3")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            c = T.alloc_shared((n,), "float8_e4m3")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.Cast("float8_e4m3", a[i])
            T.copy(c, C)

    return main


def _i8_add_1d(n, lanes):
    """8-bit control: c[i] = a[i] + b[i] over int8, a whitelisted category."""

    @T.prim_func
    def main(A: T.Buffer((n,), "int8"), B: T.Buffer((n,), "int8"), C: T.Buffer((n,), "int8")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "int8")
            b = T.alloc_shared((n,), "int8")
            c = T.alloc_shared((n,), "int8")
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = a[i] + b[i]
            T.copy(c, C)

    return main


def _new_path_signatures(source):
    """Signatures proving the kernel took the new Parallel->VMI path."""
    return {
        "vload": source.count("pto.vmi.vload("),
        "vstore": source.count("pto.vmi.vstore("),
        "create_mask": source.count("pto.vmi.create_mask("),
        "vlds": source.count(".vlds("),
        "set_flag": source.count("set_flag"),
        "wait_flag": source.count("wait_flag"),
    }


def _assert_new_path(source, *, expect_vinterpret_cast=False, expect_scalar_cast=False, expect_vcvt_to=None):
    """The compiled PTODSL must come from VectorizeParallelToPTO: VMI ops with
    full mask, MTE sync from auto-schedule, and no legacy simd vlds."""
    sig = _new_path_signatures(source)
    assert sig["vload"] > 0 and sig["vstore"] > 0, f"no VMI memory ops in source:\n{source}"
    assert sig["create_mask"] > 0, f"no create_mask in source:\n{source}"
    assert sig["vlds"] == 0, f"legacy simd vlds leaked into source:\n{source}"
    assert sig["set_flag"] > 0 and sig["wait_flag"] > 0, (
        f"no MTE sync (set_flag/wait_flag) in source; auto-schedule must stay enabled for numeric correctness:\n{source}"
    )
    if expect_vinterpret_cast:
        assert "pto.vmi.vinterpret_cast(" in source, f"integer VCI bridge (vinterpret_cast) missing:\n{source}"
    if expect_scalar_cast:
        # PR269 emitted `scalar.cast(...)` here; the current PTOAS PTODSL surface
        # exposes the same cast as `pto.cast(...)`, so the bridge marker is the
        # cast nested directly in the reinterpreted VMI source.
        bridged = "vinterpret_cast(pto.vmi.vci(pto.cast(" in source or "vinterpret_cast(pto.vmi.vbrc(pto.cast(" in source
        assert bridged, f"dynamic integer bridge (pto.cast inside vinterpret_cast) missing:\n{source}"
    if expect_vcvt_to is not None:
        assert "pto.vmi.vcvt(" in source, f"element-wise conversion (vcvt) missing:\n{source}"
        assert f"to_dtype=pto.{expect_vcvt_to}" in source, f"vcvt target {expect_vcvt_to} missing:\n{source}"


@pytest.fixture()
def npu():
    _require_npu_runtime()
    return torch.device("npu")


def _run_and_check(func, inputs, expected, source_checks=None, atol=None):
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    source = kernel.get_kernel_source()
    if source_checks is not None:
        source_checks(source)
    out = kernel(*inputs)
    torch.npu.synchronize()
    got = out.cpu()
    if atol is None:
        assert torch.equal(got, expected), f"numeric mismatch: max_diff={torch.max(torch.abs((got.float() - expected.float()))).item()}"
    else:
        max_diff = torch.max(torch.abs(got.float() - expected.float())).item()
        assert torch.allclose(got.float(), expected.float(), rtol=0, atol=atol), f"numeric mismatch: max_diff={max_diff}"


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_vecadd_1d_multichunk(npu, lanes):
    """1D float32, n=512: multi-chunk (q loop) across all three lane widths."""
    n = 512
    a = torch.randn(n)
    b = torch.randn(n)
    _run_and_check(
        _vecadd(n, lanes),
        (a.to(npu), b.to(npu)),
        a * (a + b),
        source_checks=lambda s: _assert_new_path(s),
    )


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [128, 256])
def test_vecadd_1d_singlechunk(npu, lanes):
    """1D float32, n=lanes: single chunk, VCI base folds to a constant."""
    n = lanes
    a = torch.randn(n)
    b = torch.randn(n)
    _run_and_check(
        _vecadd(n, lanes),
        (a.to(npu), b.to(npu)),
        a * (a + b),
        source_checks=lambda s: _assert_new_path(s),
    )


@pytest.mark.pto
def test_i32_index_multichunk(npu):
    """1D int32 index in the value stream, multi-chunk: dynamic base bridge."""
    n = 512
    a = (torch.randn(n) * 1000).to(torch.int32)
    idx = torch.arange(n, dtype=torch.int32)
    _run_and_check(
        _i32_index(n, 128),
        (a.to(npu),),
        a + idx,
        source_checks=lambda s: _assert_new_path(s, expect_vinterpret_cast=True, expect_scalar_cast=True),
    )


@pytest.mark.pto
def test_i32_index_singlechunk(npu):
    """1D int32 index, single chunk: constant base goes through the bridge too."""
    n = 128
    a = (torch.randn(n) * 1000).to(torch.int32)
    idx = torch.arange(n, dtype=torch.int32)
    _run_and_check(
        _i32_index(n, 128),
        (a.to(npu),),
        a + idx,
        source_checks=lambda s: _assert_new_path(s, expect_vinterpret_cast=True),
    )


@pytest.mark.pto
def test_2d_row_continuous(npu):
    """2D j-continuous unit over T.Parallel(4, 256)."""
    m, n = 4, 256
    a = (torch.randn(m, n) * 100).to(torch.int32)
    i = torch.arange(m, dtype=torch.int32).unsqueeze(1)
    j = torch.arange(n, dtype=torch.int32).unsqueeze(0)
    _run_and_check(
        _row_kernel(m, n, 128),
        (a.to(npu),),
        a + i + j,
        source_checks=lambda s: _assert_new_path(s, expect_vinterpret_cast=True, expect_scalar_cast=True),
    )


@pytest.mark.pto
def test_2d_column_continuous(npu):
    """2D i-continuous unit over T.Parallel(256, 4)."""
    m, n = 256, 4
    a = (torch.randn(n, m) * 100).to(torch.int32)
    i = torch.arange(m, dtype=torch.int32).unsqueeze(0)
    j = torch.arange(n, dtype=torch.int32).unsqueeze(1)
    _run_and_check(
        _column_kernel(m, n, 128),
        (a.to(npu),),
        a + j + i,
        source_checks=lambda s: _assert_new_path(s, expect_vinterpret_cast=True, expect_scalar_cast=True),
    )


@pytest.mark.pto
def test_cast_f16_to_f32_1d(npu):
    """1D f16 inputs widened with vcvt before the element-wise mul-add."""
    n = 256
    a = torch.randn(n).half()
    b = torch.randn(n).half()
    _run_and_check(
        _cast_f16_muladd_1d(n, 128),
        (a.to(npu), b.to(npu)),
        a.float() * (a.float() + b.float()),
        source_checks=lambda s: _assert_new_path(s, expect_vcvt_to="f32"),
        atol=1e-3,
    )


@pytest.mark.pto
def test_cast_bf16_to_f32_1d(npu):
    """1D bf16 widened to f32; bf16 does not satisfy ``is_float()`` here, so the
    widening rule must whitelist it explicitly."""
    n = 256
    a = torch.randn(n).bfloat16()
    _run_and_check(
        _cast_bf16_add_1d(n, 128),
        (a.to(npu),),
        a.float() + a.float(),
        source_checks=lambda s: _assert_new_path(s, expect_vcvt_to="f32"),
        atol=1e-2,
    )


@pytest.mark.pto
def test_cast_i16_to_i32_1d(npu):
    """1D int16 -> int32 widening: exact integer arithmetic after vcvt."""
    n = 256
    a = torch.randint(-30000, 30000, (n,), dtype=torch.int16)
    _run_and_check(
        _cast_i16_square_1d(n, 128),
        (a.to(npu),),
        a.int() * a.int(),
        source_checks=lambda s: _assert_new_path(s, expect_vcvt_to="si32"),
    )


@pytest.mark.pto
def test_cast_f16_to_f32_2d(npu):
    """2D j-continuous f16 -> f32 widening over T.Parallel(4, 256)."""
    m, n = 4, 256
    a = torch.randn(m, n).half()
    b = torch.randn(m, n).half()
    _run_and_check(
        _cast_f16_add_2d(m, n, 128),
        (a.to(npu), b.to(npu)),
        a.float() + b.float(),
        source_checks=lambda s: _assert_new_path(s, expect_vcvt_to="f32"),
        atol=1e-3,
    )


@pytest.mark.pto
def test_fp8_element_type_rejected():
    """fp8 is not supported, and the reason is the element
    *category*, not the bit width: fp8 carries its own dtype code, so
    ``is_float()`` is false for it and ``IsSupportedElementDType``'s category
    whitelist (float / bfloat16 / int / uint / bool) does not admit it. Verify
    rejects the buffer at the dtype check, before any Cast lowering is planned;
    the same holds when an fp8 buffer is a load source rather than a cast
    target. Supporting fp8 needs its rounding and saturation semantics plus
    backend support validated first — a whitelist entry alone is not enough.
    See ``test_int8_control_compiles`` for the same bit width inside the
    whitelisted categories."""
    with pytest.raises(Exception) as exc:
        tilelang.compile(_fp8_cast_1d(128, 128), target="pto")
    message = str(exc.value)
    assert "[VerifyParallelToPTO] dtype:" in message, message
    assert "float8_e4m3" in message, message


@pytest.mark.pto
def test_int8_control_compiles():
    """8-bit control: int8 sits in a whitelisted category, so the same width
    lowers through the new path — which is what makes the fp8 rejection above a
    category question rather than a sub-byte/width one."""
    tilelang.compile(_i8_add_1d(128, 128), target="pto")


@pytest.mark.pto
def test_lane_varying_f16_widening_numeric(npu):
    """Reading the half value per lane makes it a vector conversion, and the
    widening is exact."""
    n = 256

    @T.prim_func
    def main(A: T.Buffer((n,), "float16"), C: T.Buffer((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float16")
            c = T.alloc_shared((n,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(n):
                    c[i] = T.cast(a[i], "float32")
            T.copy(c, C)

    x = (torch.arange(n, dtype=torch.float32) - 96.0) / 8.0
    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    assert "pto.vmi.vcvt(" in kernel.get_kernel_source()
    out = kernel(x.to(torch.float16).to(npu))
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), x.to(torch.float16).float())


@pytest.mark.pto
def test_lane_uniform_int_cast_numeric(npu):
    """Integer scalar cast: int32 -> int16 with a lane-uniform source puts
    int16(a[0]) in every lane."""
    n = 256

    @T.prim_func
    def main(A: T.Buffer((n,), "int32"), C: T.Buffer((n,), "int16")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "int32")
            c = T.alloc_shared((n,), "int16")
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(n):
                    c[i] = T.cast(a[0], "int16")
            T.copy(c, C)

    x = torch.arange(n, dtype=torch.int32) - 96
    out = tilelang.compile(main, target="pto", out_idx=-1)(x.to(npu))
    torch.npu.synchronize()
    expected = torch.full((n,), int(x[0].item()), dtype=torch.int16)
    assert torch.equal(out.cpu(), expected), (out.cpu()[:4], expected[:4])


@pytest.mark.pto
def test_pipeline_routes_to_new_passes(npu):
    """Pass-instrument check: the pto pipeline must run Verify/Vectorize and
    must NOT run AscendSimdVFLowerParallel; auto-schedule must inject sync."""

    names = []

    @tvm.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            names.append(info.name.split(".")[-1])

    with tvm.transform.PassContext(opt_level=3, instruments=[Capture()]):
        lower(_vecadd(512, 128), target="pto")

    assert "VerifyParallelToPTO" in names, f"VerifyParallelToPTO not in pipeline: {names}"
    assert "VectorizeParallelToPTO" in names, f"VectorizeParallelToPTO not in pipeline: {names}"
    assert "AscendSimdVFLowerParallel" not in names, f"legacy AscendC pass ran on target=pto: {names}"
    assert "InsertSync" in names, f"InsertSync not in pipeline (auto-schedule off?): {names}"


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
