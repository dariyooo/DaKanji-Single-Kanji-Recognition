"""Backend support: the models run on XNNPACK via ExecuTorch (int8) and onnxruntime (ONNX)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from char_recognition.models.recognizer import CharRecognizer, ProbabilityModel


def test_xnnpack_int8_lowers_and_runs() -> None:
    pytest.importorskip("torchao")
    pytest.importorskip("executorch")
    from executorch.runtime import Runtime

    from char_recognition.export.loading import CAPTURE_CHANNELS
    from char_recognition.export.xnnpack import export_xnnpack
    from char_recognition.optimize.pt2e import CAPTURE_BATCH, convert_quantized, prepare_xnnpack

    # Probability model captured up front (softmax in the graph); PTQ for test speed.
    model = ProbabilityModel(CharRecognizer(8, backbone="tiny_cnn", image_size=(64, 64)))
    example = (torch.randint(0, 256, (CAPTURE_BATCH, CAPTURE_CHANNELS, 64, 64)).float(),)
    prepared = prepare_xnnpack(model, example, qat=False, dynamic=True)
    prepared(example[0])  # calibrate observers
    converted = convert_quantized(prepared)

    with tempfile.TemporaryDirectory() as tmp:
        pte = export_xnnpack(converted, Path(tmp) / "m.pte", image_size=(64, 64), dynamic=True)
        method = Runtime.get().load_program(str(pte)).load_method("forward")
        # Varying H/W and channels (1=gray, 3=RGB) proves the int8 .pte keeps them dynamic.
        for shape in [(1, 1, 64, 64), (1, 1, 128, 90), (1, 3, 64, 64)]:
            out = method.execute([torch.randint(0, 256, shape).float()])[0]
            assert out.shape == (shape[0], 8)
            assert abs(out.sum().item() - 1.0) < 1e-3  # softmax probabilities


def test_xnnpack_arena_follows_the_declared_bound() -> None:
    """The .pte's memory arena is planned to max_side, not to whatever the runtime is fed.

    Regression test: a 4096px bound with a dynamic batch planned a 4.7 GiB arena, which the
    low-memory killer took out on device. Arena is quadratic in max_side, so this also pins
    batch to 1 (a dynamic batch dim multiplies it again).
    """
    pytest.importorskip("torchao")
    pytest.importorskip("executorch")
    from executorch.exir._serialize._program import deserialize_pte_binary

    from char_recognition.export.loading import CAPTURE_CHANNELS
    from char_recognition.export.xnnpack import export_xnnpack
    from char_recognition.optimize.pt2e import CAPTURE_BATCH, convert_quantized, prepare_xnnpack

    model = ProbabilityModel(CharRecognizer(8, backbone="tiny_cnn", image_size=(64, 64)))
    example = (torch.randint(0, 256, (CAPTURE_BATCH, CAPTURE_CHANNELS, 64, 64)).float(),)
    prepared = prepare_xnnpack(model, example, qat=False, dynamic=True)
    prepared(example[0])  # calibrate observers
    converted = convert_quantized(prepared)

    def arena(path: Path) -> int:
        program = deserialize_pte_binary(path.read_bytes()).program
        return max(max(plan.non_const_buffer_sizes) for plan in program.execution_plan)

    with tempfile.TemporaryDirectory() as tmp:
        sizes = {
            side: arena(
                export_xnnpack(converted, Path(tmp) / f"m{side}.pte", image_size=(64, 64), max_side=side)
            )
            for side in (64, 1024)
        }

    assert sizes[64] < 1024 * 1024, sizes
    assert sizes[1024] < 32 * 1024 * 1024, sizes
    # Quadratic in the bound, so a 16x side ratio is a 256x arena ratio.
    ratio = sizes[1024] / sizes[64]
    assert 200 < ratio < 300, ratio


def test_quantized_graph_survives_a_save_and_relower() -> None:
    """A saved int8 graph re-lowers to any bound at or below its own, so QAT is never redone."""
    pytest.importorskip("torchao")
    pytest.importorskip("executorch")
    from executorch.exir._serialize._program import deserialize_pte_binary

    from char_recognition.export.loading import CAPTURE_CHANNELS
    from char_recognition.export.xnnpack import export_xnnpack, load_quantized, save_quantized
    from char_recognition.optimize.pt2e import CAPTURE_BATCH, convert_quantized, prepare_xnnpack

    model = ProbabilityModel(CharRecognizer(8, backbone="tiny_cnn", image_size=(64, 64)))
    example = (torch.randint(0, 256, (CAPTURE_BATCH, CAPTURE_CHANNELS, 64, 64)).float(),)
    prepared = prepare_xnnpack(model, example, qat=False, dynamic=True, max_side=1024)
    prepared(example[0])  # calibrate observers
    converted = convert_quantized(prepared)

    def arena(path: Path) -> int:
        program = deserialize_pte_binary(path.read_bytes()).program
        return max(max(plan.non_const_buffer_sizes) for plan in program.execution_plan)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        saved = save_quantized(converted, out / "q.pt2", image_size=(64, 64), max_side=1024)
        reloaded = load_quantized(saved)
        for side in (64, 1024):
            direct = export_xnnpack(converted, out / f"d{side}.pte", image_size=(64, 64), max_side=side)
            again = export_xnnpack(reloaded, out / f"r{side}.pte", image_size=(64, 64), max_side=side)
            assert arena(direct) == arena(again), side


def test_xnnpack_onnx_provider_runs() -> None:
    """The ONNX export runs on onnxruntime's XNNPACK execution provider.

    The XNNPACK EP ships only in mobile/custom onnxruntime builds (built with --use_xnnpack);
    a desktop pip wheel silently falls back to CPU. We assert the EP actually loaded and skip
    where it's absent; testing CPU under an XNNPACK label would be misleading.
    """
    import numpy as np

    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    if "XnnpackExecutionProvider" not in ort.get_available_providers():
        pytest.skip("this onnxruntime build has no XNNPACK execution provider")

    from char_recognition.export.onnx import export_onnx

    model = CharRecognizer(8, backbone="tiny_cnn", image_size=(64, 64))
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = export_onnx(model, Path(tmp) / "m.onnx", image_size=(64, 64), dynamic=True)
        session = ort.InferenceSession(
            str(onnx_path), providers=["XnnpackExecutionProvider", "CPUExecutionProvider"]
        )
        assert "XnnpackExecutionProvider" in session.get_providers()  # not a silent CPU fallback
        name = session.get_inputs()[0].name
        # Two different H/W prove the ONNX graph keeps dynamic shapes on XNNPACK.
        for shape in [(1, 1, 64, 64), (1, 1, 128, 90)]:
            out = session.run(None, {name: np.random.randint(0, 256, shape).astype("float32")})[0]
            assert out.shape == (shape[0], 8)
            assert abs(float(out.sum()) - 1.0) < 1e-3  # softmax probabilities
