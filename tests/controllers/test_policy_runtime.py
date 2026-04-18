from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper
import torch

from booster_deploy.controllers.base_controller import Policy


class _DummyController:
    def __init__(self) -> None:
        self._step_count = 0

    def get_time(self) -> float:
        return 0.0


class _TestPolicy(Policy):
    def reset(self) -> None:
        pass

    def inference(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.float32)


def _make_cfg(*, checkpoint_path: str = "", stat_log_path: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_path=checkpoint_path,
        device="cpu",
        enable_safety_fallback=True,
        stat_log_path=stat_log_path,
        stat_log_inference_io=False,
        onnx_providers=["CPUExecutionProvider"],
    )


def _save_torchscript_model(path: Path) -> None:
    class _TinyTorchModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

    scripted = torch.jit.script(_TinyTorchModel())
    scripted.save(str(path))


def _save_onnx_model(path: Path) -> None:
    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 3])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Identity", inputs=["obs"], outputs=["action"])
    graph = helper.make_graph([node], "policy_runtime_test", [obs], [action])
    model = helper.make_model(
        graph,
        producer_name="policy_runtime_test",
        ir_version=11,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    onnx.save_model(model, str(path))


class TestPolicyRuntime(unittest.TestCase):
    def test_initialize_model_runtime_supports_torchscript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model.pt"
            _save_torchscript_model(ckpt_path)

            policy = _TestPolicy(_make_cfg(checkpoint_path=str(ckpt_path)), _DummyController())
            policy.initialize_model_runtime(str(ckpt_path))

            self.assertEqual(policy._backend, "torch")
            self.assertIsNotNone(policy._model)

    def test_initialize_model_runtime_supports_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model.onnx"
            _save_onnx_model(ckpt_path)

            policy = _TestPolicy(_make_cfg(checkpoint_path=str(ckpt_path)), _DummyController())
            policy.initialize_model_runtime(str(ckpt_path))

            self.assertEqual(policy._backend, "onnx")
            self.assertIsNotNone(policy._onnx_session)
            self.assertEqual(policy._onnx_input_names, ["obs"])
            self.assertEqual(policy._onnx_output_names, ["action"])

    def test_prepare_onnx_inputs_adds_batch_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model.onnx"
            _save_onnx_model(ckpt_path)

            policy = _TestPolicy(_make_cfg(checkpoint_path=str(ckpt_path)), _DummyController())
            policy.initialize_model_runtime(str(ckpt_path))
            feed = policy.prepare_onnx_inputs(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

            self.assertEqual(list(feed), ["obs"])
            self.assertEqual(feed["obs"].shape, (1, 3))
            self.assertEqual(feed["obs"].dtype, np.float32)

    def test_parse_onnx_outputs_returns_tensor(self) -> None:
        policy = _TestPolicy(_make_cfg(), _DummyController())
        parsed = policy.parse_onnx_outputs([np.array([[1.0, 2.0]], dtype=np.float32)])

        self.assertIsInstance(parsed, torch.Tensor)
        self.assertEqual(parsed.shape, (1, 2))
        self.assertEqual(parsed.dtype, torch.float32)

    def test_log_helpers_write_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stats_path = Path(tmp_dir) / "logs" / "stats.csv"
            controller = _DummyController()
            controller._step_count = 7
            policy = _TestPolicy(_make_cfg(stat_log_path=str(stats_path)), controller)

            policy.log_named_vector("commands", torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32))
            policy.log_stats({"inference_output": torch.tensor([[0.4, 0.6]], dtype=torch.float32)})

            self.assertTrue(stats_path.exists())
            content = stats_path.read_text(encoding="utf-8")
            self.assertIn("commands", content)
            self.assertIn("inference_output", content)


if __name__ == "__main__":
    unittest.main()
