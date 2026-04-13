"""
Chrona Deployment Utilities
ONNX export, TensorRT optimization, quantization.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union

from chrona.models.hybrid_model import ChronaModel, ModelConfig


def export_onnx(
    model: ChronaModel,
    output_path: Union[str, Path] = "chrona.onnx",
    context_len: int = 512,
    opset: int = 17,
    optimize: bool = True,
) -> Path:
    """Export Chrona to ONNX with dynamic batch + sequence axes."""
    model.eval()
    dummy_ts = torch.randn(1, context_len, model.cfg.input_dim)
    dummy_tf = torch.randn(1, context_len, 4)

    output_path = Path(output_path)
    torch.onnx.export(
        model,
        (dummy_ts, dummy_tf),
        str(output_path),
        opset_version=opset,
        input_names=["past_values", "time_features"],
        output_names=["forecast_mean", "forecast_std", "forecast_quantiles"],
        dynamic_axes={
            "past_values":         {0: "batch", 1: "sequence"},
            "time_features":       {0: "batch", 1: "sequence"},
            "forecast_mean":       {0: "batch"},
            "forecast_std":        {0: "batch"},
            "forecast_quantiles":  {0: "batch"},
        },
        do_constant_folding=True,
    )

    if optimize:
        try:
            import onnx
            from onnxsim import simplify
            model_onnx = onnx.load(str(output_path))
            simplified, ok = simplify(model_onnx)
            if ok:
                onnx.save(simplified, str(output_path))
                print(f"[ONNX] Simplified model saved.")
        except ImportError:
            print("[ONNX] onnxsim not installed — skipping simplification (pip install onnxsim).")

    print(f"[ONNX] Exported → {output_path}")
    return output_path


def onnx_inference(
    onnx_path: Union[str, Path],
    series: np.ndarray,
) -> dict:
    """Run inference with ONNX Runtime (2-5x faster than PyTorch CPU)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    if series.ndim == 1:
        series = series[:, None]
    ts = series[None].astype(np.float32)          # (1, T, D)
    tf = np.zeros((1, ts.shape[1], 4), np.float32) # dummy time features

    mean, std, quantiles = sess.run(None, {"past_values": ts, "time_features": tf})
    return {"mean": mean[0], "std": std[0], "quantiles": quantiles[0]}


def quantize_model(model: ChronaModel, scheme: str = "dynamic") -> ChronaModel:
    """
    Post-training quantization.
    scheme: 'dynamic' (fast, CPU), 'static' (better, needs calibration data)
    """
    model.eval()
    if scheme == "dynamic":
        quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        before = sum(p.numel() * p.element_size() for p in model.parameters())
        after  = sum(p.numel() * p.element_size() for p in quantized.parameters())
        print(f"[Quantize] {before/1e6:.1f}MB → {after/1e6:.1f}MB ({100*(1-after/before):.0f}% reduction)")
        return quantized
    raise ValueError(f"Unknown scheme: {scheme}")


def export_torchscript(
    model: ChronaModel,
    output_path: Union[str, Path] = "chrona.pt",
    context_len: int = 512,
) -> Path:
    """TorchScript export for C++ / mobile inference."""
    model.eval()
    dummy = torch.randn(1, context_len, model.cfg.input_dim)
    scripted = torch.jit.trace(model, dummy)
    scripted.save(str(output_path))
    print(f"[TorchScript] Exported → {output_path}")
    return Path(output_path)
