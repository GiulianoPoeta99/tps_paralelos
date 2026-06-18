"""Filtro sepia para frames BGR de OpenCV.

El video se lee con OpenCV, por eso los frames llegan en orden BGR.
Todas las implementaciones usan la misma formula entera para que la
salida sea comparable entre la version secuencial y PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol


METHOD_LABELS = {
    "secuencial": "secuencial",
    "pytorch_cpu": "PyTorch CPU",
    "pytorch_gpu": "PyTorch GPU",
}


@dataclass
class FilterTiming:
    filter_s: float
    compute_s: float
    transfer_h2d_s: float | None = None
    transfer_d2h_s: float | None = None

    @property
    def transfer_total_s(self) -> float | None:
        if self.transfer_h2d_s is None and self.transfer_d2h_s is None:
            return None
        return (self.transfer_h2d_s or 0.0) + (self.transfer_d2h_s or 0.0)


class SepiaProcessor(Protocol):
    method_key: str
    method_label: str
    workers: int

    def process(self, frame_bgr: Any) -> tuple[Any, FilterTiming]:
        """Devuelve el frame filtrado y los tiempos de filtrado."""


def _validate_frame(frame_bgr: Any) -> None:
    shape = getattr(frame_bgr, "shape", None)
    dtype = getattr(frame_bgr, "dtype", None)
    if shape is None or len(shape) != 3 or shape[2] != 3:
        raise ValueError("El frame debe tener forma HxWx3")
    if str(dtype) != "uint8":
        raise ValueError("El frame debe ser uint8")


def _clamp_u8(value: int) -> int:
    if value < 0:
        return 0
    if value > 255:
        return 255
    return value


def sepia_sequential_bgr(frame_bgr: Any) -> Any:
    """Version secuencial pura: recorre cada pixel con bucles de Python."""
    _validate_frame(frame_bgr)
    flags = getattr(frame_bgr, "flags", {})
    is_contiguous = bool(flags["C_CONTIGUOUS"]) if hasattr(flags, "__getitem__") else True
    source_frame = frame_bgr if is_contiguous else frame_bgr.copy()
    out_frame = frame_bgr.copy()
    raw = memoryview(source_frame).cast("B")
    out = memoryview(out_frame).cast("B")

    for offset in range(0, raw.nbytes, 3):
        b = raw[offset]
        g = raw[offset + 1]
        r = raw[offset + 2]

        out_r = _clamp_u8((393 * r + 769 * g + 189 * b) // 1000)
        out_g = _clamp_u8((349 * r + 686 * g + 168 * b) // 1000)
        out_b = _clamp_u8((272 * r + 534 * g + 131 * b) // 1000)

        # OpenCV espera BGR, no RGB.
        out[offset] = out_b
        out[offset + 1] = out_g
        out[offset + 2] = out_r

    return out_frame


class SequentialSepia:
    method_key = "secuencial"
    method_label = METHOD_LABELS[method_key]
    workers = 1

    def process(self, frame_bgr: Any) -> tuple[Any, FilterTiming]:
        start = perf_counter()
        out = sepia_sequential_bgr(frame_bgr)
        elapsed = perf_counter() - start
        return out, FilterTiming(filter_s=elapsed, compute_s=elapsed)


class TorchSepiaCPU:
    method_key = "pytorch_cpu"
    method_label = METHOD_LABELS[method_key]

    def __init__(self, workers: int | None = None) -> None:
        import torch

        self.torch = torch
        if workers is not None:
            if workers < 1:
                raise ValueError("--workers debe ser >= 1")
            torch.set_num_threads(workers)
        self.workers = int(torch.get_num_threads())

    def process(self, frame_bgr: Any) -> tuple[Any, FilterTiming]:
        _validate_frame(frame_bgr)
        start = perf_counter()
        with self.torch.no_grad():
            frame = self.torch.from_numpy(frame_bgr).to(dtype=self.torch.int32)
            out = sepia_tensor_bgr(frame, self.torch)
            out_host = out.numpy().copy()
        elapsed = perf_counter() - start
        return out_host, FilterTiming(filter_s=elapsed, compute_s=elapsed)


class TorchSepiaGPU:
    method_key = "pytorch_gpu"
    method_label = METHOD_LABELS[method_key]
    workers = 1

    def __init__(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no esta disponible para PyTorch en este entorno")
        self.torch = torch
        self.device = torch.device("cuda")

        # Warmup fuera de la medicion real.
        sample = torch.zeros((4, 4, 3), dtype=torch.uint8, device=self.device)
        sepia_tensor_bgr(sample.to(dtype=torch.int32), torch)
        torch.cuda.synchronize()

    def process(self, frame_bgr: Any) -> tuple[Any, FilterTiming]:
        _validate_frame(frame_bgr)

        transfer_in_start = perf_counter()
        with self.torch.no_grad():
            frame_device = self.torch.from_numpy(frame_bgr).to(self.device, dtype=self.torch.int32)
        self.torch.cuda.synchronize()
        transfer_h2d_s = perf_counter() - transfer_in_start

        compute_start = perf_counter()
        with self.torch.no_grad():
            out_device = sepia_tensor_bgr(frame_device, self.torch)
        self.torch.cuda.synchronize()
        compute_s = perf_counter() - compute_start

        transfer_out_start = perf_counter()
        out_host = out_device.cpu().numpy().copy()
        self.torch.cuda.synchronize()
        transfer_d2h_s = perf_counter() - transfer_out_start

        return out_host, FilterTiming(
            filter_s=transfer_h2d_s + compute_s + transfer_d2h_s,
            compute_s=compute_s,
            transfer_h2d_s=transfer_h2d_s,
            transfer_d2h_s=transfer_d2h_s,
        )


def sepia_tensor_bgr(frame_bgr, torch_module):
    """Sepia para tensores HxWx3 en BGR.

    Usa la misma formula entera que la version secuencial:
    R' = min(255, (393R + 769G + 189B) // 1000)
    G' = min(255, (349R + 686G + 168B) // 1000)
    B' = min(255, (272R + 534G + 131B) // 1000)
    """
    b = frame_bgr[:, :, 0]
    g = frame_bgr[:, :, 1]
    r = frame_bgr[:, :, 2]

    out_r = torch_module.clamp((393 * r + 769 * g + 189 * b) // 1000, min=0, max=255)
    out_g = torch_module.clamp((349 * r + 686 * g + 168 * b) // 1000, min=0, max=255)
    out_b = torch_module.clamp((272 * r + 534 * g + 131 * b) // 1000, min=0, max=255)

    return torch_module.stack((out_b, out_g, out_r), dim=2).to(dtype=torch_module.uint8)


def build_processor(method_key: str, workers: int | None = None) -> SepiaProcessor:
    if method_key == "secuencial":
        return SequentialSepia()
    if method_key == "pytorch_cpu":
        return TorchSepiaCPU(workers=workers)
    if method_key == "pytorch_gpu":
        return TorchSepiaGPU()
    valid = ", ".join(METHOD_LABELS)
    raise ValueError(f"Metodo invalido: {method_key}. Validos: {valid}")
