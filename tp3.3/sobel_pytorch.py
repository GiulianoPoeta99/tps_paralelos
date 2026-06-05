"""Versión PyTorch CPU/GPU del filtro de Sobel."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from sobel_lib import add_common_cli_args, image_path_for_size, load_rgb_image, print_metrics, save_gray_image, white_percentage


SOBEL_X = torch.tensor(
    [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
    dtype=torch.float32,
)
SOBEL_Y = torch.tensor(
    [[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class TorchMetrics:
    method: str
    size: int
    runs: int
    rgb_gray_s: float
    sobel_s: float
    total_s: float
    white_pct: float
    workers: int
    cpu_logical: int
    h2d_s: float
    d2h_s: float
    gray_kernel_s: float
    sobel_kernel_s: float
    gpu_name: str
    torch_version: str
    torch_cuda: str


def set_torch_workers(workers: int | None) -> int:
    if workers is not None:
        if workers < 1:
            raise ValueError("--workers debe ser >= 1")
        torch.set_num_threads(workers)
    return int(torch.get_num_threads())


def ensure_torch_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible para PyTorch en este entorno")


def torch_cuda_info() -> tuple[str, int]:
    ensure_torch_cuda_available()
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return properties.name, int(properties.multi_processor_count)


def sobel_kernels(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return SOBEL_X.to(device), SOBEL_Y.to(device)


def rgb_to_gray_torch(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("La imagen RGB debe tener forma HxWx3")

    rgb_f = rgb.to(torch.float32)
    gray = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    return gray.clamp(0.0, 255.0).to(torch.uint8)


def sobel_torch(gray: torch.Tensor, kernel_x: torch.Tensor, kernel_y: torch.Tensor) -> torch.Tensor:
    if gray.ndim != 2:
        raise ValueError("La imagen en gris debe ser una matriz 2D")

    height, width = gray.shape
    out = torch.zeros((height, width), dtype=torch.uint8, device=gray.device)
    if height < 3 or width < 3:
        return out

    image = gray.to(torch.float32).unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(image, kernel_x)
    gy = F.conv2d(image, kernel_y)
    magnitude = torch.sqrt(gx * gx + gy * gy).clamp(0.0, 255.0)
    out[1:-1, 1:-1] = magnitude.squeeze(0).squeeze(0).to(torch.uint8)
    return out


def warmup_pytorch_cpu() -> None:
    device = torch.device("cpu")
    rgb = torch.zeros((256, 256, 3), dtype=torch.uint8, device=device)
    kernel_x, kernel_y = sobel_kernels(device)
    with torch.no_grad():
        gray = rgb_to_gray_torch(rgb)
        sobel_torch(gray, kernel_x, kernel_y)


def warmup_pytorch_gpu() -> None:
    ensure_torch_cuda_available()
    device = torch.device("cuda")
    rgb = torch.zeros((256, 256, 3), dtype=torch.uint8, device=device)
    kernel_x, kernel_y = sobel_kernels(device)
    with torch.no_grad():
        gray = rgb_to_gray_torch(rgb)
        sobel_torch(gray, kernel_x, kernel_y)
    torch.cuda.synchronize()


def measure_pytorch_cpu_pipeline(
    *,
    size: int,
    rgb: np.ndarray,
    runs: int,
    workers: int | None,
) -> tuple[TorchMetrics, np.ndarray]:
    if runs <= 0:
        raise ValueError("runs debe ser mayor que 0")

    effective_workers = set_torch_workers(workers)
    warmup_pytorch_cpu()

    rgb_tensor = torch.from_numpy(rgb.copy())
    kernel_x, kernel_y = sobel_kernels(torch.device("cpu"))

    rgb_times: list[float] = []
    sobel_times: list[float] = []
    total_times: list[float] = []
    white_values: list[float] = []
    last_edges: np.ndarray | None = None

    for _ in range(runs):
        gc.collect()

        with torch.no_grad():
            t0 = perf_counter()
            gray = rgb_to_gray_torch(rgb_tensor)
            t1 = perf_counter()
            edges_tensor = sobel_torch(gray, kernel_x, kernel_y)
            t2 = perf_counter()

        edges = edges_tensor.numpy()
        rgb_times.append(t1 - t0)
        sobel_times.append(t2 - t1)
        total_times.append(t2 - t0)
        white_values.append(white_percentage(edges))
        last_edges = edges

    if last_edges is None:
        raise RuntimeError("No se ejecutó ninguna corrida")

    metrics = TorchMetrics(
        method="pytorch_cpu",
        size=size,
        runs=runs,
        rgb_gray_s=sum(rgb_times) / runs,
        sobel_s=sum(sobel_times) / runs,
        total_s=sum(total_times) / runs,
        white_pct=sum(white_values) / runs,
        workers=effective_workers,
        cpu_logical=effective_workers,
        h2d_s=0.0,
        d2h_s=0.0,
        gray_kernel_s=sum(rgb_times) / runs,
        sobel_kernel_s=sum(sobel_times) / runs,
        gpu_name="CPU",
        torch_version=torch.__version__,
        torch_cuda=str(torch.version.cuda or "no disponible"),
    )
    return metrics, last_edges


def measure_pytorch_gpu_pipeline(*, size: int, rgb: np.ndarray, runs: int) -> tuple[TorchMetrics, np.ndarray]:
    if runs <= 0:
        raise ValueError("runs debe ser mayor que 0")

    ensure_torch_cuda_available()
    warmup_pytorch_gpu()
    gpu_name, gpu_multiprocessors = torch_cuda_info()
    device = torch.device("cuda")

    rgb_host = torch.from_numpy(rgb.copy())
    kernel_x, kernel_y = sobel_kernels(device)

    h2d_times: list[float] = []
    d2h_times: list[float] = []
    gray_kernel_times: list[float] = []
    sobel_kernel_times: list[float] = []
    rgb_gray_times: list[float] = []
    sobel_times: list[float] = []
    total_times: list[float] = []
    white_values: list[float] = []
    last_edges: np.ndarray | None = None

    for _ in range(runs):
        gc.collect()
        torch.cuda.empty_cache()

        t0 = perf_counter()
        rgb_device = rgb_host.to(device)
        torch.cuda.synchronize()
        t1 = perf_counter()

        with torch.no_grad():
            gray_device = rgb_to_gray_torch(rgb_device)
            torch.cuda.synchronize()
            t2 = perf_counter()

            edges_device = sobel_torch(gray_device, kernel_x, kernel_y)
            torch.cuda.synchronize()
            t3 = perf_counter()

        edges = edges_device.cpu().numpy()
        torch.cuda.synchronize()
        t4 = perf_counter()

        h2d_time = t1 - t0
        gray_kernel_time = t2 - t1
        sobel_kernel_time = t3 - t2
        d2h_time = t4 - t3
        rgb_gray_time = h2d_time + gray_kernel_time
        sobel_time = sobel_kernel_time + d2h_time
        total_time = t4 - t0

        h2d_times.append(h2d_time)
        gray_kernel_times.append(gray_kernel_time)
        sobel_kernel_times.append(sobel_kernel_time)
        d2h_times.append(d2h_time)
        rgb_gray_times.append(rgb_gray_time)
        sobel_times.append(sobel_time)
        total_times.append(total_time)
        white_values.append(white_percentage(edges))
        last_edges = edges

    if last_edges is None:
        raise RuntimeError("No se ejecutó ninguna corrida")

    metrics = TorchMetrics(
        method="pytorch_gpu",
        size=size,
        runs=runs,
        rgb_gray_s=sum(rgb_gray_times) / runs,
        sobel_s=sum(sobel_times) / runs,
        total_s=sum(total_times) / runs,
        white_pct=sum(white_values) / runs,
        workers=gpu_multiprocessors,
        cpu_logical=gpu_multiprocessors,
        h2d_s=sum(h2d_times) / runs,
        d2h_s=sum(d2h_times) / runs,
        gray_kernel_s=sum(gray_kernel_times) / runs,
        sobel_kernel_s=sum(sobel_kernel_times) / runs,
        gpu_name=gpu_name,
        torch_version=torch.__version__,
        torch_cuda=str(torch.version.cuda or "no disponible"),
    )
    return metrics, last_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_args(parser)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=None, help="Hilos PyTorch CPU.")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else image_path_for_size(args.size)
    rgb = load_rgb_image(image_path)

    if args.device == "cuda":
        metrics, edges = measure_pytorch_gpu_pipeline(size=args.size, rgb=rgb, runs=args.runs)
    else:
        metrics, edges = measure_pytorch_cpu_pipeline(size=args.size, rgb=rgb, runs=args.runs, workers=args.workers)

    print_metrics(
        type(
            "MetricsCompat",
            (),
            {
                "method": metrics.method,
                "size": metrics.size,
                "runs": metrics.runs,
                "workers": metrics.workers,
                "rgb_gray_s": metrics.rgb_gray_s,
                "sobel_s": metrics.sobel_s,
                "total_s": metrics.total_s,
                "white_pct": metrics.white_pct,
                "cpu_logical": metrics.cpu_logical,
            },
        )()
    )

    if args.output:
        save_gray_image(edges, Path(args.output))


if __name__ == "__main__":
    main()
