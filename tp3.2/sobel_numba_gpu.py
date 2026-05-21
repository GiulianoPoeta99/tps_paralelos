"""Version Numba GPU/CUDA del filtro de Sobel."""

from __future__ import annotations

import argparse
import gc
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import cuda

from sobel_lib import add_common_cli_args, load_rgb_image, print_metrics, save_gray_image, white_percentage


THREADS_PER_BLOCK = (16, 16)

gx_kernel = (
    (-1, 0, 1),
    (-2, 0, 2),
    (-1, 0, 1),
)
gy_kernel = (
    (1, 2, 1),
    (0, 0, 0),
    (-1, -2, -1),
)


@dataclass(frozen=True)
class NumbaGpuMetrics:
    method: str
    size: int
    runs: int
    rgb_gray_s: float
    sobel_s: float
    total_s: float
    white_pct: float
    h2d_s: float
    d2h_s: float
    gray_kernel_s: float
    sobel_kernel_s: float
    gpu_name: str
    gpu_multiprocessors: int


@cuda.jit
def rgb_to_gray_cuda(rgb: np.ndarray, gray: np.ndarray) -> None:
    # Cada hilo obtiene una coordenada (y, x) única en una grilla 2D.
    y, x = cuda.grid(2)
    height = rgb.shape[0]
    width = rgb.shape[1]
    if y >= height or x >= width:
        return

    # Un hilo procesa un solo píxel: el recorrido completo lo hace la grilla.
    r = float(rgb[y, x, 0])
    g = float(rgb[y, x, 1])
    b = float(rgb[y, x, 2])

    i = int(0.299 * r + 0.587 * g + 0.114 * b)
    i = 0 if i < 0 else 255 if i > 255 else i
    gray[y, x] = i


@cuda.jit
def sobel_cuda(gray: np.ndarray, out: np.ndarray) -> None:
    # Igual que en rgb_to_gray_cuda: cada hilo trabaja sobre un píxel (y, x).
    y, x = cuda.grid(2)
    height = gray.shape[0]
    width = gray.shape[1]
    if y >= height or x >= width:
        return

    if y == 0 or x == 0 or y == height - 1 or x == width - 1:
        out[y, x] = 0
        return

    gx = 0
    gy = 0

    # Estos for recorren solo la vecindad 3x3 del píxel asignado al hilo.
    for ky in range(3):
        for kx in range(3):
            p = int(gray[y + ky - 1, x + kx - 1])
            gx += p * gx_kernel[ky][kx]
            gy += p * gy_kernel[ky][kx]

    mag = int(math.sqrt(gx * gx + gy * gy))
    out[y, x] = 255 if mag > 255 else mag


def ensure_cuda_available() -> None:
    if not cuda.is_available():
        raise RuntimeError(
            "Numba CUDA no esta disponible en este equipo. "
            "Hace falta una GPU compatible y driver CUDA instalado."
        )


def gpu_info() -> tuple[str, int]:
    ensure_cuda_available()
    device = cuda.get_current_device()
    name = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
    multiprocessors = int(getattr(device, "MULTIPROCESSOR_COUNT", 1))
    return name, multiprocessors


def _blocks_for_shape(height: int, width: int) -> tuple[int, int]:
    threads_y, threads_x = THREADS_PER_BLOCK
    return ((height + threads_y - 1) // threads_y, (width + threads_x - 1) // threads_x)


def warmup_numba_gpu() -> None:
    ensure_cuda_available()
    # Warmup grande para evitar warnings de baja ocupacion por grilla 1x1.
    sample = np.zeros((256, 256, 3), dtype=np.uint8)
    d_rgb = cuda.to_device(sample)
    d_gray = cuda.device_array((256, 256), dtype=np.uint8)
    d_edges = cuda.device_array((256, 256), dtype=np.uint8)
    blocks = _blocks_for_shape(256, 256)
    rgb_to_gray_cuda[blocks, THREADS_PER_BLOCK](d_rgb, d_gray)
    sobel_cuda[blocks, THREADS_PER_BLOCK](d_gray, d_edges)
    cuda.synchronize()


def rgb_to_gray_numba_gpu(rgb: np.ndarray):
    ensure_cuda_available()
    height, width, _ = rgb.shape
    d_rgb = cuda.to_device(rgb)
    d_gray = cuda.device_array((height, width), dtype=np.uint8)
    rgb_to_gray_cuda[_blocks_for_shape(height, width), THREADS_PER_BLOCK](d_rgb, d_gray)
    cuda.synchronize()
    return d_gray


def sobel_numba_gpu(d_gray) -> np.ndarray:
    ensure_cuda_available()
    height, width = d_gray.shape
    d_edges = cuda.device_array((height, width), dtype=np.uint8)
    sobel_cuda[_blocks_for_shape(height, width), THREADS_PER_BLOCK](d_gray, d_edges)
    cuda.synchronize()
    edges = d_edges.copy_to_host()
    cuda.synchronize()
    return edges


def measure_numba_gpu_pipeline(*, size: int, rgb: np.ndarray, runs: int) -> tuple[NumbaGpuMetrics, np.ndarray]:
    if runs <= 0:
        raise ValueError("runs debe ser mayor que 0")

    warmup_numba_gpu()
    gpu_name, gpu_multiprocessors = gpu_info()

    h2d_times: list[float] = []
    gray_kernel_times: list[float] = []
    sobel_kernel_times: list[float] = []
    d2h_times: list[float] = []
    rgb_gray_times: list[float] = []
    sobel_times: list[float] = []
    total_times: list[float] = []
    white_values: list[float] = []
    last_edges: np.ndarray | None = None

    height, width, _ = rgb.shape
    blocks = _blocks_for_shape(height, width)

    for _ in range(runs):
        gc.collect()

        t0 = perf_counter()
        d_rgb = cuda.to_device(rgb)
        cuda.synchronize()
        t1 = perf_counter()

        d_gray = cuda.device_array((height, width), dtype=np.uint8)
        rgb_to_gray_cuda[blocks, THREADS_PER_BLOCK](d_rgb, d_gray)
        cuda.synchronize()
        t2 = perf_counter()

        d_edges = cuda.device_array((height, width), dtype=np.uint8)
        sobel_cuda[blocks, THREADS_PER_BLOCK](d_gray, d_edges)
        cuda.synchronize()
        t3 = perf_counter()

        edges = d_edges.copy_to_host()
        cuda.synchronize()
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
        raise RuntimeError("No se ejecuto ninguna corrida")

    metrics = NumbaGpuMetrics(
        method="numba_gpu",
        size=size,
        runs=runs,
        rgb_gray_s=sum(rgb_gray_times) / runs,
        sobel_s=sum(sobel_times) / runs,
        total_s=sum(total_times) / runs,
        white_pct=sum(white_values) / runs,
        h2d_s=sum(h2d_times) / runs,
        d2h_s=sum(d2h_times) / runs,
        gray_kernel_s=sum(gray_kernel_times) / runs,
        sobel_kernel_s=sum(sobel_kernel_times) / runs,
        gpu_name=gpu_name,
        gpu_multiprocessors=gpu_multiprocessors,
    )
    return metrics, last_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_args(parser)
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else None
    if image_path is None:
        from sobel_lib import image_path_for_size

        image_path = image_path_for_size(args.size)

    rgb = load_rgb_image(image_path)
    metrics, edges = measure_numba_gpu_pipeline(size=args.size, rgb=rgb, runs=args.runs)
    print_metrics(
        type(
            "MetricsCompat",
            (),
            {
                "method": metrics.method,
                "size": metrics.size,
                "runs": metrics.runs,
                "workers": metrics.gpu_multiprocessors,
                "rgb_gray_s": metrics.rgb_gray_s,
                "sobel_s": metrics.sobel_s,
                "total_s": metrics.total_s,
                "white_pct": metrics.white_pct,
                "cpu_logical": metrics.gpu_multiprocessors,
            },
        )()
    )
    print(f"GPU: {metrics.gpu_name}")
    print(f"H2D_AVG_SECONDS: {metrics.h2d_s:.9f}")
    print(f"D2H_AVG_SECONDS: {metrics.d2h_s:.9f}")
    print(f"GRAY_KERNEL_AVG_SECONDS: {metrics.gray_kernel_s:.9f}")
    print(f"SOBEL_KERNEL_AVG_SECONDS: {metrics.sobel_kernel_s:.9f}")

    if args.output:
        save_gray_image(edges, Path(args.output))


if __name__ == "__main__":
    main()
