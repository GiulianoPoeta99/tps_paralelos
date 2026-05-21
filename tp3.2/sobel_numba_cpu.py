"""Version Numba CPU paralela del filtro de Sobel."""

from __future__ import annotations

import argparse
import os

import numpy as np
from numba import config, njit, prange, set_num_threads

from sobel_lib import add_common_cli_args, run_single_method_cli


@njit(parallel=True, cache=True)
def _rgb_to_gray_numba_kernel(rgb: np.ndarray, gray: np.ndarray) -> None:
    height = rgb.shape[0]
    width = rgb.shape[1]
    for y in prange(height):
        for x in range(width):
            r = rgb[y, x, 0]
            g = rgb[y, x, 1]
            b = rgb[y, x, 2]
            value = int(0.299 * r + 0.587 * g + 0.114 * b)
            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            gray[y, x] = value


@njit(parallel=True, cache=True)
def _sobel_numba_kernel(gray: np.ndarray, edges: np.ndarray) -> None:
    height = gray.shape[0]
    width = gray.shape[1]
    for y in prange(1, height - 1):
        for x in range(1, width - 1):
            p00 = np.int32(gray[y - 1, x - 1])
            p01 = np.int32(gray[y - 1, x])
            p02 = np.int32(gray[y - 1, x + 1])
            p10 = np.int32(gray[y, x - 1])
            p12 = np.int32(gray[y, x + 1])
            p20 = np.int32(gray[y + 1, x - 1])
            p21 = np.int32(gray[y + 1, x])
            p22 = np.int32(gray[y + 1, x + 1])

            gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22
            gy = p00 + 2 * p01 + p02 - p20 - 2 * p21 - p22
            magnitude = int((gx * gx + gy * gy) ** 0.5)
            edges[y, x] = 255 if magnitude > 255 else magnitude


def configure_numba_threads(workers: int | None = None) -> int:
    requested = workers or (os.cpu_count() or 1)
    effective = max(1, min(int(requested), int(config.NUMBA_NUM_THREADS)))
    set_num_threads(effective)
    return effective


def rgb_to_gray_numba_cpu(rgb: np.ndarray) -> np.ndarray:
    gray = np.empty((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
    _rgb_to_gray_numba_kernel(rgb, gray)
    return gray


def sobel_numba_cpu(gray: np.ndarray) -> np.ndarray:
    edges = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        _sobel_numba_kernel(gray, edges)
    return edges


def warmup_numba_cpu(workers: int | None = None) -> int:
    effective = configure_numba_threads(workers)
    sample = np.zeros((8, 8, 3), dtype=np.uint8)
    gray = rgb_to_gray_numba_cpu(sample)
    sobel_numba_cpu(gray)
    return effective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_args(parser)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Hilos Numba CPU. Por defecto usa los lógicos disponibles.",
    )
    args, _ = parser.parse_known_args()
    workers = warmup_numba_cpu(args.workers)
    run_single_method_cli(
        parser=parser,
        method="numba_cpu",
        rgb_to_gray=rgb_to_gray_numba_cpu,
        sobel=sobel_numba_cpu,
        workers=workers,
    )


if __name__ == "__main__":
    main()
