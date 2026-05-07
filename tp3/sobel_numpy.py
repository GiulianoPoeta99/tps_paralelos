"""Version NumPy vectorizada del filtro de Sobel."""

from __future__ import annotations

import argparse

import numpy as np

from sobel_lib import add_common_cli_args, run_single_method_cli


def rgb_to_gray_numpy(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    gray = 0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def sobel_numpy(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    edges = np.zeros((height, width), dtype=np.uint8)
    if height < 3 or width < 3:
        return edges

    g = gray.astype(np.float32)
    gx = (
        -g[:-2, :-2]
        + g[:-2, 2:]
        - 2.0 * g[1:-1, :-2]
        + 2.0 * g[1:-1, 2:]
        - g[2:, :-2]
        + g[2:, 2:]
    )
    gy = (
        g[:-2, :-2]
        + 2.0 * g[:-2, 1:-1]
        + g[:-2, 2:]
        - g[2:, :-2]
        - 2.0 * g[2:, 1:-1]
        - g[2:, 2:]
    )
    magnitude = np.sqrt(gx * gx + gy * gy)
    edges[1:-1, 1:-1] = np.clip(magnitude, 0, 255).astype(np.uint8)
    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_args(parser)
    run_single_method_cli(
        parser=parser,
        method="numpy",
        rgb_to_gray=rgb_to_gray_numpy,
        sobel=sobel_numpy,
        workers=1,
    )


if __name__ == "__main__":
    main()
