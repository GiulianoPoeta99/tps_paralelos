"""Utilidades compartidas para el TP3: filtro de Sobel.

La carga/guardado de imagenes queda fuera de las mediciones. Los tiempos se
miden solamente para RGB->gris y para el nucleo Sobel, como pide la consigna.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = ROOT_DIR / "imagenes"
DEFAULT_SIZES = (750, 1500, 3000, 6000)

RgbToGrayFn = Callable[[Any], Any]
SobelFn = Callable[[Any], Any]
RgbLoaderFn = Callable[[Path], Any]


@dataclass(frozen=True)
class SobelMetrics:
    method: str
    size: int
    runs: int
    rgb_gray_s: float
    sobel_s: float
    total_s: float
    white_pct: float
    workers: int
    cpu_logical: int


@dataclass(frozen=True)
class RgbImagePython:
    width: int
    height: int
    data: bytes


@dataclass(frozen=True)
class GrayImagePython:
    width: int
    height: int
    data: bytearray


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("La lista no puede quedar vacia")
    return values


def image_path_for_size(size: int, image_dir: Path = DEFAULT_IMAGE_DIR) -> Path:
    return image_dir / f"IMG_0358_{size}x{size}.jpg"


def load_rgb_image(path: Path) -> Any:
    import numpy as np

    with Image.open(path) as img:
        rgb_img = img.convert("RGB")
        return np.asarray(rgb_img, dtype=np.uint8)


def load_rgb_image_python(path: Path) -> RgbImagePython:
    with Image.open(path) as img:
        rgb_img = img.convert("RGB")
        width, height = rgb_img.size
        data = rgb_img.tobytes()
    return RgbImagePython(width=width, height=height, data=data)


load_rgb_image_list = load_rgb_image_python


def save_gray_image(values: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(values, GrayImagePython):
        Image.frombytes("L", (values.width, values.height), bytes(values.data)).save(path)
        return

    if _is_numpy_array(values):
        import numpy as np

        Image.fromarray(values.astype(np.uint8, copy=False), mode="L").save(path)
        return

    height = len(values)
    width = len(values[0]) if height > 0 else 0
    image = Image.new("L", (width, height))
    image.putdata([int(pixel) for row in values for pixel in row])
    image.save(path)


def white_percentage(values: Any) -> float:
    if isinstance(values, GrayImagePython):
        total = len(values.data)
        if total == 0:
            return 0.0
        return values.data.count(255) * 100.0 / total

    if _is_numpy_array(values):
        import numpy as np

        if values.size == 0:
            return 0.0
        white = int(np.count_nonzero(values == 255))
        return white * 100.0 / int(values.size)

    total = sum(len(row) for row in values)
    if total == 0:
        return 0.0
    white = 0
    for row in values:
        for pixel in row:
            if pixel == 255:
                white += 1
    return white * 100.0 / total


def _is_numpy_array(value: Any) -> bool:
    value_type = type(value)
    return value_type.__module__.split(".", maxsplit=1)[0] == "numpy"


def measure_pipeline(
    *,
    method: str,
    size: int,
    rgb: Any,
    rgb_to_gray: RgbToGrayFn,
    sobel: SobelFn,
    runs: int,
    workers: int,
) -> tuple[SobelMetrics, Any]:
    if runs <= 0:
        raise ValueError("runs debe ser mayor que 0")

    rgb_times: list[float] = []
    sobel_times: list[float] = []
    total_times: list[float] = []
    white_values: list[float] = []
    last_edges: Any | None = None

    for _ in range(runs):
        gc.collect()

        t0 = perf_counter()
        gray = rgb_to_gray(rgb)
        t1 = perf_counter()
        edges = sobel(gray)
        t2 = perf_counter()

        rgb_time = t1 - t0
        sobel_time = t2 - t1
        total_time = t2 - t0

        rgb_times.append(rgb_time)
        sobel_times.append(sobel_time)
        total_times.append(total_time)
        white_values.append(white_percentage(edges))
        last_edges = edges

    if last_edges is None:
        raise RuntimeError("No se ejecuto ninguna corrida")

    metrics = SobelMetrics(
        method=method,
        size=size,
        runs=runs,
        rgb_gray_s=sum(rgb_times) / runs,
        sobel_s=sum(sobel_times) / runs,
        total_s=sum(total_times) / runs,
        white_pct=sum(white_values) / runs,
        workers=workers,
        cpu_logical=os.cpu_count() or 1,
    )
    return metrics, last_edges


def print_metrics(metrics: SobelMetrics) -> None:
    print(f"METODO: {metrics.method}")
    print(f"SIZE: {metrics.size}")
    print(f"RUNS: {metrics.runs}")
    print(f"WORKERS: {metrics.workers}")
    print(f"RGB_GRAY_AVG_SECONDS: {metrics.rgb_gray_s:.9f}")
    print(f"SOBEL_AVG_SECONDS: {metrics.sobel_s:.9f}")
    print(f"TOTAL_AVG_SECONDS: {metrics.total_s:.9f}")
    print(f"WHITE_PERCENT_AVG: {metrics.white_pct:.9f}")
    print("BENCHMARK_JSON:", json.dumps(asdict(metrics), sort_keys=True))


def run_single_method_cli(
    *,
    parser,
    method: str,
    rgb_to_gray: RgbToGrayFn,
    sobel: SobelFn,
    workers: int,
    load_rgb: RgbLoaderFn = load_rgb_image,
) -> None:
    args = parser.parse_args()
    image_path = Path(args.image) if args.image else image_path_for_size(args.size)
    rgb = load_rgb(image_path)

    metrics, edges = measure_pipeline(
        method=method,
        size=args.size,
        rgb=rgb,
        rgb_to_gray=rgb_to_gray,
        sobel=sobel,
        runs=args.runs,
        workers=workers,
    )
    print_metrics(metrics)

    if args.output:
        save_gray_image(edges, Path(args.output))


def add_common_cli_args(parser) -> None:
    parser.add_argument("--size", type=int, default=750, choices=DEFAULT_SIZES)
    parser.add_argument("--image", type=str, default=None, help="Ruta a imagen RGB. Si se omite usa tp3/imagenes.")
    parser.add_argument("--runs", type=int, default=5, help="Cantidad de corridas a promediar.")
    parser.add_argument("--output", type=str, default=None, help="Opcional: guardar la imagen Sobel resultante.")
