"""Benchmark de la entrega 2: solo el caso nuevo Numba GPU."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from sobel_lib import DEFAULT_IMAGE_DIR, DEFAULT_SIZES, ROOT_DIR, image_path_for_size, load_rgb_image, parse_int_list, save_gray_image
from sobel_numba_gpu import measure_numba_gpu_pipeline


CSV_FIELDS = [
    "tamano",
    "metodo",
    "corridas",
    "tiempo_rgb_gris_s",
    "tiempo_sobel_s",
    "tiempo_total_s",
    "porcentaje_blancos",
    "workers",
    "cpu_logicos",
    "gpu",
    "transfer_h2d_s",
    "transfer_d2h_s",
    "kernel_rgb_gris_s",
    "kernel_sobel_s",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, rows: list[dict[str, object]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Resultados benchmark - Sobel entrega 2",
        "",
        f"Fuente CSV: `{csv_path.name}`.",
        "",
        "Esta entrega mide solo el caso nuevo `numba_gpu`. Los resultados de `secuencial`, `numpy` y `numba_cpu` se toman de la entrega 1 para el informe combinado.",
        "",
        "| Tamaño | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | GPU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        cells = [
            row["tamano"],
            row["tiempo_rgb_gris_s"],
            row["tiempo_sobel_s"],
            row["tiempo_total_s"],
            row["porcentaje_blancos"],
            row["transfer_h2d_s"],
            row["kernel_rgb_gris_s"],
            row["kernel_sobel_s"],
            row["transfer_d2h_s"],
            row["gpu"],
        ]
        lines.append("| " + " | ".join(escape_markdown_cell(cell) for cell in cells) + " |")

    lines.extend(
        [
            "",
            "## Notas",
            "",
            "- La carga y el guardado de imágenes quedan fuera de las mediciones.",
            "- En GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.",
            "- En GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark_entrega2(
    *,
    sizes: list[int],
    runs: int,
    image_dir: Path,
    output: Path,
    md_output: Path,
    save_preview: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for size in sizes:
        path = image_path_for_size(size, image_dir)
        rgb = load_rgb_image(path)

        print(f"Ejecutando numba_gpu | {size}x{size} | runs={runs}", flush=True)
        gpu_metrics, gpu_edges = measure_numba_gpu_pipeline(size=size, rgb=rgb, runs=runs)
        rows.append(
            {
                "tamano": gpu_metrics.size,
                "metodo": gpu_metrics.method,
                "corridas": gpu_metrics.runs,
                "tiempo_rgb_gris_s": f"{gpu_metrics.rgb_gray_s:.9f}",
                "tiempo_sobel_s": f"{gpu_metrics.sobel_s:.9f}",
                "tiempo_total_s": f"{gpu_metrics.total_s:.9f}",
                "porcentaje_blancos": f"{gpu_metrics.white_pct:.9f}",
                "workers": gpu_metrics.gpu_multiprocessors,
                "cpu_logicos": os.cpu_count() or 1,
                "gpu": gpu_metrics.gpu_name,
                "transfer_h2d_s": f"{gpu_metrics.h2d_s:.9f}",
                "transfer_d2h_s": f"{gpu_metrics.d2h_s:.9f}",
                "kernel_rgb_gris_s": f"{gpu_metrics.gray_kernel_s:.9f}",
                "kernel_sobel_s": f"{gpu_metrics.sobel_kernel_s:.9f}",
            }
        )
        if save_preview:
            save_gray_image(gpu_edges, ROOT_DIR / "salidas" / f"sobel_numba_gpu_{size}.png")

    write_csv(output, rows)
    write_markdown(md_output, rows, output)
    print(f"CSV escrito en {output}")
    print(f"Markdown escrito en {md_output}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=parse_int_list, default=list(DEFAULT_SIZES))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "resultados_sobel_entrega2.csv")
    parser.add_argument("--md-output", type=Path, default=ROOT_DIR / "resultados_sobel_entrega2.md")
    parser.add_argument("--save-preview", action="store_true")
    args = parser.parse_args()

    run_benchmark_entrega2(
        sizes=args.sizes,
        runs=args.runs,
        image_dir=args.image_dir,
        output=args.output,
        md_output=args.md_output,
        save_preview=args.save_preview,
    )


if __name__ == "__main__":
    main()
