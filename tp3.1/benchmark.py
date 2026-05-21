"""Benchmark de la entrega 1 del TP Sobel.

Genera un CSV con una fila por metodo y tamanio de imagen:
secuencial, numpy y numba_cpu.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from sobel_lib import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_SIZES,
    ROOT_DIR,
    image_path_for_size,
    load_rgb_image,
    load_rgb_image_python,
    measure_pipeline,
    parse_int_list,
    save_gray_image,
)
from sobel_numba_cpu import rgb_to_gray_numba_cpu, sobel_numba_cpu, warmup_numba_cpu
from sobel_numpy import rgb_to_gray_numpy, sobel_numpy
from sobel_secuencial import rgb_to_gray_sequential, sobel_sequential


CSV_FIELDS = [
    "tamano",
    "metodo",
    "corridas",
    "tiempo_rgb_gris_s",
    "tiempo_sobel_s",
    "tiempo_total_s",
    "porcentaje_blancos",
    "speed_up",
    "performance_pct",
    "workers",
    "cpu_logicos",
]


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {"secuencial", "numpy", "numba_cpu"}
    unknown = sorted(set(methods) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"Metodos desconocidos: {', '.join(unknown)}")
    return methods


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
    sizes = sorted({int(row["tamano"]) for row in rows})

    lines: list[str] = [
        "# Resultados benchmark - filtro de Sobel",
        "",
        f"Fuente CSV: `{csv_path.name}`.",
        "",
        "Cada fila usa promedios sobre la cantidad de corridas indicada en la columna `corridas` del CSV.",
        "El speed-up se calcula respecto del tiempo total secuencial del mismo tamanio de imagen.",
        "",
    ]

    headers = [
        "Metodo",
        "Tiempo RGB->gris (s)",
        "Tiempo Sobel (s)",
        "Tiempo total (s)",
        "% blancos",
        "Speed-up",
        "Performance (%)",
    ]

    for size in sizes:
        lines.extend(
            [
                f"## Imagen {size}x{size}",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---", "---:", "---:", "---:", "---:", "---:", "---:"]) + " |",
            ]
        )

        for row in rows:
            if int(row["tamano"]) != size:
                continue
            cells = [
                row["metodo"],
                row["tiempo_rgb_gris_s"],
                row["tiempo_sobel_s"],
                row["tiempo_total_s"],
                row["porcentaje_blancos"],
                row["speed_up"],
                row["performance_pct"],
            ]
            lines.append("| " + " | ".join(escape_markdown_cell(cell) for cell in cells) + " |")

        lines.append("")

    lines.extend(
        [
            "## Notas",
            "",
            "- El tiempo total se mide como conversion `RGB->gris` mas aplicacion de Sobel.",
            "- La carga y el guardado de imagenes quedan fuera de las mediciones.",
            "- `performance (%) = speed-up / workers * 100`; los `workers` estan en el CSV.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    *,
    sizes: list[int],
    methods: list[str],
    runs: int,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    workers: int | None = None,
    output: Path = ROOT_DIR / "resultados_sobel_entrega1.csv",
    md_output: Path | None = None,
    save_preview: bool = False,
) -> list[dict[str, object]]:
    workers_value = workers or (os.cpu_count() or 1)
    workers_numba = warmup_numba_cpu(workers_value) if "numba_cpu" in methods else 1
    method_impls = {
        "secuencial": (load_rgb_image_python, rgb_to_gray_sequential, sobel_sequential, 1),
        "numpy": (load_rgb_image, rgb_to_gray_numpy, sobel_numpy, 1),
        "numba_cpu": (load_rgb_image, rgb_to_gray_numba_cpu, sobel_numba_cpu, workers_numba),
    }

    rows: list[dict[str, object]] = []
    for size in sizes:
        path = image_path_for_size(size, image_dir)
        rgb_cache = {}
        size_metrics = []

        for method in methods:
            load_rgb, rgb_to_gray, sobel, workers = method_impls[method]
            if load_rgb not in rgb_cache:
                rgb_cache[load_rgb] = load_rgb(path)
            rgb = rgb_cache[load_rgb]
            print(f"Ejecutando {method} | {size}x{size} | runs={runs} | workers={workers}", flush=True)
            metrics, edges = measure_pipeline(
                method=method,
                size=size,
                rgb=rgb,
                rgb_to_gray=rgb_to_gray,
                sobel=sobel,
                runs=runs,
                workers=workers,
            )
            size_metrics.append((metrics, edges))

            if save_preview:
                save_gray_image(edges, ROOT_DIR / "salidas" / f"sobel_{method}_{size}.png")

        baseline = next((m.total_s for m, _ in size_metrics if m.method == "secuencial"), None)
        if baseline is None:
            baseline = size_metrics[0][0].total_s

        for metrics, _ in size_metrics:
            speed_up = baseline / metrics.total_s if metrics.total_s > 0 else 0.0
            performance_pct = speed_up / metrics.workers * 100.0
            rows.append(
                {
                    "tamano": metrics.size,
                    "metodo": metrics.method,
                    "corridas": metrics.runs,
                    "tiempo_rgb_gris_s": f"{metrics.rgb_gray_s:.9f}",
                    "tiempo_sobel_s": f"{metrics.sobel_s:.9f}",
                    "tiempo_total_s": f"{metrics.total_s:.9f}",
                    "porcentaje_blancos": f"{metrics.white_pct:.9f}",
                    "speed_up": f"{speed_up:.6f}",
                    "performance_pct": f"{performance_pct:.2f}",
                    "workers": metrics.workers,
                    "cpu_logicos": metrics.cpu_logical,
                }
            )

    write_csv(output, rows)
    final_md_output = md_output or output.with_suffix(".md")
    write_markdown(final_md_output, rows, output)
    print(f"CSV escrito en {output}")
    print(f"Markdown escrito en {final_md_output}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=parse_int_list, default=list(DEFAULT_SIZES))
    parser.add_argument("--methods", type=parse_methods, default=["secuencial", "numpy", "numba_cpu"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "resultados_sobel_entrega1.csv")
    parser.add_argument(
        "--md-output",
        type=Path,
        default=None,
        help="Ruta del Markdown de resultados. Por defecto usa el mismo nombre del CSV con extension .md.",
    )
    parser.add_argument("--save-preview", action="store_true", help="Guarda la ultima salida Sobel por metodo/tamanio.")
    args = parser.parse_args()
    run_benchmark(
        sizes=args.sizes,
        methods=args.methods,
        runs=args.runs,
        image_dir=args.image_dir,
        workers=args.workers,
        output=args.output,
        md_output=args.md_output,
        save_preview=args.save_preview,
    )


if __name__ == "__main__":
    main()
