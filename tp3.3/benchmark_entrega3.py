"""Benchmark de la entrega 3: solo los casos nuevos PyTorch CPU/GPU."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from sobel_lib import DEFAULT_IMAGE_DIR, DEFAULT_SIZES, ROOT_DIR, image_path_for_size, load_rgb_image, parse_int_list, save_gray_image
from sobel_pytorch import measure_pytorch_cpu_pipeline, measure_pytorch_gpu_pipeline


METHODS = ("pytorch_cpu", "pytorch_gpu")

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
    "torch_version",
    "torch_cuda",
    "transfer_h2d_s",
    "transfer_d2h_s",
    "kernel_rgb_gris_s",
    "kernel_sobel_s",
]


def parse_method_list(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Métodos no soportados: {', '.join(invalid)}")
    if not methods:
        raise argparse.ArgumentTypeError("La lista de métodos no puede quedar vacía")
    return methods


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle) if row.get("tamano") and row.get("metodo")]


def merge_rows(existing_rows: list[dict[str, object]], new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[int, str], dict[str, object]] = {}
    for row in existing_rows + new_rows:
        key = (int(row["tamano"]), str(row["metodo"]))
        merged[key] = row

    return [merged[key] for key in sorted(merged, key=lambda item: (item[0], METHODS.index(item[1])))]


def escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, rows: list[dict[str, object]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Resultados benchmark - Sobel entrega 3",
        "",
        f"Fuente CSV: `{csv_path.name}`.",
        "",
        "Esta entrega mide solo los casos nuevos `pytorch_cpu` y `pytorch_gpu`. "
        "Los resultados de las entregas anteriores se toman desde sus CSV para el informe combinado.",
        "",
        "| Tamaño | Método | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | Dispositivo |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        cells = [
            f"{row['tamano']}x{row['tamano']}",
            row["metodo"],
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
            "- En PyTorch CPU, los tensores se preparan fuera de la medición y se mide el cómputo sobre CPU.",
            "- En PyTorch GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.",
            "- En PyTorch GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _row_from_metrics(metrics) -> dict[str, object]:
    return {
        "tamano": metrics.size,
        "metodo": metrics.method,
        "corridas": metrics.runs,
        "tiempo_rgb_gris_s": f"{metrics.rgb_gray_s:.9f}",
        "tiempo_sobel_s": f"{metrics.sobel_s:.9f}",
        "tiempo_total_s": f"{metrics.total_s:.9f}",
        "porcentaje_blancos": f"{metrics.white_pct:.9f}",
        "workers": metrics.workers,
        "cpu_logicos": os.cpu_count() or 1,
        "gpu": metrics.gpu_name,
        "torch_version": metrics.torch_version,
        "torch_cuda": metrics.torch_cuda,
        "transfer_h2d_s": f"{metrics.h2d_s:.9f}",
        "transfer_d2h_s": f"{metrics.d2h_s:.9f}",
        "kernel_rgb_gris_s": f"{metrics.gray_kernel_s:.9f}",
        "kernel_sobel_s": f"{metrics.sobel_kernel_s:.9f}",
    }


def run_benchmark_entrega3(
    *,
    sizes: list[int],
    runs: int,
    methods: list[str],
    image_dir: Path,
    output: Path,
    md_output: Path,
    save_preview: bool,
    workers: int | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for size in sizes:
        path = image_path_for_size(size, image_dir)
        rgb = load_rgb_image(path)

        if "pytorch_cpu" in methods:
            print(f"Ejecutando pytorch_cpu | {size}x{size} | runs={runs} | workers={workers or 'torch_default'}", flush=True)
            cpu_metrics, cpu_edges = measure_pytorch_cpu_pipeline(size=size, rgb=rgb, runs=runs, workers=workers)
            rows.append(_row_from_metrics(cpu_metrics))
            if save_preview:
                save_gray_image(cpu_edges, ROOT_DIR / "salidas" / f"sobel_pytorch_cpu_{size}.png")

        if "pytorch_gpu" in methods:
            print(f"Ejecutando pytorch_gpu | {size}x{size} | runs={runs}", flush=True)
            gpu_metrics, gpu_edges = measure_pytorch_gpu_pipeline(size=size, rgb=rgb, runs=runs)
            rows.append(_row_from_metrics(gpu_metrics))
            if save_preview:
                save_gray_image(gpu_edges, ROOT_DIR / "salidas" / f"sobel_pytorch_gpu_{size}.png")

    rows = merge_rows(read_existing_rows(output), rows)
    write_csv(output, rows)
    write_markdown(md_output, rows, output)
    print(f"CSV escrito en {output}")
    print(f"Markdown escrito en {md_output}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=parse_int_list, default=list(DEFAULT_SIZES))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--methods", type=parse_method_list, default=list(METHODS))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "resultados_sobel_entrega3.csv")
    parser.add_argument("--md-output", type=Path, default=ROOT_DIR / "resultados_sobel_entrega3.md")
    parser.add_argument("--save-preview", action="store_true")
    args = parser.parse_args()

    run_benchmark_entrega3(
        sizes=args.sizes,
        runs=args.runs,
        methods=args.methods,
        image_dir=args.image_dir,
        output=args.output,
        md_output=args.md_output,
        save_preview=args.save_preview,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
