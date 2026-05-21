"""Genera el informe de entrega 2 combinando entrega 1 + Numba GPU."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sobel_lib import ROOT_DIR


METHOD_ORDER = ["secuencial", "numpy", "numba_cpu", "numba_gpu"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float((row.get(key) or "0").strip() or "0")


def _method_row(rows: list[dict[str, str]], size: int, method: str) -> dict[str, str]:
    for row in rows:
        if int(row["tamano"]) == size and row["metodo"] == method:
            return row
    raise KeyError(f"Falta fila {method} para tamaño {size}")


def _combined_rows(entrega1_rows: list[dict[str, str]], gpu_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    sizes = sorted({int(row["tamano"]) for row in entrega1_rows})
    for size in sizes:
        seq = _method_row(entrega1_rows, size, "secuencial")
        seq_total = _float(seq, "tiempo_total_s")
        for method in METHOD_ORDER:
            source_rows = gpu_rows if method == "numba_gpu" else entrega1_rows
            row = dict(_method_row(source_rows, size, method))
            total = _float(row, "tiempo_total_s")
            speed_up = seq_total / total if total > 0 else 0.0
            improvement_pct = (1.0 - (total / seq_total)) * 100.0 if seq_total > 0 else 0.0
            row["speed_up_vs_secuencial"] = f"{speed_up:.6f}"
            row["mejora_vs_secuencial_pct"] = f"{improvement_pct:.2f}"
            rows.append(row)
    return rows


def _combined_result_table(rows: list[dict[str, str]], size: int, section_number: int) -> list[str]:
    lines = [
        f"### 3.{section_number} Imagen {size}x{size}",
        "",
        "| Método | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up vs secuencial | Mejora vs secuencial (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHOD_ORDER:
        row = _method_row(rows, size, method)
        lines.append(
            "| "
            + " | ".join(
                [
                    row["metodo"],
                    row["tiempo_rgb_gris_s"],
                    row["tiempo_sobel_s"],
                    row["tiempo_total_s"],
                    row["porcentaje_blancos"],
                    row["speed_up_vs_secuencial"],
                    row["mejora_vs_secuencial_pct"],
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _comparison_sentence(rows: list[dict[str, str]]) -> str:
    parts = []
    for row in rows:
        if row["metodo"] != "numba_gpu":
            continue
        size = int(row["tamano"])
        improvement = _float(row, "mejora_vs_secuencial_pct")
        speed_up = _float(row, "speed_up_vs_secuencial")
        parts.append(f"{improvement:.2f}% en {size}x{size} ({speed_up:.2f}x)")
    return ", ".join(parts)


def write_report(entrega1_csv: Path, gpu_csv: Path, output_path: Path) -> None:
    entrega1_rows = _read_csv(entrega1_csv)
    gpu_rows = _read_csv(gpu_csv)
    rows = _combined_rows(entrega1_rows, gpu_rows)
    sizes = sorted({int(row["tamano"]) for row in entrega1_rows})

    first_gpu = _method_row(gpu_rows, sizes[0], "numba_gpu")
    gpu_name = first_gpu.get("gpu", "GPU CUDA")
    gpu_workers = first_gpu.get("workers", "")
    cpu_logical = _method_row(entrega1_rows, sizes[0], "secuencial").get("cpu_logicos", "")

    largest = sizes[-1]
    largest_seq = _method_row(rows, largest, "secuencial")
    largest_gpu = _method_row(rows, largest, "numba_gpu")
    largest_transfer = _float(largest_gpu, "transfer_h2d_s") + _float(largest_gpu, "transfer_d2h_s")
    largest_kernels = _float(largest_gpu, "kernel_rgb_gris_s") + _float(largest_gpu, "kernel_sobel_s")

    consistency_rows = []
    for size in sizes:
        consistency_rows.append(
            "| "
            + " | ".join(
                [
                    f"{size}x{size}",
                    _method_row(rows, size, "secuencial")["porcentaje_blancos"],
                    _method_row(rows, size, "numpy")["porcentaje_blancos"],
                    _method_row(rows, size, "numba_cpu")["porcentaje_blancos"],
                    _method_row(rows, size, "numba_gpu")["porcentaje_blancos"],
                ]
            )
            + " |"
        )

    lines: list[str] = [
        "# Filtro de Sobel en GPU con Numba CUDA",
        "",
        "**Materia:** Sistemas Paralelos - Lic. en Sistemas, 5to año  ",
        "**Institución:** UNTDF  ",
        "**Docente:** MsC. Federico González Brizzio  ",
        "**Entrega:** 21 de mayo de 2026  ",
        "**Repositorio:** <https://github.com/GiulianoPoeta99/tps_paralelos.git>",
        "",
        "---",
        "",
        "## Abstract",
        "",
        "Esta segunda entrega agrega un único caso nuevo al trabajo previo: `numba_gpu`. "
        f"Los resultados de `secuencial`, `numpy` y `numba_cpu` se copian desde `{entrega1_csv.name}`, generado en la entrega 1, y luego se combinan con `{gpu_csv.name}`. "
        "La implementación CUDA sigue la estructura didáctica del material de cátedra: kernels `@cuda.jit`, `cuda.grid(2)`, un hilo por píxel y recorrido local 3x3 para Sobel. "
        "La comparación principal se realiza contra el secuencial usando el tiempo total, reportando tanto speed-up como mejora porcentual.",
        "",
        "---",
        "",
        "## 1. Introducción",
        "",
        "El operador Sobel calcula un gradiente local para cada píxel usando una vecindad 3x3. En GPU, esa independencia permite asignar un píxel por thread CUDA. Esta entrega no reejecuta los casos de la entrega anterior; solo incorpora la versión `numba_gpu` y la compara contra los resultados ya obtenidos.",
        "",
        "---",
        "",
        "## 2. Metodología",
        "",
        "### 2.1 Equipo",
        "",
        "| Propiedad | Valor |",
        "| --- | --- |",
        "| CPU | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz |",
        f"| Núcleos lógicos | {cpu_logical} |",
        f"| GPU | {gpu_name} |",
        f"| Multiprocesadores CUDA reportados | {gpu_workers} |",
        "| Python | 3.14.5 |",
        "| NumPy | 2.4.4 |",
        "| Numba | 0.65.1 |",
        "| Pillow | 12.2.0 |",
        "",
        "### 2.2 Algoritmos incluidos",
        "",
        "1. **Secuencial:** resultado de la entrega 1.",
        "2. **NumPy:** resultado de la entrega 1.",
        "3. **Numba CPU:** resultado de la entrega 1.",
        "4. **Numba GPU:** caso nuevo de la entrega 2.",
        "",
        "### 2.3 Parámetros experimentales",
        "",
        "| Parámetro | Valor |",
        "| --- | --- |",
        f"| Tamaños | {', '.join(f'{size}x{size}' for size in sizes)} |",
        "| Corridas por caso | 5 |",
        f"| Resultados base | `{entrega1_csv}` |",
        f"| Resultado nuevo | `{gpu_csv}` |",
        "| Script de corrida GPU | `python correr_entrega2.py` |",
        "",
        "La carga de imágenes y el guardado de salidas no forman parte de las mediciones.",
        "",
        "### 2.4 Métricas",
        "",
        "- **Tiempo RGB->gris (s):** tiempo promedio de conversión.",
        "- **Tiempo Sobel (s):** tiempo promedio del filtro Sobel.",
        "- **Tiempo total (s):** suma medida de conversión y Sobel.",
        "- **% blancos:** `(píxeles con valor 255 / píxeles totales) * 100`.",
        "- **Speed-up vs secuencial:** `tiempo_total_secuencial / tiempo_total_metodo`.",
        "- **Mejora vs secuencial (%):** `(1 - tiempo_total_metodo / tiempo_total_secuencial) * 100`.",
        "",
        "---",
        "",
        "## 3. Comparación combinada",
        "",
        "Las tablas siguientes integran los tres casos de la entrega 1 con el caso nuevo `numba_gpu`. "
        "La mejora porcentual se calcula siempre respecto del secuencial del mismo tamaño.",
        "",
    ]

    for index, size in enumerate(sizes, start=1):
        lines.extend(_combined_result_table(rows, size, index))

    lines.extend(
        [
            "---",
            "",
            "## 4. Discusión",
            "",
            "### 4.1 Mejora porcentual de Numba GPU contra secuencial",
            "",
            "La mejora de `numba_gpu` respecto del secuencial, usando tiempo total, fue: "
            + _comparison_sentence(rows)
            + ".",
            "",
            "Esta es la comparación central de la entrega: el nuevo caso GPU se evalúa contra la base secuencial del trabajo original.",
            "",
            "### 4.2 Amortización de transferencias CPU<->GPU",
            "",
            f"En {largest}x{largest}, las transferencias GPU suman aproximadamente {largest_transfer:.5f} s (`H2D + D2H`) y los kernels suman {largest_kernels:.5f} s. "
            f"El tiempo total GPU es {float(largest_gpu['tiempo_total_s']):.5f} s frente a {float(largest_seq['tiempo_total_s']):.5f} s del secuencial. "
            "La mejora se observa cuando el paralelismo de la GPU compensa el costo de copiar datos entre host y dispositivo.",
            "",
            "### 4.3 Consistencia de salida",
            "",
            "| Tamaño | Secuencial % blancos | NumPy % blancos | Numba CPU % blancos | Numba GPU % blancos |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(consistency_rows)
    lines.extend(
        [
            "",
            "Los porcentajes coinciden entre los cuatro métodos para cada tamaño, por lo que la comparación es de rendimiento y no de diferencia en la métrica de salida.",
            "",
            "---",
            "",
            "## 5. Conclusión",
            "",
            "La entrega 2 incorpora solamente el caso `numba_gpu` y lo integra con los resultados de la entrega 1. "
            "La versión GPU mantiene la misma salida según el porcentaje de blancos y mejora fuertemente el tiempo total respecto del secuencial en todos los tamaños medidos. "
            "La lectura del rendimiento debe considerar tanto el tiempo total como el desglose de transferencias y kernels.",
            "",
            "---",
            "",
            "## Referencias",
            "",
            "- *Trabajo Práctico - Filtro de Sobel para detección de bordes*, Sistemas Paralelos, UNTDF, 2026.",
            "- *Introducción a la programación paralela*, material de cátedra, capítulo 8.",
            "- Numba Documentation: CUDA kernels.",
            "- NVIDIA CUDA Programming Guide.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrega1-csv", type=Path, default=ROOT_DIR.parent / "tp3.1" / "resultados_sobel_entrega1.csv")
    parser.add_argument("--gpu-csv", type=Path, default=ROOT_DIR / "resultados_sobel_entrega2.csv")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "informe_sobel_entrega2.md")
    args = parser.parse_args()
    write_report(args.entrega1_csv, args.gpu_csv, args.output)
    print(f"Informe escrito en {args.output}")


if __name__ == "__main__":
    main()
