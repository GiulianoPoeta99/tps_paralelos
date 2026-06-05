"""Genera el informe final de Sobel integrando Secuencial, NumPy, Numba y PyTorch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sobel_lib import ROOT_DIR


METHOD_ORDER = ["secuencial", "numpy", "numba_cpu", "numba_gpu", "pytorch_cpu", "pytorch_gpu"]
PREVIOUS_METHODS = ["secuencial", "numpy", "numba_cpu", "numba_gpu"]
METHOD_LABELS = {
    "secuencial": "Secuencial",
    "numpy": "NumPy",
    "numba_cpu": "Numba CPU",
    "numba_gpu": "Numba GPU",
    "pytorch_cpu": "PyTorch CPU",
    "pytorch_gpu": "PyTorch GPU",
}


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


def _row_source(
    entrega1_rows: list[dict[str, str]],
    entrega2_rows: list[dict[str, str]],
    pytorch_rows: list[dict[str, str]],
    method: str,
) -> list[dict[str, str]]:
    if method in ("secuencial", "numpy", "numba_cpu"):
        return entrega1_rows
    if method == "numba_gpu":
        return entrega2_rows
    return pytorch_rows


def _combined_rows(
    entrega1_rows: list[dict[str, str]],
    entrega2_rows: list[dict[str, str]],
    pytorch_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows = []
    sizes = sorted({int(row["tamano"]) for row in entrega1_rows})
    for size in sizes:
        seq = _method_row(entrega1_rows, size, "secuencial")
        seq_total = _float(seq, "tiempo_total_s")
        for method in METHOD_ORDER:
            source_rows = _row_source(entrega1_rows, entrega2_rows, pytorch_rows, method)
            row = dict(_method_row(source_rows, size, method))
            total = _float(row, "tiempo_total_s")
            speed_up = seq_total / total if total > 0 else 0.0
            improvement_pct = (1.0 - (total / seq_total)) * 100.0 if seq_total > 0 else 0.0
            row["speed_up_vs_secuencial"] = f"{speed_up:.6f}"
            row["mejora_vs_secuencial_pct"] = f"{improvement_pct:.2f}"
            row["method_label"] = METHOD_LABELS[method]
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
                    row["method_label"],
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


def _gpu_vs_cpu_table(rows: list[dict[str, str]], sizes: list[int]) -> list[str]:
    lines = [
        "| Tamaño | PyTorch CPU total (s) | PyTorch GPU total con transferencias (s) | Speed-up GPU vs CPU |",
        "| --- | ---: | ---: | ---: |",
    ]
    for size in sizes:
        cpu = _method_row(rows, size, "pytorch_cpu")
        gpu = _method_row(rows, size, "pytorch_gpu")
        cpu_total = _float(cpu, "tiempo_total_s")
        gpu_total = _float(gpu, "tiempo_total_s")
        speed_up = cpu_total / gpu_total if gpu_total > 0 else 0.0
        lines.append(f"| {size}x{size} | {cpu_total:.9f} | {gpu_total:.9f} | {speed_up:.6f} |")
    lines.append("")
    return lines


def _best_previous(rows: list[dict[str, str]], size: int) -> dict[str, str]:
    candidates = [_method_row(rows, size, method) for method in PREVIOUS_METHODS]
    return min(candidates, key=lambda row: _float(row, "tiempo_total_s"))


def _previous_comparison_table(rows: list[dict[str, str]], sizes: list[int]) -> list[str]:
    lines = [
        "| Tamaño | Mejor método Numba/NumPy | Mejor tiempo Numba/NumPy (s) | PyTorch CPU / mejor Numba-NumPy | PyTorch GPU / mejor Numba-NumPy |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for size in sizes:
        best = _best_previous(rows, size)
        best_total = _float(best, "tiempo_total_s")
        cpu_total = _float(_method_row(rows, size, "pytorch_cpu"), "tiempo_total_s")
        gpu_total = _float(_method_row(rows, size, "pytorch_gpu"), "tiempo_total_s")
        cpu_ratio = best_total / cpu_total if cpu_total > 0 else 0.0
        gpu_ratio = best_total / gpu_total if gpu_total > 0 else 0.0
        lines.append(
            f"| {size}x{size} | {best['method_label']} | {best_total:.9f} | {cpu_ratio:.6f} | {gpu_ratio:.6f} |"
        )
    lines.append("")
    return lines


def _comparison_sentence(rows: list[dict[str, str]], method: str) -> str:
    parts = []
    for row in rows:
        if row["metodo"] != method:
            continue
        size = int(row["tamano"])
        improvement = _float(row, "mejora_vs_secuencial_pct")
        speed_up = _float(row, "speed_up_vs_secuencial")
        parts.append(f"{improvement:.2f}% en {size}x{size} ({speed_up:.2f}x)")
    return ", ".join(parts)


def write_report(entrega1_csv: Path, entrega2_csv: Path, pytorch_csv: Path, output_path: Path) -> None:
    entrega1_rows = _read_csv(entrega1_csv)
    entrega2_rows = _read_csv(entrega2_csv)
    pytorch_rows = _read_csv(pytorch_csv)
    rows = _combined_rows(entrega1_rows, entrega2_rows, pytorch_rows)
    sizes = sorted({int(row["tamano"]) for row in entrega1_rows})

    first_seq = _method_row(entrega1_rows, sizes[0], "secuencial")
    first_pytorch_cpu = _method_row(pytorch_rows, sizes[0], "pytorch_cpu")
    first_pytorch_gpu = _method_row(pytorch_rows, sizes[0], "pytorch_gpu")
    cpu_logical = first_seq.get("cpu_logicos", "")
    gpu_name = first_pytorch_gpu.get("gpu", "GPU CUDA")
    torch_version = first_pytorch_cpu.get("torch_version", "")
    torch_cuda = first_pytorch_gpu.get("torch_cuda", "")
    cpu_physical = "4"
    threads_per_core = "2"

    largest = sizes[-1]
    largest_gpu = _method_row(rows, largest, "pytorch_gpu")
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
                    _method_row(rows, size, "pytorch_cpu")["porcentaje_blancos"],
                    _method_row(rows, size, "pytorch_gpu")["porcentaje_blancos"],
                ]
            )
            + " |"
        )

    lines: list[str] = [
        "# Filtro de Sobel en CPU y GPU: Secuencial, NumPy, Numba y PyTorch",
        "",
        "**Materia:** Sistemas Paralelos - Lic. en Sistemas, 5to año  ",
        "**Institución:** UNTDF  ",
        "**Docente:** MsC. Federico González Brizzio  ",
        "**Entrega:** 28 de mayo de 2026  ",
        "**Repositorio:** <https://github.com/GiulianoPoeta99/tps_paralelos.git>",
        "",
        "---",
        "",
        "## Abstract",
        "",
        "Este informe integra la evolución completa del trabajo: una versión secuencial, una versión vectorizada con NumPy, una versión paralela en CPU con Numba, una versión GPU con Numba CUDA y dos variantes con PyTorch sobre CPU y GPU. "
        "Todas las implementaciones aplican el mismo flujo de procesamiento: conversión RGB->gris por luminancia y filtro Sobel 3x3 usando la magnitud del gradiente. "
        "La comparación se organiza por tamaño de imagen y utiliza el tiempo total promedio como referencia para calcular speed-up y mejora porcentual.",
        "",
        "---",
        "",
        "## 1. Introducción",
        "",
        "El operador Sobel calcula un gradiente local para cada píxel usando una vecindad 3x3. El objetivo del trabajo es analizar cómo cambia el rendimiento cuando el mismo algoritmo se expresa con distintos enfoques: Python secuencial, operaciones vectorizadas, compilación/paralelismo en CPU, kernels CUDA y tensores PyTorch sobre CPU/GPU.",
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
        f"| Núcleos físicos | {cpu_physical} |",
        f"| Procesadores lógicos (hilos) | {cpu_logical} |",
        f"| Threads por núcleo | {threads_per_core} |",
        f"| GPU | {gpu_name} |",
        "| Python | 3.14.5 |",
        "| NumPy | 2.4.4 |",
        "| Numba | 0.65.1 |",
        f"| PyTorch | {torch_version} |",
        f"| CUDA reportado por PyTorch | {torch_cuda} |",
        "| Pillow | 12.2.0 |",
        "",
        "### 2.2 Algoritmos incluidos",
        "",
        "1. **Secuencial:** implementación base sin librerías vectorizadas.",
        "2. **NumPy:** implementación vectorizada sobre arreglos.",
        "3. **Numba CPU:** implementación compilada y paralela sobre CPU.",
        "4. **Numba GPU:** implementación con kernels CUDA.",
        "5. **PyTorch CPU:** implementación con tensores sobre CPU.",
        "6. **PyTorch GPU:** implementación con tensores sobre GPU.",
        "",
        "### 2.3 Parámetros experimentales",
        "",
        "| Parámetro | Valor |",
        "| --- | --- |",
        f"| Tamaños | {', '.join(f'{size}x{size}' for size in sizes)} |",
        "| Corridas por caso | 5 |",
        f"| Fuente Secuencial/NumPy/Numba CPU | `{entrega1_csv}` |",
        f"| Fuente Numba GPU | `{entrega2_csv}` |",
        f"| Fuente PyTorch CPU/GPU | `{pytorch_csv}` |",
        "| Script de corrida PyTorch | `python correr_entrega3.py` |",
        "",
        "La carga de imágenes y el guardado de salidas no forman parte de las mediciones.",
        "",
        "### 2.4 Métricas",
        "",
        "- **Tiempo RGB->gris (s):** tiempo promedio de conversión.",
        "- **Tiempo Sobel (s):** tiempo promedio del filtro Sobel.",
        "- **Tiempo total (s):** suma medida de conversión y Sobel. En PyTorch GPU incluye transferencias CPU<->GPU.",
        "- **% blancos:** `(píxeles con valor 255 / píxeles totales) * 100`.",
        "- **Speed-up vs secuencial:** `tiempo_total_secuencial / tiempo_total_metodo`.",
        "- **Mejora vs secuencial (%):** `(1 - tiempo_total_metodo / tiempo_total_secuencial) * 100`.",
        "",
        "---",
        "",
        "## 3. Comparación combinada",
        "",
        "Las tablas siguientes integran todas las implementaciones medidas. La mejora porcentual se calcula siempre respecto del secuencial del mismo tamaño.",
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
            "### 4.1 PyTorch CPU vs PyTorch GPU",
            "",
            "La siguiente tabla compara directamente PyTorch CPU contra PyTorch GPU. El total de PyTorch GPU incluye transferencia CPU->GPU, cómputo y transferencia GPU->CPU.",
            "",
        ]
    )
    lines.extend(_gpu_vs_cpu_table(rows, sizes))
    lines.extend(
        [
            "### 4.2 PyTorch contra las mejores implementaciones Numba/NumPy",
            "",
            "Un valor mayor a `1` indica que PyTorch fue más rápido que la mejor implementación Numba/NumPy para ese tamaño. Un valor menor a `1` indica que Numba/NumPy siguió siendo más rápido.",
            "",
        ]
    )
    lines.extend(_previous_comparison_table(rows, sizes))
    lines.extend(
        [
            "### 4.3 Mejora porcentual de PyTorch contra secuencial",
            "",
            "La mejora de `pytorch_cpu` respecto del secuencial fue: " + _comparison_sentence(rows, "pytorch_cpu") + ".",
            "",
            "La mejora de `pytorch_gpu` respecto del secuencial fue: " + _comparison_sentence(rows, "pytorch_gpu") + ".",
            "",
            "### 4.4 Transferencias en PyTorch GPU",
            "",
            f"En {largest}x{largest}, las transferencias PyTorch GPU suman aproximadamente {largest_transfer:.5f} s (`H2D + D2H`) y los kernels suman {largest_kernels:.5f} s. "
            "Este desglose permite ver cuánto del tiempo total corresponde al movimiento de datos y cuánto al cómputo sobre GPU.",
            "",
            "### 4.5 Consistencia de salida",
            "",
            "| Tamaño | Secuencial % blancos | NumPy % blancos | Numba CPU % blancos | Numba GPU % blancos | PyTorch CPU % blancos | PyTorch GPU % blancos |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(consistency_rows)
    lines.extend(
        [
            "",
            "Los porcentajes de blancos permiten controlar que la comparación sea de rendimiento y no de una diferencia en la salida del algoritmo.",
            "",
            "### 4.6 Síntesis del análisis",
            "",
            "**1. ¿Qué diferencias de rendimiento se observan entre PyTorch CPU y PyTorch GPU para cada tamaño de imagen?**",
            "",
            "PyTorch GPU fue más rápido que PyTorch CPU en todos los tamaños medidos. "
            "La diferencia es moderada en 750x750 y crece en tamaños mayores, especialmente en 3000x3000 y 6000x6000. "
            "Esto muestra que el uso de GPU se justifica mejor cuando hay más trabajo por corrida, aunque el tiempo total sigue incluyendo las transferencias entre CPU y GPU.",
            "",
            "**2. ¿Cómo se comparan PyTorch CPU/PyTorch GPU frente a las mejores implementaciones Numba/NumPy?**",
            "",
            "PyTorch CPU no supera a las mejores implementaciones Numba/NumPy en ningún tamaño. "
            "PyTorch GPU mejora claramente a PyTorch CPU, pero tampoco supera al mejor resultado global. "
            "Aun así, PyTorch GPU queda cerca de Numba GPU en algunos tamaños y mejora ampliamente a NumPy y PyTorch CPU en los tamaños grandes.",
            "",
            "**3. ¿Las salidas de PyTorch (CPU/GPU) son consistentes con Numba/NumPy en términos de bordes detectados y porcentaje de píxeles blancos, y qué factores podrían explicar diferencias?**",
            "",
            "Sí. En los cuatro tamaños, PyTorch CPU y PyTorch GPU dan el mismo porcentaje de píxeles blancos que Secuencial, NumPy, Numba CPU y Numba GPU. "
            "Esto indica que las salidas son consistentes para la métrica usada. "
            "Si aparecieran diferencias, podrían deberse a cambios en el redondeo al convertir de `float` a `uint8`, diferencias en el manejo de bordes, uso de otra fórmula de magnitud del gradiente, signos distintos en las máscaras Sobel, o diferencias de precisión entre operaciones CPU y GPU.",
            "",
            "---",
            "",
            "## 5. Conclusión",
            "",
            "El informe final integra las seis implementaciones desarrolladas para el filtro Sobel. "
            "Todas mantienen la misma salida según el porcentaje de blancos, por lo que la comparación se centra en rendimiento. "
            "PyTorch GPU supera a PyTorch CPU en todos los tamaños, especialmente en 3000x3000 y 6000x6000, pero las mejores variantes globales siguen siendo las implementaciones basadas en Numba CPU/GPU según el tamaño de imagen.",
            "",
            "---",
            "",
            "## Referencias",
            "",
            "- *Trabajo Práctico - Filtro de Sobel para detección de bordes*, Sistemas Paralelos, UNTDF, 2026.",
            "- *Introducción a la programación paralela*, material de cátedra, capítulos 7 y 8.",
            "- PyTorch Documentation: Tensors and `torch.nn.functional.conv2d`.",
            "- Numba Documentation: CUDA kernels.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrega1-csv", type=Path, default=ROOT_DIR.parent / "tp3.1" / "resultados_sobel_entrega1.csv")
    parser.add_argument("--entrega2-csv", type=Path, default=ROOT_DIR.parent / "tp3.2" / "resultados_sobel_entrega2.csv")
    parser.add_argument("--pytorch-csv", type=Path, default=ROOT_DIR / "resultados_sobel_entrega3.csv")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "informe_sobel_entrega3.md")
    args = parser.parse_args()
    write_report(args.entrega1_csv, args.entrega2_csv, args.pytorch_csv, args.output)
    print(f"Informe escrito en {args.output}")


if __name__ == "__main__":
    main()
