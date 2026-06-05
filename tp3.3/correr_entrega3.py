"""Corre los casos nuevos de la entrega 3 del TP Sobel."""

from __future__ import annotations

import os

from benchmark_entrega3 import run_benchmark_entrega3
from generar_informe_entrega3 import write_report
from sobel_lib import DEFAULT_IMAGE_DIR, ROOT_DIR


SIZES = [750, 1500, 3000, 6000]
RUNS = 5
METHODS = ["pytorch_cpu", "pytorch_gpu"]
WORKERS = os.cpu_count() or 1
SAVE_PREVIEW = True

ENTREGA1_CSV = ROOT_DIR.parent / "tp3.1" / "resultados_sobel_entrega1.csv"
ENTREGA2_CSV = ROOT_DIR.parent / "tp3.2" / "resultados_sobel_entrega2.csv"
OUTPUT_CSV = ROOT_DIR / "resultados_sobel_entrega3.csv"
OUTPUT_MD = ROOT_DIR / "resultados_sobel_entrega3.md"
OUTPUT_REPORT = ROOT_DIR / "informe_sobel_entrega3.md"


def main() -> None:
    print("Configuración entrega 3")
    print(f"Tamaños: {SIZES}")
    print(f"Métodos nuevos: {METHODS}")
    print(f"Resultados base entrega 1: {ENTREGA1_CSV}")
    print(f"Resultados base entrega 2: {ENTREGA2_CSV}")
    print(f"Corridas por caso: {RUNS}")
    print(f"Workers PyTorch CPU: {WORKERS}")
    print()

    run_benchmark_entrega3(
        sizes=SIZES,
        runs=RUNS,
        methods=METHODS,
        image_dir=DEFAULT_IMAGE_DIR,
        output=OUTPUT_CSV,
        md_output=OUTPUT_MD,
        save_preview=SAVE_PREVIEW,
        workers=WORKERS,
    )
    write_report(ENTREGA1_CSV, ENTREGA2_CSV, OUTPUT_CSV, OUTPUT_REPORT)
    print(f"Informe escrito en {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
