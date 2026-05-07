"""Corre todos los casos de la entrega 1 del TP Sobel."""

from __future__ import annotations

import os

from benchmark import run_benchmark
from sobel_lib import DEFAULT_IMAGE_DIR, ROOT_DIR


SIZES = [750, 1500, 3000, 6000]
METHODS = ["secuencial", "numpy", "numba_cpu"]
RUNS = 5
WORKERS = os.cpu_count() or 1
SAVE_PREVIEW = True

OUTPUT_CSV = ROOT_DIR / "resultados_sobel_entrega1.csv"
OUTPUT_MD = ROOT_DIR / "resultados_sobel_entrega1.md"


def main() -> None:
    print("Configuracion entrega 1")
    print(f"Tamanios: {SIZES}")
    print(f"Metodos: {METHODS}")
    print(f"Corridas por caso: {RUNS}")
    print(f"Workers Numba CPU: {WORKERS}")
    print()

    run_benchmark(
        sizes=SIZES,
        methods=METHODS,
        runs=RUNS,
        image_dir=DEFAULT_IMAGE_DIR,
        workers=WORKERS,
        output=OUTPUT_CSV,
        md_output=OUTPUT_MD,
        save_preview=SAVE_PREVIEW,
    )


if __name__ == "__main__":
    main()
