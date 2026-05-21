"""Corre todos los casos de la entrega 2 del TP Sobel."""

from __future__ import annotations

from benchmark_entrega2 import run_benchmark_entrega2
from generar_informe_entrega2 import write_report
from sobel_lib import DEFAULT_IMAGE_DIR, ROOT_DIR


SIZES = [750, 1500, 3000, 6000]
RUNS = 5
SAVE_PREVIEW = True

ENTREGA1_CSV = ROOT_DIR.parent / "tp3.1" / "resultados_sobel_entrega1.csv"
OUTPUT_CSV = ROOT_DIR / "resultados_sobel_entrega2.csv"
OUTPUT_MD = ROOT_DIR / "resultados_sobel_entrega2.md"
OUTPUT_REPORT = ROOT_DIR / "informe_sobel_entrega2.md"


def main() -> None:
    print("Configuración entrega 2")
    print(f"Tamaños: {SIZES}")
    print("Métodos nuevos: ['numba_gpu']")
    print(f"Resultados base entrega 1: {ENTREGA1_CSV}")
    print(f"Corridas por caso: {RUNS}")
    print()

    run_benchmark_entrega2(
        sizes=SIZES,
        runs=RUNS,
        image_dir=DEFAULT_IMAGE_DIR,
        output=OUTPUT_CSV,
        md_output=OUTPUT_MD,
        save_preview=SAVE_PREVIEW,
    )
    write_report(ENTREGA1_CSV, OUTPUT_CSV, OUTPUT_REPORT)
    print(f"Informe escrito en {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
