from __future__ import annotations

import argparse
from time import perf_counter

from suma_lib import DEFAULT_C, DEFAULT_SEED, print_result, sum_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Suma de vector en forma secuencial")
    parser.add_argument("--c", type=int, default=DEFAULT_C, help="Tamano del vector")
    parser.add_argument("--workers", type=int, default=1, help="Parametro informativo para mantener interfaz comun")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla para vector deterministico")
    args = parser.parse_args()

    start = perf_counter()
    total = sum_chunk(0, args.c, args.seed)
    elapsed = perf_counter() - start

    print_result("secuencial", args.c, args.workers, total, elapsed)


if __name__ == "__main__":
    main()
