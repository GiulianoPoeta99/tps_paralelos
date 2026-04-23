from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from suma_lib import DEFAULT_C, DEFAULT_SEED, build_chunks, print_result, sum_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Suma de vector en paralelo con ThreadPoolExecutor")
    parser.add_argument("--c", type=int, default=DEFAULT_C, help="Tamano del vector")
    parser.add_argument("--workers", type=int, default=4, help="Cantidad de workers")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla para vector deterministico")
    args = parser.parse_args()

    chunks = build_chunks(args.c, args.workers)

    start = perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        partials = list(executor.map(lambda chunk: sum_chunk(chunk[0], chunk[1], args.seed), chunks))
    total = sum(partials)
    elapsed = perf_counter() - start

    print_result("concurrent.futures.ThreadPoolExecutor", args.c, args.workers, total, elapsed)


if __name__ == "__main__":
    main()
