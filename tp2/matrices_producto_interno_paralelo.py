"""
Patron suma paralela sobre k (un producto interno). Vectores aleatorios acotados.

La longitud del vector se controla con --c (complejidad del vector, analogo a sumaParalela).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from matrices_lib import (
    DEFAULT_MAX_MAGNITUDE,
    DEFAULT_SEED,
    build_chunks,
    build_matrix_random,
    dot_chunk,
    dot_sequential,
)

DEFAULT_VECTOR_C = 500_000


def dot_parallel_chunks(row_a: list[int], row_bt: list[int], max_workers: int) -> int:
    n = len(row_a)
    chunks = build_chunks(n, max_workers)
    workers = max_workers if max_workers > 0 else 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        partials = list(executor.map(lambda ch: dot_chunk(ch[0], ch[1], row_a, row_bt), chunks))
    return sum(partials, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Un producto interno con suma paralela (trozos de k)")
    parser.add_argument(
        "--c",
        type=int,
        default=DEFAULT_VECTOR_C,
        dest="complejidad_vector",
        help="Longitud n de los vectores (complejidad del producto interno)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Workers para trozos")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla")
    parser.add_argument(
        "--max-val",
        type=int,
        default=DEFAULT_MAX_MAGNITUDE,
        dest="max_val",
        help="Cota absoluta de cada componente aleatoria",
    )
    args = parser.parse_args()

    if args.complejidad_vector <= 0:
        raise SystemExit("--c debe ser positivo")

    row_a = build_matrix_random(1, args.complejidad_vector, args.seed, args.max_val)[0]
    row_bt = build_matrix_random(1, args.complejidad_vector, args.seed + 1, args.max_val)[0]

    print(f"Complejidad vector (n): {args.complejidad_vector}  |  max-val: {args.max_val}  |  seed: {args.seed}")

    start = perf_counter()
    total_seq = dot_sequential(row_a, row_bt)
    elapsed_seq = perf_counter() - start

    start = perf_counter()
    total_par = dot_parallel_chunks(row_a, row_bt, args.workers)
    elapsed_par = perf_counter() - start

    if total_seq != total_par:
        raise SystemExit("Resultado distinto entre secuencial y paralelo")

    print("--- Secuencial (un producto interno) ---")
    print(f"Tiempo (segundos): {elapsed_seq:.6f}")
    print(f"Resultado: {total_seq}")
    print()
    print("--- Paralelo (trozos de k, ThreadPoolExecutor) ---")
    print(f"Workers: {args.workers}")
    print(f"Tiempo (segundos): {elapsed_par:.6f}")
    print(f"Resultado: {total_par}")
    if elapsed_par > 0 and elapsed_seq > 0:
        print(f"Speed-up aproximado (seq/par): {elapsed_seq / elapsed_par:.4f}")


if __name__ == "__main__":
    main()
