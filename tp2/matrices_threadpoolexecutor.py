"""Multiplicacion en paralelo por filas (ThreadPoolExecutor); verificacion traspuesta."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from matrices_lib import (
    build_matrix_random,
    configure_matrix_parser,
    dot_sequential,
    matrix_trace_sum,
    print_implementation_title,
    print_result_block,
    print_result_footer,
    print_run_header,
    resolve_dimensions,
    transpose,
)


def matmul_rows_parallel(a: list[list[int]], b: list[list[int]], max_workers: int, *, progress: bool = True) -> list[list[int]]:
    if not a or not b:
        return []
    m = len(a)
    bt = transpose(b)
    workers = max_workers if max_workers > 0 else 1
    progress_step = max(1, m // 20)

    def compute_row(i: int) -> tuple[int, list[int]]:
        return i, [dot_sequential(a[i], bt[j]) for j in range(len(bt))]

    if progress:
        print(f"[Progreso] ThreadPool: {m} filas a calcular con {workers} workers.", flush=True)
    out: list[list[int] | None] = [None] * m
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for done, result in enumerate(executor.map(compute_row, range(m)), start=1):
            i, row = result
            out[i] = row
            if progress and (done % progress_step == 0 or done == m):
                pct = (done * 100) // m
                print(f"[Progreso] ThreadPool: {done}/{m} filas ({pct}%).", flush=True)

    return [row if row is not None else [] for row in out]


def main() -> None:
    parser = argparse.ArgumentParser(description="Matmul paralelo por filas (ThreadPoolExecutor)")
    configure_matrix_parser(parser)
    parser.add_argument("--workers", type=int, default=4, help="Tamano del pool")
    args = parser.parse_args()

    m, n, p = resolve_dimensions(args.complejidad, args.perfil, args.mnp)
    a = build_matrix_random(m, n, args.seed, args.max_val)
    b = build_matrix_random(n, p, args.seed + 100_003, args.max_val)

    print_run_header(args.complejidad, args.perfil, m, n, p, args.max_val, args.seed, args.mnp)
    print_implementation_title("concurrent.futures.ThreadPoolExecutor (paralelo por filas)")

    print("[Progreso] Iniciando fase A·B...", flush=True)
    t0 = perf_counter()
    c_ab = matmul_rows_parallel(a, b, args.workers, progress=True)
    t1 = perf_counter()

    bt = transpose(b)
    at = transpose(a)
    print("[Progreso] Iniciando fase B^T · A^T...", flush=True)
    t2 = perf_counter()
    c_bt_at = matmul_rows_parallel(bt, at, args.workers, progress=True)
    t3 = perf_counter()

    if transpose(c_ab) != c_bt_at:
        raise SystemExit("Error: (AB)^T != B^T A^T")

    print_result_block("A · B", matrix_trace_sum(c_ab), t1 - t0)
    print_result_block("B^T · A^T  (equiv. a (A·B)^T)", matrix_trace_sum(c_bt_at), t3 - t2)
    print("Verificacion: (A·B)^T == B^T·A^T  ->  OK")
    print_result_footer(args.workers)


if __name__ == "__main__":
    main()
