"""
Multiplicacion en paralelo por filas (ProcessPoolExecutor);
misma verificacion (AB)^T = B^T A^T. Los procesos evitan el GIL en carga CPU-bound.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

from matrices_lib import (
    MatrixInt,
    build_chunks,
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

_A_GLOBAL: MatrixInt = []
_BT_GLOBAL: MatrixInt = []


def _init_worker(a: MatrixInt, bt: MatrixInt) -> None:
    global _A_GLOBAL, _BT_GLOBAL
    _A_GLOBAL = a
    _BT_GLOBAL = bt


def _row_block(interval: tuple[int, int]) -> tuple[int, int, MatrixInt]:
    """Top-level: filas [i0, i1) de C = A·B con B ya transpuesta a bt."""
    i0, i1 = interval
    p = len(_BT_GLOBAL)
    block: MatrixInt = []
    for i in range(i0, i1):
        row = [dot_sequential(_A_GLOBAL[i], _BT_GLOBAL[j]) for j in range(p)]
        block.append(row)
    return (i0, i1, block)


def matmul_rows_process_pool(a: MatrixInt, b: MatrixInt, max_workers: int, *, show_progress: bool = True) -> MatrixInt:
    m = len(a)
    p = len(b[0])
    bt = transpose(b)
    if not m or not p:
        return []
    w = max_workers if max_workers > 0 else 1
    intervals = build_chunks(m, w)
    total_parts = len(intervals)
    if show_progress:
        print(f"[Progreso] ProcessPool: {total_parts} bloques de filas con {w} procesos.", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=w, initializer=_init_worker, initargs=(a, bt)) as executor:
        futures = [executor.submit(_row_block, interval) for interval in intervals]
        done = 0
        for fut in as_completed(futures):
            parts.append(fut.result())
            done += 1
            if show_progress:
                pct = (done * 100) // total_parts
                print(f"[Progreso] ProcessPool: {done}/{total_parts} bloques ({pct}%).", flush=True)
    parts.sort(key=lambda t: t[0])
    c: MatrixInt = []
    for _i0, _i1, block in parts:
        c.extend(block)
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description="Matmul paralelo por filas (ProcessPoolExecutor)")
    configure_matrix_parser(parser)
    parser.add_argument("--workers", type=int, default=4, help="Numero de procesos (pool)")
    args = parser.parse_args()

    m, n, p = resolve_dimensions(args.complejidad, args.perfil, args.mnp)
    a = build_matrix_random(m, n, args.seed, args.max_val)
    b = build_matrix_random(n, p, args.seed + 100_003, args.max_val)

    print_run_header(args.complejidad, args.perfil, m, n, p, args.max_val, args.seed, args.mnp)
    print_implementation_title("concurrent.futures.ProcessPoolExecutor (paralelo por filas)")

    print("[Progreso] Iniciando fase A·B...", flush=True)
    t0 = perf_counter()
    c_ab = matmul_rows_process_pool(a, b, args.workers, show_progress=True)
    t1 = perf_counter()

    bt = transpose(b)
    at = transpose(a)
    print("[Progreso] Iniciando fase B^T · A^T...", flush=True)
    t2 = perf_counter()
    c_bt_at = matmul_rows_process_pool(bt, at, args.workers, show_progress=True)
    t3 = perf_counter()

    if transpose(c_ab) != c_bt_at:
        raise SystemExit("Error: (AB)^T != B^T A^T")

    print_result_block("A · B", matrix_trace_sum(c_ab), t1 - t0)
    print_result_block("B^T · A^T  (equiv. a (A·B)^T)", matrix_trace_sum(c_bt_at), t3 - t2)
    print("Verificacion: (A·B)^T == B^T·A^T  ->  OK")
    print_result_footer(args.workers)


if __name__ == "__main__":
    main()
