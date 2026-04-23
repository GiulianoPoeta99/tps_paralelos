"""Multiplicacion en paralelo por filas; mismo caso con verificacion B^T A^T."""

from __future__ import annotations

import argparse
import threading
from queue import Queue
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


def worker(
    task_queue: Queue,
    results: list[list[int] | None],
    a: list[list[int]],
    bt: list[list[int]],
    total_rows: int,
    progress_step: int,
    progress_state: dict[str, int],
    progress_lock: threading.Lock,
    show_progress: bool,
) -> None:
    while True:
        row_idx = task_queue.get()
        if row_idx is None:
            task_queue.task_done()
            break

        results[row_idx] = [dot_sequential(a[row_idx], bt[j]) for j in range(len(bt))]
        if show_progress:
            with progress_lock:
                progress_state["done"] += 1
                done = progress_state["done"]
                if done >= progress_state["next"] or done == total_rows:
                    pct = (done * 100) // total_rows
                    print(f"[Progreso] threading: {done}/{total_rows} filas ({pct}%).", flush=True)
                    progress_state["next"] = done + progress_step
        task_queue.task_done()


def matmul_rows_parallel(a: list[list[int]], b: list[list[int]], num_workers: int, *, show_progress: bool = True) -> list[list[int]]:
    if not a or not b:
        return []
    m = len(a)
    bt = transpose(b)
    results: list[list[int] | None] = [None] * m
    progress_step = max(1, m // 20)
    progress_state = {"done": 0, "next": progress_step}
    progress_lock = threading.Lock()

    task_queue: Queue = Queue()
    threads: list[threading.Thread] = []
    workers = num_workers if num_workers > 0 else 1
    if show_progress:
        print(f"[Progreso] threading: {m} filas a calcular con {workers} workers.", flush=True)
    for _ in range(workers):
        thread = threading.Thread(
            target=worker,
            args=(task_queue, results, a, bt, m, progress_step, progress_state, progress_lock, show_progress),
        )
        thread.start()
        threads.append(thread)

    for i in range(m):
        task_queue.put(i)

    for _ in threads:
        task_queue.put(None)

    task_queue.join()
    for thread in threads:
        thread.join()

    return [row if row is not None else [] for row in results]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matmul paralelo por filas (threading + cola)",
    )
    configure_matrix_parser(parser)
    parser.add_argument("--workers", type=int, default=4, help="Hilos worker")
    args = parser.parse_args()

    m, n, p = resolve_dimensions(args.complejidad, args.perfil, args.mnp)
    a = build_matrix_random(m, n, args.seed, args.max_val)
    b = build_matrix_random(n, p, args.seed + 100_003, args.max_val)

    print_run_header(args.complejidad, args.perfil, m, n, p, args.max_val, args.seed, args.mnp)
    print_implementation_title("threading (paralelo por filas)")

    print("[Progreso] Iniciando fase A·B...", flush=True)
    t0 = perf_counter()
    c_ab = matmul_rows_parallel(a, b, args.workers, show_progress=True)
    t1 = perf_counter()

    bt = transpose(b)
    at = transpose(a)
    print("[Progreso] Iniciando fase B^T · A^T...", flush=True)
    t2 = perf_counter()
    c_bt_at = matmul_rows_parallel(bt, at, args.workers, show_progress=True)
    t3 = perf_counter()

    if transpose(c_ab) != c_bt_at:
        raise SystemExit("Error: (AB)^T != B^T A^T")

    print_result_block("A · B", matrix_trace_sum(c_ab), t1 - t0)
    print_result_block("B^T · A^T  (equiv. a (A·B)^T)", matrix_trace_sum(c_bt_at), t3 - t2)
    print("Verificacion: (A·B)^T == B^T·A^T  ->  OK")
    print_result_footer(args.workers)


if __name__ == "__main__":
    main()
