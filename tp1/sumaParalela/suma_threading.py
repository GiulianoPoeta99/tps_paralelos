from __future__ import annotations

import argparse
import threading
from queue import Queue
from time import perf_counter

from suma_lib import DEFAULT_C, DEFAULT_SEED, build_chunks, print_result, sum_chunk


def worker(task_queue: Queue, results: list[float], seed: int) -> None:
    while True:
        item = task_queue.get()
        if item is None:
            task_queue.task_done()
            break

        index, start, end = item
        results[index] = sum_chunk(start, end, seed)
        task_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Suma de vector en paralelo con threading")
    parser.add_argument("--c", type=int, default=DEFAULT_C, help="Tamano del vector")
    parser.add_argument("--workers", type=int, default=4, help="Cantidad de hilos")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla para vector deterministico")
    args = parser.parse_args()

    chunks = build_chunks(args.c, args.workers)
    results: list[float] = [0.0] * len(chunks)

    task_queue: Queue = Queue()
    threads: list[threading.Thread] = []
    for _ in range(args.workers):
        thread = threading.Thread(target=worker, args=(task_queue, results, args.seed))
        thread.start()
        threads.append(thread)

    start = perf_counter()
    for index, (chunk_start, chunk_end) in enumerate(chunks):
        task_queue.put((index, chunk_start, chunk_end))

    for _ in threads:
        task_queue.put(None)

    task_queue.join()
    for thread in threads:
        thread.join()

    total = sum(results)
    elapsed = perf_counter() - start

    print_result("threading", args.c, args.workers, total, elapsed)


if __name__ == "__main__":
    main()
