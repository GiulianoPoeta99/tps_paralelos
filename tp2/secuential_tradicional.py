"""Multiplicacion C = A*B secuencial (triple bucle / tradicional); verificacion con B^T A^T = (AB)^T."""

from __future__ import annotations

import argparse
from time import perf_counter

from matrices_lib import (
    build_matrix_random,
    configure_matrix_parser,
    matmul_traditional,
    matrix_trace_sum,
    print_implementation_title,
    print_result_block,
    print_result_footer,
    print_run_header,
    resolve_dimensions,
    transpose,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matmul secuencial tradicional (triple bucle)")
    configure_matrix_parser(parser)
    parser.add_argument("--workers", type=int, default=1, help="Informativo (interfaz comun con variantes paralelas)")
    args = parser.parse_args()

    m, n, p = resolve_dimensions(args.complejidad, args.perfil, args.mnp)
    print("[Progreso] Generando matrices de entrada...", flush=True)
    a = build_matrix_random(m, n, args.seed, args.max_val)
    b = build_matrix_random(n, p, args.seed + 100_003, args.max_val)

    print_run_header(args.complejidad, args.perfil, m, n, p, args.max_val, args.seed, args.mnp)
    print_implementation_title("secuencial (tradicional)")

    print("[Progreso] Iniciando fase A·B...", flush=True)
    t0 = perf_counter()
    c_ab = matmul_traditional(a, b)
    t1 = perf_counter()

    bt = transpose(b)
    at = transpose(a)
    print("[Progreso] Iniciando fase B^T · A^T...", flush=True)
    t2 = perf_counter()
    c_bt_at = matmul_traditional(bt, at)
    t3 = perf_counter()

    if transpose(c_ab) != c_bt_at:
        raise SystemExit("Error: (AB)^T != B^T A^T")

    print_result_block("A · B", matrix_trace_sum(c_ab), t1 - t0)
    print_result_block("B^T · A^T  (equiv. a (A·B)^T)", matrix_trace_sum(c_bt_at), t3 - t2)
    print("Verificacion: (A·B)^T == B^T·A^T  ->  OK")
    print_result_footer(args.workers)


if __name__ == "__main__":
    main()
