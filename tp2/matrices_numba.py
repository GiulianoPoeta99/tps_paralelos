"""
Numba: multiplicacion A·B con enteros.
- Sin NumPy: arreglos planos como listas de int.
- --parallel: prange + numba.set_num_threads(workers); workers=1 -> serial nopython.
"""

from __future__ import annotations

import argparse
from time import perf_counter

from matrices_lib import (
    build_matrix_random,
    configure_matrix_parser,
    matrix_trace_sum,
    print_implementation_title,
    print_result_block,
    print_result_footer,
    print_run_header,
    resolve_dimensions,
    transpose,
)

try:
    from numba import njit, prange, set_num_threads
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Falta el paquete 'numba'. Instalá con: pip install numba\n"
        f"({exc})"
    ) from exc


def matrix_to_flat_ints(mat: list[list[int]], rows: int, cols: int) -> list[int]:
    out: list[int] = [0] * (rows * cols)
    t = 0
    for i in range(rows):
        for j in range(cols):
            out[t] = mat[i][j]
            t += 1
    return out


@njit(cache=True, fastmath=False)  # type: ignore[misc]
def matmul_kernel_serial(m: int, n: int, p: int, a: list, b: list, c: list) -> None:
    for i in range(m):
        for j in range(p):
            s = 0
            for k in range(n):
                s += a[i * n + k] * b[k * p + j]
            c[i * p + j] = s


@njit(cache=True, fastmath=False, parallel=True)  # type: ignore[misc]
def matmul_kernel_parallel(m: int, n: int, p: int, a: list, b: list, c: list) -> None:
    for i in prange(m):
        for j in range(p):
            s = 0
            for k in range(n):
                s += a[i * n + k] * b[k * p + j]
            c[i * p + j] = s


def matmul_int_flat(m: int, n: int, p: int, a: list, b: list, parallel: bool) -> list:
    c = [0] * (m * p)
    if parallel:
        matmul_kernel_parallel(m, n, p, a, b, c)
    else:
        matmul_kernel_serial(m, n, p, a, b, c)
    return c


def flat_to_int_matrix(flat: list, m: int, p: int) -> list[list[int]]:
    out: list[list[int]] = []
    t = 0
    for _i in range(m):
        row: list[int] = []
        for _j in range(p):
            row.append(flat[t])
            t += 1
        out.append(row)
    return out


def transpose_int(mat: list[list[int]]) -> list[list[int]]:
    if not mat:
        return []
    r = len(mat)
    c = len(mat[0])
    return [[mat[i][j] for i in range(r)] for j in range(c)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matmul A·B con numba njit (enteros, sin NumPy)",
    )
    configure_matrix_parser(parser)
    parser.add_argument("--workers", type=int, default=1, help="Hilos numba (parallel) o 1 (serial njit)")
    mx = parser.add_mutually_exclusive_group()
    mx.add_argument(
        "--parallel",
        action="store_true",
        help="Usar prange (parallel=True). Setea numba.set_num_threads(--workers) antes de la corrida.",
    )
    mx.add_argument(
        "--no-parallel",
        action="store_true",
        help="Solo nopython serial (parallel=False). Aplicable con --workers 1 (benchmark sugerido).",
    )
    args = parser.parse_args()

    m, n, p = resolve_dimensions(args.complejidad, args.perfil, args.mnp)
    print("[Progreso] Generando matrices de entrada...", flush=True)
    a_int = build_matrix_random(m, n, args.seed, args.max_val)
    b_int = build_matrix_random(n, p, args.seed + 100_003, args.max_val)

    a = matrix_to_flat_ints(a_int, m, n)
    b = matrix_to_flat_ints(b_int, n, p)
    a_bt_int = transpose(b_int)
    a_at_int = transpose(a_int)
    bt = matrix_to_flat_ints(a_bt_int, p, n)
    at_ = matrix_to_flat_ints(a_at_int, n, m)

    if args.workers < 1:
        raise SystemExit("--workers debe ser >= 1")
    if args.no_parallel or args.workers == 1:
        use_parallel = False
        w = 1
    else:
        use_parallel = bool(args.parallel) or (not args.no_parallel and args.workers > 1)
        w = int(args.workers)
        set_num_threads(w)

    print_run_header(args.complejidad, args.perfil, m, n, p, args.max_val, args.seed, args.mnp)
    label = f"numba njit, parallel={use_parallel}, threads={w}"
    print_implementation_title(label)

    print("[Progreso] Compilando kernels Numba (warm-up)...", flush=True)
    # Primera pasada: compila njit; no cuenta como tiempo de benchmark.
    matmul_int_flat(2, 2, 2, [1, 0, 0, 1], [1, 0, 0, 1], use_parallel)

    print("[Progreso] Iniciando fase A·B...", flush=True)
    t0 = perf_counter()
    c_ab_flat = matmul_int_flat(m, n, p, a, b, use_parallel)
    t1 = perf_counter()

    print("[Progreso] Iniciando fase B^T · A^T...", flush=True)
    t2 = perf_counter()
    c_btat_flat = matmul_int_flat(p, n, m, bt, at_, use_parallel)
    t3 = perf_counter()

    c_ab = flat_to_int_matrix(c_ab_flat, m, p)
    c_bt_at = flat_to_int_matrix(c_btat_flat, p, m)

    if transpose_int(c_ab) != c_bt_at:
        raise SystemExit("Error: (AB)^T != B^T A^T (verificación con enteros)")

    ch_ab = matrix_trace_sum(c_ab)
    ch_btat = matrix_trace_sum(c_bt_at)

    print_result_block("A · B", ch_ab, t1 - t0)
    print_result_block("B^T · A^T  (equiv. a (A·B)^T)", ch_btat, t3 - t2)
    print("Verificacion: (A·B)^T == B^T·A^T  ->  OK")
    print_result_footer(w)


if __name__ == "__main__":
    main()
