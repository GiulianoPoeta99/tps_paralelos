"""Multiplicacion A*B con metodo transpuesto; complejidad c, aleatorio acotado, verificacion (AB)^T = B^T A^T."""

from __future__ import annotations

import os
import random
import sys
from argparse import ArgumentParser
from typing import Callable, Sequence

MatrixInt = list[list[int]]

DEFAULT_SEED = 2026
DEFAULT_COMPLEXITY = 64
DEFAULT_MAX_MAGNITUDE = 256


def gil_status_line() -> str:
    """Indica si el GIL esta activo (Python 3.13+)."""
    enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    return f"GIL habilitado: {enabled}"


def configure_matrix_parser(parser: ArgumentParser) -> None:
    """Argumentos comunes: complejidad, dimensiones, semilla y cota de aleatorio."""
    parser.add_argument(
        "--c",
        type=int,
        default=DEFAULT_COMPLEXITY,
        dest="complejidad",
        help="Complejidad: con --perfil define m,n,p (salvo --mnp)",
    )
    parser.add_argument(
        "--perfil",
        choices=("cubo", "mixto"),
        default="cubo",
        help="cubo: A CxC y B CxC; mixto: A Cx(2C) y B (2C)x(3C)",
    )
    parser.add_argument(
        "--mnp",
        type=str,
        default=None,
        help="Sobrescribe dimensiones: m,n,p con A m x n y B n x p (coma-separado)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla del generador aleatorio")
    parser.add_argument(
        "--max-val",
        type=int,
        default=DEFAULT_MAX_MAGNITUDE,
        dest="max_val",
        help="Cota absoluta de enteros aleatorios en cada celda (|valor| <= max-val)",
    )


def resolve_dimensions(complejidad: int, perfil: str, mnp: str | None) -> tuple[int, int, int]:
    """Devuelve (m, n, p): A es m x n, B es n x p."""
    if mnp is not None and str(mnp).strip():
        parts = [int(x.strip()) for x in str(mnp).split(",") if x.strip()]
        if len(parts) != 3:
            raise ValueError("--mnp debe ser tres enteros: m,n,p")
        m, n, p = parts
        if min(m, n, p) <= 0:
            raise ValueError("m, n y p deben ser positivos")
        return m, n, p

    if complejidad <= 0:
        raise ValueError("--c (complejidad) debe ser positiva")

    if perfil == "cubo":
        c = complejidad
        return c, c, c

    if perfil == "mixto":
        c = complejidad
        return c, 2 * c, 3 * c

    raise ValueError(f"perfil desconocido: {perfil}")


def random_int_bounded(rng: random.Random, max_magnitude: int) -> int:
    """Entero en [-max_magnitude, max_magnitude]."""
    if max_magnitude < 0:
        raise ValueError("max-val no puede ser negativo")
    if max_magnitude == 0:
        return 0
    return rng.randint(-max_magnitude, max_magnitude)


def build_matrix_random(rows: int, cols: int, seed: int, max_magnitude: int) -> MatrixInt:
    """Matriz pseudoaleatoria reproducible; |cada elemento| <= max_magnitude."""
    rng = random.Random(seed)
    return [[random_int_bounded(rng, max_magnitude) for _ in range(cols)] for _ in range(rows)]


def transpose(m: MatrixInt) -> MatrixInt:
    if not m:
        return []
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


def build_chunks(size: int, workers: int) -> list[tuple[int, int]]:
    if size < 0:
        raise ValueError("size no puede ser negativo")
    if workers <= 0:
        raise ValueError("workers debe ser mayor que 0")

    real_workers = min(workers, size) if size > 0 else 1
    base = size // real_workers
    remainder = size % real_workers

    chunks: list[tuple[int, int]] = []
    start = 0
    for worker in range(real_workers):
        extra = 1 if worker < remainder else 0
        end = start + base + extra
        chunks.append((start, end))
        start = end
    return chunks


def dot_chunk(
    start: int,
    end: int,
    row_a: Sequence[int],
    row_bt: Sequence[int],
) -> int:
    total = 0
    for k in range(start, end):
        total += row_a[k] * row_bt[k]
    return total


def dot_sequential(row_a: Sequence[int], row_bt: Sequence[int]) -> int:
    return dot_chunk(0, len(row_a), row_a, row_bt)


def matmul_traditional(a: MatrixInt, b: MatrixInt) -> MatrixInt:
    """
    C = A * B con triple bucle (acceso directo a B[k][j]).
    """
    if not a or not b:
        return []
    m = len(a)
    inner = len(a[0])
    if any(len(row) != inner for row in a):
        raise ValueError("A debe ser rectangular")
    if len(b) != inner:
        raise ValueError("A.columns debe coincidir con B.filas")
    p = len(b[0])
    if any(len(row) != p for row in b):
        raise ValueError("B debe ser rectangular")

    c: MatrixInt = []
    for i in range(m):
        row_c: list[int] = []
        for j in range(p):
            acc = 0
            for k in range(inner):
                acc += a[i][k] * b[k][j]
            row_c.append(acc)
        c.append(row_c)
    return c


def matmul_transpose_method(
    a: MatrixInt,
    b: MatrixInt,
    dot_fn: Callable[[Sequence[int], Sequence[int]], int],
) -> MatrixInt:
    """
    C = A * B usando filas de B^T: C[i][j] = A[i] · B^T[j].
    """
    if not a or not b:
        return []
    m = len(a)
    inner = len(a[0])
    if any(len(row) != inner for row in a):
        raise ValueError("A debe ser rectangular")
    if len(b) != inner:
        raise ValueError("A.columns debe coincidir con B.filas")
    p = len(b[0])
    if any(len(row) != p for row in b):
        raise ValueError("B debe ser rectangular")

    bt = transpose(b)
    c: MatrixInt = []
    for i in range(m):
        row_c: list[int] = []
        for j in range(p):
            row_c.append(dot_fn(a[i], bt[j]))
        c.append(row_c)
    return c


def matrix_trace_sum(c: MatrixInt) -> int:
    total = 0
    for row in c:
        for x in row:
            total += x
    return total


def print_run_header(
    complejidad: int,
    perfil: str,
    m: int,
    n: int,
    p: int,
    max_val: int,
    seed: int,
    mnp_override: str | None = None,
) -> None:
    if mnp_override is not None and str(mnp_override).strip():
        print(f"Dimensiones (--mnp): A {m}x{n}  B {n}x{p}  ->  C {m}x{p}")
        print(f"(La complejidad --c={complejidad} no fija m,n,p cuando se usa --mnp.)")
    else:
        print(f"Complejidad c: {complejidad}")
        print(f"Perfil: {perfil}  |  A: {m}x{n}  B: {n}x{p}  ->  C: {m}x{p}")
    print(f"Semilla: {seed}  |  Cota aleatorio (max-val): {max_val}")


def print_result_block(
    etiqueta: str,
    checksum: int,
    elapsed: float,
) -> None:
    print(f"  [{etiqueta}] tiempo (s): {elapsed:.6f}  |  suma elementos: {checksum}")
    if etiqueta == "A · B":
        print(f"BENCHMARK_AB_TIME: {elapsed:.6f}")
        print(f"BENCHMARK_AB_CHECKSUM: {checksum}")
    elif etiqueta.startswith("B^T"):
        print(f"BENCHMARK_BTAT_TIME: {elapsed:.6f}")
        print(f"BENCHMARK_BTAT_CHECKSUM: {checksum}")


def print_result_footer(workers: int) -> None:
    print(f"Workers: {workers}")
    print(f"Equipo (cores disponibles): {os.cpu_count()}")
    print(gil_status_line())


def print_implementation_title(title: str) -> None:
    print(f"Implementacion: {title}")
