"""Funciones compartidas para el ejercicio de suma paralela de un vector."""

from __future__ import annotations

import os
from typing import Sequence


DEFAULT_SEED = 2026
DEFAULT_C = 1_000_000


def vector_value(index: int, seed: int = DEFAULT_SEED) -> float:
    """Genera un valor deterministico para la posicion index."""
    state = (1103515245 * (index + seed) + 12345) & 0x7FFFFFFF
    return state / 0x7FFFFFFF


def sum_chunk(start: int, end: int, seed: int = DEFAULT_SEED) -> float:
    """Suma secuencialmente el segmento [start, end)."""
    total = 0.0
    for index in range(start, end):
        total += vector_value(index, seed)
    return total


def build_chunks(size: int, workers: int) -> list[tuple[int, int]]:
    """Divide un rango [0, size) en bloques casi uniformes."""
    if size < 0:
        raise ValueError("c no puede ser negativo")
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


def print_result(title: str, c_value: int, workers: int, total: float, elapsed: float) -> None:
    """Imprime un resumen comun para comparar variantes."""
    print(f"Implementacion: {title}")
    print(f"Complejidad c: {c_value}")
    print(f"Workers: {workers}")
    print(f"Tiempo (segundos): {elapsed:.6f}")
    print(f"Suma final: {total:.6f}")
    print(f"Equipo (cores disponibles): {os.cpu_count()}")


def parse_c_values(c_values_arg: str | None, fallback: Sequence[int] | None = None) -> list[int]:
    """Parsea una lista de c separada por comas."""
    if c_values_arg is None:
        if fallback is None:
            return [DEFAULT_C]
        return list(fallback)

    parsed = [int(item.strip()) for item in c_values_arg.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Debes indicar al menos un valor de c")
    return parsed
