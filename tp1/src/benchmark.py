from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TIME_RE = re.compile(r"Tiempo \(segundos\): ([0-9]*\.?[0-9]+)")
SUM_RE = re.compile(r"Suma final: ([0-9]*\.?[0-9]+)")
CSV_FIELDS = [
    "algoritmo",
    "complejidad_c",
    "procesos_workers",
    "tiempo_segundos",
    "speed_up",
    "eficiencia",
    "equipo_cores",
    "estado",
    "error",
    "suma_final",
]


@dataclass
class RunResult:
    algorithm: str
    c_value: int
    workers: int
    elapsed: float | None
    total_sum: float | None
    status: str
    error: str


METHODS: dict[str, tuple[str, str]] = {
    "secuencial": ("secuential.py", "secuencial"),
    "threadpoolexecutor": ("suma_threadpoolexecutor.py", "concurrent.futures.ThreadPoolExecutor"),
    "processpoolexecutor": ("suma_processpoolexecutor.py", "concurrent.futures.ProcessPoolExecutor"),
    "multiprocessing": ("suma_multiprocessing.py", "multiprocessing.Pool"),
    "threading": ("suma_threading.py", "threading"),
}


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Debes indicar al menos un valor")
    return parsed


def parse_metrics(output: str) -> tuple[float | None, float | None]:
    time_match = TIME_RE.search(output)
    sum_match = SUM_RE.search(output)

    elapsed = float(time_match.group(1)) if time_match else None
    total_sum = float(sum_match.group(1)) if sum_match else None
    return elapsed, total_sum


def result_key(algorithm: str, c_value: int, workers: int) -> tuple[str, int, int]:
    return (algorithm, c_value, workers)


def to_float_or_none(value: str) -> float | None:
    raw = (value or "").strip()
    if raw == "":
        return None
    return float(raw)


def load_existing_results(output_path: Path) -> list[RunResult]:
    if not output_path.exists():
        return []

    results: list[RunResult] = []
    with output_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                algorithm = (row.get("algoritmo") or "").strip()
                c_value = int((row.get("complejidad_c") or "").strip())
                workers = int((row.get("procesos_workers") or "").strip())
                elapsed = to_float_or_none(row.get("tiempo_segundos", ""))
                total_sum = to_float_or_none(row.get("suma_final", ""))
                status = (row.get("estado") or "error").strip() or "error"
                error = (row.get("error") or "").strip()
            except Exception:
                # Ignora filas corruptas para no cortar ejecuciones largas.
                continue

            results.append(
                RunResult(
                    algorithm=algorithm,
                    c_value=c_value,
                    workers=workers,
                    elapsed=elapsed,
                    total_sum=total_sum,
                    status=status,
                    error=error,
                )
            )
    return results


def run_one(
    python_exec: str,
    script_path: Path,
    algorithm_name: str,
    c_value: int,
    workers: int,
    seed: int,
) -> RunResult:
    command = [
        python_exec,
        str(script_path),
        "--c",
        str(c_value),
        "--workers",
        str(workers),
        "--seed",
        str(seed),
    ]

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception as exc:
        return RunResult(
            algorithm=algorithm_name,
            c_value=c_value,
            workers=workers,
            elapsed=None,
            total_sum=None,
            status="error",
            error=f"Fallo al ejecutar comando: {exc}",
        )

    full_output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    elapsed, total_sum = parse_metrics(full_output)

    if completed.returncode != 0:
        return RunResult(
            algorithm=algorithm_name,
            c_value=c_value,
            workers=workers,
            elapsed=elapsed,
            total_sum=total_sum,
            status="error",
            error=f"Exit code {completed.returncode}: {(completed.stderr or '').strip()}",
        )

    if elapsed is None or total_sum is None:
        return RunResult(
            algorithm=algorithm_name,
            c_value=c_value,
            workers=workers,
            elapsed=elapsed,
            total_sum=total_sum,
            status="error",
            error="No se pudieron parsear tiempo/suma del output",
        )

    return RunResult(
        algorithm=algorithm_name,
        c_value=c_value,
        workers=workers,
        elapsed=elapsed,
        total_sum=total_sum,
        status="ok",
        error="",
    )


def compute_speedup_and_efficiency(results: Iterable[RunResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    baseline_by_c: dict[int, float] = {}
    for result in results:
        if result.algorithm == "secuencial" and result.status == "ok" and result.elapsed is not None:
            baseline_by_c[result.c_value] = result.elapsed

    cores = os.cpu_count() or 0

    for result in results:
        baseline = baseline_by_c.get(result.c_value)
        speedup: float | None = None
        efficiency: float | None = None

        if result.status == "ok" and baseline is not None and result.elapsed is not None and result.elapsed > 0:
            speedup = baseline / result.elapsed
            if result.workers > 0:
                efficiency = speedup / result.workers

        rows.append(
            {
                "algoritmo": result.algorithm,
                "complejidad_c": str(result.c_value),
                "procesos_workers": str(result.workers),
                "tiempo_segundos": f"{result.elapsed:.6f}" if result.elapsed is not None else "",
                "speed_up": f"{speedup:.6f}" if speedup is not None else "",
                "eficiencia": f"{efficiency:.6f}" if efficiency is not None else "",
                "equipo_cores": str(cores),
                "estado": result.status,
                "error": result.error,
                "suma_final": f"{result.total_sum:.6f}" if result.total_sum is not None else "",
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta benchmark de sumaParalela y exporta CSV")
    parser.add_argument("--p-values", type=parse_int_list, default=[1, 2, 4, 8, 16], help="Lista separada por comas")
    parser.add_argument("--c-values", type=parse_int_list, default=[12, 1_000_000, 100_000_000], help="Lista separada por comas")
    parser.add_argument("--seed", type=int, default=2026, help="Semilla para vector deterministico")
    parser.add_argument("--output", type=str, default="resultados_suma.csv", help="Archivo CSV de salida")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe el CSV e ignora resultados previos")
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Vuelve a correr combinaciones que ya existen en el CSV",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="secuencial,threadpoolexecutor,processpoolexecutor,multiprocessing,threading",
        help="Metodos separados por comas",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Ruta al ejecutable de Python")
    args = parser.parse_args()

    selected_methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    for method in selected_methods:
        if method not in METHODS:
            valid = ", ".join(METHODS.keys())
            raise ValueError(f"Metodo invalido: {method}. Metodos validos: {valid}")

    base_dir = Path(__file__).resolve().parent
    output_path = (base_dir / args.output).resolve()
    existing_results = [] if args.overwrite else load_existing_results(output_path)
    existing_keys = {
        result_key(item.algorithm, item.c_value, item.workers)
        for item in existing_results
    }

    new_results: list[RunResult] = []
    skipped = 0

    for c_value in args.c_values:
        sec_script, sec_name = METHODS["secuencial"]
        sec_key = result_key(sec_name, c_value, 1)
        if args.rerun_existing or sec_key not in existing_keys:
            sec_result = run_one(
                python_exec=args.python,
                script_path=base_dir / sec_script,
                algorithm_name=sec_name,
                c_value=c_value,
                workers=1,
                seed=args.seed,
            )
            new_results.append(sec_result)
        else:
            skipped += 1

        for workers in args.p_values:
            for method in selected_methods:
                if method == "secuencial":
                    continue
                script_name, algorithm_name = METHODS[method]
                row_key = result_key(algorithm_name, c_value, workers)
                if args.rerun_existing or row_key not in existing_keys:
                    new_results.append(
                        run_one(
                            python_exec=args.python,
                            script_path=base_dir / script_name,
                            algorithm_name=algorithm_name,
                            c_value=c_value,
                            workers=workers,
                            seed=args.seed,
                        )
                    )
                else:
                    skipped += 1

    merged_by_key: dict[tuple[str, int, int], RunResult] = {}
    for item in existing_results:
        merged_by_key[result_key(item.algorithm, item.c_value, item.workers)] = item
    for item in new_results:
        merged_by_key[result_key(item.algorithm, item.c_value, item.workers)] = item

    all_results = list(merged_by_key.values())
    output_rows = compute_speedup_and_efficiency(all_results)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    total = len(output_rows)
    ok = sum(1 for row in output_rows if row["estado"] == "ok")
    errors = total - ok
    print(f"CSV generado: {output_path}")
    print(f"Nuevas filas ejecutadas: {len(new_results)} | Saltadas por existentes: {skipped}")
    print(f"Filas: {total} | OK: {ok} | Error: {errors}")


if __name__ == "__main__":
    main()
