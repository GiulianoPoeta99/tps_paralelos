"""Ejecuta corridas de multiplicacion de matrices y exporta CSV (estilo sumaParalela/benchmark.py)."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TIME_AB_RE = re.compile(r"^BENCHMARK_AB_TIME:\s+([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
CHECK_AB_RE = re.compile(r"^BENCHMARK_AB_CHECKSUM:\s+(\S+)\s*$", re.MULTILINE)
TIME_BTAT_RE = re.compile(r"^BENCHMARK_BTAT_TIME:\s+([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
CHECK_BTAT_RE = re.compile(r"^BENCHMARK_BTAT_CHECKSUM:\s+(\S+)\s*$", re.MULTILINE)

# Consigna / docente: no mezclar cantidades de workers en una sola tanda lógica; grilla 1, 4, 10.
# (Se puede pasar --workers-values para otra grilla; por defecto 1,4,10 como en el práctico.)
DEFAULT_P_VALUES = [1, 4, 10]
DEFAULT_C_VALUES = [16, 32, 64, 128]

CSV_FIELDS = [
    "algoritmo",
    "complejidad_c",
    "perfil",
    "max_val",
    "procesos_workers",
    "tiempo_segundos_ab",
    "tiempo_segundos_btat",
    "speed_up_ab",
    "eficiencia_ab",
    "speed_up_btat",
    "eficiencia_btat",
    "equipo_cores",
    "estado",
    "error",
    "checksum_ab",
    "checksum_btat",
]


@dataclass
class RunResult:
    algorithm: str
    c_value: int
    workers: int
    perfil: str
    max_val: int
    elapsed_ab: float | None
    elapsed_btat: float | None
    checksum_ab: str | None
    checksum_btat: str | None
    status: str
    error: str


METHODS: dict[str, tuple[str, str]] = {
    "secuencial": ("secuential.py", "secuencial"),
    "secuencial_tradicional": ("secuential_tradicional.py", "secuencial (tradicional)"),
    "threadpoolexecutor": ("matrices_threadpoolexecutor.py", "concurrent.futures.ThreadPoolExecutor"),
    "threading": ("matrices_threading.py", "threading"),
    "multiprocessing": ("matrices_multiprocessing.py", "concurrent.futures.ProcessPoolExecutor"),
    "numba": ("matrices_numba.py", "numba (njit)"),
}

# Secuenciales: solo --workers 1; no se repiten en el bucle de --workers-values.
SEQUENTIAL_METHOD_KEYS = ("secuencial", "secuencial_tradicional")


def extra_args_for_method(method: str, workers: int) -> list[str]:
    """P.ej. numba: serial con 1 hilo o parallel con 4/10 según --workers (consigna)."""
    if method == "numba":
        if workers <= 1:
            return ["--no-parallel"]
        return ["--parallel"]
    return []


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Debes indicar al menos un valor")
    return parsed


def parse_matrix_metrics(output: str) -> tuple[float | None, float | None, str | None, str | None]:
    t_ab = TIME_AB_RE.search(output)
    c_ab = CHECK_AB_RE.search(output)
    t_bt = TIME_BTAT_RE.search(output)
    c_bt = CHECK_BTAT_RE.search(output)

    elapsed_ab = float(t_ab.group(1)) if t_ab else None
    elapsed_btat = float(t_bt.group(1)) if t_bt else None
    sum_ab = c_ab.group(1) if c_ab else None
    sum_btat = c_bt.group(1) if c_bt else None
    return elapsed_ab, elapsed_btat, sum_ab, sum_btat


def result_key(
    algorithm: str,
    c_value: int,
    workers: int,
    perfil: str,
    max_val: int,
) -> tuple[str, int, int, str, int]:
    return (algorithm, c_value, workers, perfil, max_val)


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
                perfil = (row.get("perfil") or "cubo").strip() or "cubo"
                max_val = int((row.get("max_val") or "256").strip())
                elapsed_ab = to_float_or_none(row.get("tiempo_segundos_ab", ""))
                elapsed_btat = to_float_or_none(row.get("tiempo_segundos_btat", ""))
                checksum_ab = (row.get("checksum_ab") or "").strip() or None
                checksum_btat = (row.get("checksum_btat") or "").strip() or None
                status = (row.get("estado") or "error").strip() or "error"
                error = (row.get("error") or "").strip()
            except Exception:
                continue

            results.append(
                RunResult(
                    algorithm=algorithm,
                    c_value=c_value,
                    workers=workers,
                    perfil=perfil,
                    max_val=max_val,
                    elapsed_ab=elapsed_ab,
                    elapsed_btat=elapsed_btat,
                    checksum_ab=checksum_ab,
                    checksum_btat=checksum_btat,
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
    perfil: str,
    max_val: int,
    *,
    extra_args: list[str] | None = None,
    stream_output: bool = False,
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
        "--perfil",
        perfil,
        "--max-val",
        str(max_val),
    ]
    if extra_args:
        command.extend(extra_args)

    try:
        if stream_output:
            print("Comando:", " ".join(shlex.quote(part) for part in command))
            print(flush=True)
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            line_chunks: list[str] = []
            if proc.stdout is not None:
                for line in proc.stdout:
                    line_chunks.append(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
            returncode = proc.wait()
            full_output = "".join(line_chunks)
            completed_like_returncode = returncode
        else:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            full_output = (completed.stdout or "") + "\n" + (completed.stderr or "")
            completed_like_returncode = completed.returncode
    except Exception as exc:
        return RunResult(
            algorithm=algorithm_name,
            c_value=c_value,
            workers=workers,
            perfil=perfil,
            max_val=max_val,
            elapsed_ab=None,
            elapsed_btat=None,
            checksum_ab=None,
            checksum_btat=None,
            status="error",
            error=f"Fallo al ejecutar comando: {exc}",
        )

    elapsed_ab, elapsed_btat, sum_ab, sum_btat = parse_matrix_metrics(full_output)

    if completed_like_returncode != 0:
        return RunResult(
            algorithm=algorithm_name,
            c_value=c_value,
            workers=workers,
            perfil=perfil,
            max_val=max_val,
            elapsed_ab=elapsed_ab,
            elapsed_btat=elapsed_btat,
            checksum_ab=sum_ab,
            checksum_btat=sum_btat,
            status="error",
            error=f"Exit code {completed_like_returncode}: (ver salida arriba)",
        )

    if elapsed_ab is None or sum_ab is None:
        return RunResult(
            algorithm=algorithm_name,
            c_value=c_value,
            workers=workers,
            perfil=perfil,
            max_val=max_val,
            elapsed_ab=elapsed_ab,
            elapsed_btat=elapsed_btat,
            checksum_ab=sum_ab,
            checksum_btat=sum_btat,
            status="error",
            error="No se pudieron parsear BENCHMARK_AB_* del output",
        )

    if elapsed_btat is None or sum_btat is None:
        return RunResult(
            algorithm=algorithm_name,
            c_value=c_value,
            workers=workers,
            perfil=perfil,
            max_val=max_val,
            elapsed_ab=elapsed_ab,
            elapsed_btat=elapsed_btat,
            checksum_ab=sum_ab,
            checksum_btat=sum_btat,
            status="error",
            error="No se pudieron parsear BENCHMARK_BTAT_* del output",
        )

    return RunResult(
        algorithm=algorithm_name,
        c_value=c_value,
        workers=workers,
        perfil=perfil,
        max_val=max_val,
        elapsed_ab=elapsed_ab,
        elapsed_btat=elapsed_btat,
        checksum_ab=sum_ab,
        checksum_btat=sum_btat,
        status="ok",
        error="",
    )


def compute_speedup_and_efficiency(results: Iterable[RunResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def group_key(r: RunResult) -> tuple[int, str, int]:
        return (r.c_value, r.perfil, r.max_val)

    # Baseline: preferir "secuencial (tradicional)" si hay fila; si no, "secuencial" (método transpuesto).
    baseline_ab: dict[tuple[int, str, int], float] = {}
    baseline_btat: dict[tuple[int, str, int], float] = {}
    gkeys = {group_key(r) for r in results if r.status == "ok"}
    for g in gkeys:
        cand_trad: RunResult | None = None
        cand_trans: RunResult | None = None
        for result in results:
            if result.status != "ok" or group_key(result) != g:
                continue
            if result.algorithm == "secuencial (tradicional)":
                cand_trad = result
            elif result.algorithm == "secuencial":
                cand_trans = result
        pick = cand_trad or cand_trans
        if pick is not None and pick.elapsed_ab is not None:
            baseline_ab[g] = pick.elapsed_ab
        if pick is not None and pick.elapsed_btat is not None:
            baseline_btat[g] = pick.elapsed_btat

    cores = os.cpu_count() or 0

    for result in results:
        g = (result.c_value, result.perfil, result.max_val)
        b_ab = baseline_ab.get(g)
        b_bt = baseline_btat.get(g)
        speedup_ab: float | None = None
        eff_ab: float | None = None
        speedup_btat: float | None = None
        eff_btat: float | None = None

        if (
            result.status == "ok"
            and b_ab is not None
            and result.elapsed_ab is not None
            and result.elapsed_ab > 0
        ):
            speedup_ab = b_ab / result.elapsed_ab
            if result.workers > 0:
                eff_ab = speedup_ab / result.workers

        if (
            result.status == "ok"
            and b_bt is not None
            and result.elapsed_btat is not None
            and result.elapsed_btat > 0
        ):
            speedup_btat = b_bt / result.elapsed_btat
            if result.workers > 0:
                eff_btat = speedup_btat / result.workers

        rows.append(
            {
                "algoritmo": result.algorithm,
                "complejidad_c": str(result.c_value),
                "perfil": result.perfil,
                "max_val": str(result.max_val),
                "procesos_workers": str(result.workers),
                "tiempo_segundos_ab": f"{result.elapsed_ab:.6f}" if result.elapsed_ab is not None else "",
                "tiempo_segundos_btat": f"{result.elapsed_btat:.6f}" if result.elapsed_btat is not None else "",
                "speed_up_ab": f"{speedup_ab:.6f}" if speedup_ab is not None else "",
                "eficiencia_ab": f"{eff_ab:.6f}" if eff_ab is not None else "",
                "speed_up_btat": f"{speedup_btat:.6f}" if speedup_btat is not None else "",
                "eficiencia_btat": f"{eff_btat:.6f}" if eff_btat is not None else "",
                "equipo_cores": str(cores),
                "estado": result.status,
                "error": result.error,
                "checksum_ab": result.checksum_ab or "",
                "checksum_btat": result.checksum_btat or "",
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark de matrices -> CSV. Por cada c: corridas secuenciales (workers 1) y luego "
            "cada metodo paralelo con cada valor de --workers-values (1,4,10 por defecto: "
            "no mezclar en la misma corrida otras grillas; ejecutar tandas separadas si hace falta). "
            "Usa --c para una sola complejidad. Numba: metodo 'numba' (ver matrices_numba.py, hilos vía --workers)."
        ),
    )
    parser.add_argument(
        "--c",
        type=int,
        default=None,
        help="Un solo valor de complejidad: ejecuta toda la grilla de workers/metodos solo para este c (ignora --c-values)",
    )
    parser.add_argument(
        "--workers-values",
        "--p-values",
        type=parse_int_list,
        default=DEFAULT_P_VALUES,
        dest="workers_values",
        metavar="LIST",
        help=(
            "Workers/hilos o procesos para metodos paralelos (separados por comas). Defecto: 1,4,10. "
            "Alias --p-values. Numba parallel usa set_num_threads(workers) cuando --parallel (ver matrices_numba.py)."
        ),
    )
    parser.add_argument(
        "--c-values",
        type=parse_int_list,
        default=DEFAULT_C_VALUES,
        help="Lista de complejidades c (separados por comas). No se usa si se pasa --c",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Semilla para matrices aleatorias")
    parser.add_argument(
        "--perfil",
        choices=("cubo", "mixto"),
        default="cubo",
        help="Perfil de dimensiones (igual que los scripts)",
    )
    parser.add_argument("--max-val", type=int, default=256, dest="max_val", help="Cota de aleatorio")
    parser.add_argument(
        "--output",
        type=str,
        default="resultados_matrices_v2.csv",
        help="CSV de salida; el informe MD sera resultados/<mismo_nombre>.md",
    )
    parser.add_argument("--overwrite", action="store_true", help="Ignora CSV previo")
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Vuelve a correr combinaciones ya presentes en el CSV",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="secuencial,secuencial_tradicional,threadpoolexecutor,threading,multiprocessing,numba",
        help="Metodos (keys de METHODS) separados por comas. Sin numpy/pytorch (consigna ajustada en clase).",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Ejecutable de Python")
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Muestra la salida en vivo de cada script ejecutado (recomendado para corridas largas).",
    )
    parser.add_argument(
        "--no-informe",
        action="store_true",
        help="No regenerar resultados/resultados_matrices.md",
    )
    args = parser.parse_args()

    if args.c is not None:
        c_values_list = [args.c]
    else:
        c_values_list = list(args.c_values)

    selected_methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    for method in selected_methods:
        if method not in METHODS:
            valid = ", ".join(METHODS.keys())
            raise ValueError(f"Metodo invalido: {method}. Validos: {valid}")

    base_dir = Path(__file__).resolve().parent
    output_path = (base_dir / args.output).resolve()
    existing_results = [] if args.overwrite else load_existing_results(output_path)
    existing_keys = {
        result_key(item.algorithm, item.c_value, item.workers, item.perfil, item.max_val)
        for item in existing_results
    }

    new_results: list[RunResult] = []
    skipped = 0

    for c_value in c_values_list:
        for seq_key in SEQUENTIAL_METHOD_KEYS:
            if seq_key not in selected_methods:
                continue
            sec_script, sec_name = METHODS[seq_key]
            sec_key = result_key(sec_name, c_value, 1, args.perfil, args.max_val)
            if args.rerun_existing or sec_key not in existing_keys:
                print(
                    f"[Progreso] Ejecutando {sec_name} | c={c_value} | workers=1 | perfil={args.perfil}",
                    flush=True,
                )
                result = run_one(
                    python_exec=args.python,
                    script_path=base_dir / sec_script,
                    algorithm_name=sec_name,
                    c_value=c_value,
                    workers=1,
                    seed=args.seed,
                    perfil=args.perfil,
                    max_val=args.max_val,
                    extra_args=extra_args_for_method(seq_key, 1),
                    stream_output=args.stream_output,
                )
                new_results.append(result)
                print(f"[Progreso] Finalizado {sec_name}: estado={result.status}", flush=True)
            else:
                skipped += 1

        for workers in args.workers_values:
            for method in selected_methods:
                if method in SEQUENTIAL_METHOD_KEYS:
                    continue
                script_name, algorithm_name = METHODS[method]
                row_key = result_key(algorithm_name, c_value, workers, args.perfil, args.max_val)
                if args.rerun_existing or row_key not in existing_keys:
                    print(
                        f"[Progreso] Ejecutando {algorithm_name} | c={c_value} | workers={workers} | perfil={args.perfil}",
                        flush=True,
                    )
                    result = run_one(
                        python_exec=args.python,
                        script_path=base_dir / script_name,
                        algorithm_name=algorithm_name,
                        c_value=c_value,
                        workers=workers,
                        seed=args.seed,
                        perfil=args.perfil,
                        max_val=args.max_val,
                        extra_args=extra_args_for_method(method, workers),
                        stream_output=args.stream_output,
                    )
                    new_results.append(result)
                    print(
                        f"[Progreso] Finalizado {algorithm_name} (workers={workers}): estado={result.status}",
                        flush=True,
                    )
                else:
                    skipped += 1

    merged_by_key: dict[tuple[str, int, int, str, int], RunResult] = {}
    for item in existing_results:
        merged_by_key[result_key(item.algorithm, item.c_value, item.workers, item.perfil, item.max_val)] = item
    for item in new_results:
        merged_by_key[result_key(item.algorithm, item.c_value, item.workers, item.perfil, item.max_val)] = item

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

    if not args.no_informe:
        try:
            from generar_informe_md import informe_md_path_for_csv, write_informe_md

            md_path = informe_md_path_for_csv(output_path, base_dir)
            write_informe_md(output_path.resolve(), md_path)
            print(f"Informe Markdown: {md_path}")
        except Exception as exc:
            print(f"Aviso: no se pudo generar el informe MD: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
