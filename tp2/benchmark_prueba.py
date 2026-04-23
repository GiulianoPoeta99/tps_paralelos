#!/usr/bin/env python3
"""
Una ejecucion = un valor de c + toda la grilla de workers (como sumaParalela).

Sin --method: secuencial (workers 1) y, para cada valor en --workers-values (por defecto 1,4,10),
cada metodo paralelo (ThreadPoolExecutor y threading) con esa cantidad de hilos. Mismo criterio que
`benchmark.py` / `benchmark.py --c` pero con salida en vivo. Alias: --p-values (ver ayuda).

Con --method y --workers: una sola corrida puntual (depuracion).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from benchmark import (
    CSV_FIELDS,
    DEFAULT_P_VALUES,
    METHODS,
    RunResult,
    SEQUENTIAL_METHOD_KEYS,
    compute_speedup_and_efficiency,
    extra_args_for_method,
    load_existing_results,
    parse_int_list,
    result_key,
    run_one,
)
from generar_informe_md import informe_md_path_for_csv, write_informe_md


def print_resumen(result: RunResult) -> None:
    print("\n=== Resumen parseado ===", flush=True)
    print(f"Algoritmo: {result.algorithm}", flush=True)
    print(f"c={result.c_value}  workers={result.workers}  perfil={result.perfil}  max-val={result.max_val}", flush=True)
    print(f"Estado: {result.status}", flush=True)
    if result.error:
        print(f"Error: {result.error}", flush=True)
    if result.elapsed_ab is not None:
        print(f"Tiempo A·B (s): {result.elapsed_ab:.6f}", flush=True)
    if result.elapsed_btat is not None:
        print(f"Tiempo B^T·A^T (s): {result.elapsed_btat:.6f}", flush=True)
    if result.checksum_ab is not None:
        print(f"Checksum A·B: {result.checksum_ab}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Por cada ejecucion indicas solo --c: corre secuencial + cada cantidad de workers "
            "en --workers-values para cada metodo paralelo (igual criterio que sumaParalela). Salida en vivo."
        ),
    )
    parser.add_argument("--c", type=int, required=True, help="Complejidad c (unica de esta corrida)")
    parser.add_argument(
        "--workers-values",
        "--p-values",
        type=parse_int_list,
        default=DEFAULT_P_VALUES,
        dest="workers_values",
        metavar="LIST",
        help=(
            f"Lista de workers/hilos para paralelos (defecto {','.join(map(str, DEFAULT_P_VALUES))}). "
            "Alias --p-values (no es la dimension de columnas de B)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Solo con --method: cantidad de hilos de esa corrida",
    )
    parser.add_argument(
        "--method",
        choices=sorted(METHODS.keys()),
        default=None,
        help="Si no se indica: se ejecutan secuencial + todos los paralelos con --workers-values",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--perfil", choices=("cubo", "mixto"), default="cubo")
    parser.add_argument("--max-val", type=int, default=256, dest="max_val")
    parser.add_argument(
        "--output",
        type=str,
        default="resultados_matrices_v2.csv",
        help="CSV (merge con el existente). El informe MD sera resultados/<mismo_nombre>.md",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Solo imprimir; no leer ni escribir CSV",
    )
    parser.add_argument(
        "--no-informe",
        action="store_true",
        help="No regenerar resultados/resultados_matrices.md",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Ejecutable de Python")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    runs: list[tuple[str, str, int, int, list[str]]] = []
    if args.method is not None:
        if args.workers is None:
            parser.error("Con --method hace falta --workers (una corrida puntual).")
        runs.append(
            (METHODS[args.method][0], METHODS[args.method][1], args.c, args.workers, extra_args_for_method(args.method, args.workers or 1)),
        )
    else:
        for sk in SEQUENTIAL_METHOD_KEYS:
            s_script, s_name = METHODS[sk]
            runs.append((s_script, s_name, args.c, 1, extra_args_for_method(sk, 1)))
        for worker_count in args.workers_values:
            for key in ("threadpoolexecutor", "threading", "multiprocessing", "numba"):
                script_name, algorithm_name = METHODS[key]
                runs.append(
                    (script_name, algorithm_name, args.c, worker_count, extra_args_for_method(key, worker_count)),
                )

    print(
        f"=== Prueba c={args.c}  perfil={args.perfil}  max-val={args.max_val}  "
        f"({len(runs)} corrida(s)) ===\n",
        flush=True,
    )

    results: list[RunResult] = []
    any_error = False
    for script_name, algorithm_name, c_value, workers, extra in runs:
        print(f"\n>>> {algorithm_name} | c={c_value} | workers={workers}\n", flush=True)
        result = run_one(
            python_exec=args.python,
            script_path=base_dir / script_name,
            algorithm_name=algorithm_name,
            c_value=c_value,
            workers=workers,
            seed=args.seed,
            perfil=args.perfil,
            max_val=args.max_val,
            extra_args=extra,
            stream_output=True,
        )
        print_resumen(result)
        results.append(result)
        if result.status != "ok":
            any_error = True

    if args.no_csv:
        sys.exit(1 if any_error else 0)

    output_path = (base_dir / args.output).resolve()
    existing = load_existing_results(output_path)
    merged: dict[tuple[str, int, int, str, int], RunResult] = {
        result_key(r.algorithm, r.c_value, r.workers, r.perfil, r.max_val): r for r in existing
    }
    for result in results:
        merged[result_key(result.algorithm, result.c_value, result.workers, result.perfil, result.max_val)] = result

    rows = compute_speedup_and_efficiency(merged.values())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV actualizado: {output_path}  ({len(results)} fila(s) tocadas en esta sesion)", flush=True)

    if not args.no_informe:
        try:
            md_path = informe_md_path_for_csv(output_path, base_dir)
            write_informe_md(output_path, md_path)
            print(f"Informe Markdown: {md_path}", flush=True)
        except Exception as exc:
            print(f"Aviso: no se pudo generar el informe MD: {exc}", file=sys.stderr)

    sys.exit(1 if any_error else 0)


if __name__ == "__main__":
    main()
