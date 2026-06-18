"""Ejecuta las mediciones sepia en orden y guarda resultados acumulados."""

from __future__ import annotations

import argparse

from procesamiento_video import add_common_arguments, config_from_args, print_divider
from pytorch_cpu import run as run_pytorch_cpu
from pytorch_gpu import run as run_pytorch_gpu
from secuencial import run as run_secuencial
from sepia_filter import METHOD_LABELS


RUNNERS = {
    "secuencial": run_secuencial,
    "pytorch_cpu": run_pytorch_cpu,
    "pytorch_gpu": run_pytorch_gpu,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta mediciones de filtro sepia: secuencial, PyTorch CPU y PyTorch GPU.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(RUNNERS),
        default=["secuencial", "pytorch_cpu", "pytorch_gpu"],
        help="Metodos a ejecutar en orden.",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Omite PyTorch GPU aunque figure en --methods.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    methods = list(args.methods)
    if args.skip_gpu:
        methods = [method for method in methods if method != "pytorch_gpu"]

    print_divider("Configuracion de medicion")
    print(f"Entrada: {config.input_path}")
    print(f"Salida: {config.output_dir}")
    print(f"Corridas por metodo: {config.runs}")
    print(f"Workers CPU: {config.workers}")
    print(f"Segundos a procesar: {config.seconds}")
    print(f"Max frames: {config.max_frames}")
    print(f"Codec salida: {config.codec}")
    print(f"Merge audio: {config.merge_audio}")
    print(f"Metodos: {', '.join(METHOD_LABELS[method] for method in methods)}")

    rows = []
    for method in methods:
        print_divider(f"Medicion {METHOD_LABELS[method]}")
        row = RUNNERS[method](config)
        rows.append(row)
        if row.status == "ok":
            print(
                f"[{METHOD_LABELS[method]}] OK: "
                f"{row.frames} frames, total pipeline {row.pipeline_total_s:.3f} s",
                flush=True,
            )
        else:
            print(f"[{METHOD_LABELS[method]}] ERROR registrado: {row.error}", flush=True)

    print_divider("Resumen de ejecucion")
    for row in rows:
        if row.status == "ok":
            print(
                f"- {row.method_label}: OK, {row.frames} frames, "
                f"{row.pipeline_total_s:.3f} s, {row.effective_fps:.3f} FPS efectivos"
            )
        else:
            print(f"- {row.method_label}: ERROR, {row.error}")

    if any(row.status != "ok" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
