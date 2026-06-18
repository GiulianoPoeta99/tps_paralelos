"""Procesamiento comun del video 4K con filtro sepia."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from time import perf_counter

from sepia_filter import METHOD_LABELS, build_processor
from resultados_video import (
    RunMeasurement,
    SummaryRow,
    average_measurements,
    csv_path,
    environment_info,
    frame_hash,
    merge_audio_with_ffmpeg,
    output_video_path,
    output_video_with_audio_path,
    partial_md_path,
    peak_rss_mb,
    update_csv,
    video_info,
    write_results_md,
)


DEFAULT_WORKERS = 4
DEFAULT_SECONDS = 30.0
DEFAULT_RUNS = 1
DEFAULT_CODEC = "mp4v"


def default_input_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    converted_clip = base_dir / "entrada" / "video_facultad_30s_h264.mp4"
    if converted_clip.exists():
        return converted_clip
    return base_dir / "video_facultad.webm"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "resultados"


@dataclass
class ConfigMedicion:
    input_path: Path = field(default_factory=default_input_path)
    output_dir: Path = field(default_factory=default_output_dir)
    runs: int = DEFAULT_RUNS
    workers: int = DEFAULT_WORKERS
    codec: str = DEFAULT_CODEC
    seconds: float | None = DEFAULT_SECONDS
    max_frames: int | None = None
    merge_audio: bool = False
    show_progress: bool = True
    progress_every: int = 120


def frame_limit_from_config(fps: float, config: ConfigMedicion) -> int | None:
    if config.max_frames is not None:
        return config.max_frames
    if config.seconds is not None:
        return max(1, int(ceil(fps * config.seconds)))
    return None


def process_video_once(
    *,
    run_index: int,
    input_path: Path,
    output_dir: Path,
    method_key: str,
    processor,
    codec: str,
    frame_limit: int | None,
    merge_audio: bool,
    show_progress: bool,
    progress_every: int,
) -> RunMeasurement:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Falta OpenCV en el entorno de ejecucion.") from exc

    info = video_info(input_path)
    if info.width <= 0 or info.height <= 0:
        raise RuntimeError("No se pudo detectar la resolucion del video")
    if info.fps <= 0:
        raise RuntimeError("No se pudo detectar el FPS del video")

    output_path = output_video_path(output_dir, method_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, info.fps, (info.width, info.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo crear el video de salida: {output_path}")

    frames = 0
    read_s = 0.0
    filter_s = 0.0
    filter_compute_s = 0.0
    write_s = 0.0
    transfer_h2d_s = 0.0
    transfer_d2h_s = 0.0
    checksum = 0
    hasher = frame_hash()

    try:
        while frame_limit is None or frames < frame_limit:
            read_start = perf_counter()
            ok, frame = cap.read()
            read_s += perf_counter() - read_start
            if not ok:
                break

            out_frame, timing = processor.process(frame)
            filter_s += timing.filter_s
            filter_compute_s += timing.compute_s
            if timing.transfer_h2d_s is not None:
                transfer_h2d_s += timing.transfer_h2d_s
                transfer_d2h_s += timing.transfer_d2h_s or 0.0

            # Control de salida fuera de las mediciones de I/O y filtrado.
            output_bytes = out_frame.tobytes()
            checksum += sum(output_bytes)
            hasher.update(output_bytes)

            write_start = perf_counter()
            writer.write(out_frame)
            write_s += perf_counter() - write_start

            frames += 1
            if show_progress and (frames == 1 or frames % progress_every == 0):
                limit_label = str(frame_limit) if frame_limit is not None else "fin del video"
                print(
                    f"[{METHOD_LABELS[method_key]}] corrida {run_index}: {frames}/{limit_label} frames",
                    flush=True,
                )

            del frame
            del out_frame
    finally:
        cap.release()
        writer.release()

    pipeline_total_s = read_s + filter_s + write_s
    effective_fps = frames / pipeline_total_s if pipeline_total_s > 0 else 0.0

    audio_merge_s = None
    audio_path = ""
    if merge_audio:
        audio_output = output_video_with_audio_path(output_dir, method_key)
        audio_merge_s = merge_audio_with_ffmpeg(input_path, output_path, audio_output)
        audio_path = str(audio_output)

    return RunMeasurement(
        run_index=run_index,
        frames=frames,
        read_s=read_s,
        filter_s=filter_s,
        filter_compute_s=filter_compute_s,
        write_s=write_s,
        pipeline_total_s=pipeline_total_s,
        effective_fps=effective_fps,
        transfer_h2d_s=transfer_h2d_s if transfer_h2d_s > 0.0 else None,
        transfer_d2h_s=transfer_d2h_s if transfer_d2h_s > 0.0 else None,
        transfer_total_s=(transfer_h2d_s + transfer_d2h_s) if transfer_h2d_s > 0.0 else None,
        audio_merge_s=audio_merge_s,
        peak_rss_mb=peak_rss_mb(),
        checksum=checksum,
        output_hash=hasher.hexdigest()[:16],
        output_video_path=str(output_path),
        output_video_with_audio_path=audio_path,
    )


def ejecutar_medicion(method_key: str, config: ConfigMedicion) -> SummaryRow:
    if method_key not in METHOD_LABELS:
        valid = ", ".join(METHOD_LABELS)
        raise ValueError(f"Metodo invalido: {method_key}. Validos: {valid}")

    if config.runs < 1:
        raise ValueError("runs debe ser >= 1")
    if config.workers < 1:
        raise ValueError("workers debe ser >= 1")
    if len(config.codec) != 4:
        raise ValueError("codec debe tener exactamente 4 caracteres, por ejemplo mp4v")
    if config.seconds is not None and config.seconds <= 0:
        raise ValueError("seconds debe ser > 0")
    if config.max_frames is not None and config.max_frames < 1:
        raise ValueError("max_frames debe ser >= 1")

    input_path = config.input_path.resolve()
    output_dir = config.output_dir.resolve()
    info = video_info(input_path)
    frame_limit = frame_limit_from_config(info.fps, config)
    env = environment_info()
    measurements: list[RunMeasurement] = []
    processor_workers = config.workers

    try:
        processor = build_processor(method_key, workers=config.workers)
        processor_workers = int(getattr(processor, "workers", config.workers))
        for run_index in range(1, config.runs + 1):
            print(
                f"[{METHOD_LABELS[method_key]}] iniciando corrida {run_index}/{config.runs}",
                flush=True,
            )
            measurement = process_video_once(
                run_index=run_index,
                input_path=input_path,
                output_dir=output_dir,
                method_key=method_key,
                processor=processor,
                codec=config.codec,
                frame_limit=frame_limit,
                merge_audio=config.merge_audio,
                show_progress=config.show_progress,
                progress_every=config.progress_every,
            )
            measurements.append(measurement)
            print(
                f"[{METHOD_LABELS[method_key]}] corrida {run_index} finalizada: "
                f"{measurement.frames} frames, {measurement.pipeline_total_s:.3f} s",
                flush=True,
            )
    except Exception as exc:
        row = average_measurements(
            method_key,
            config.runs,
            processor_workers,
            input_path,
            info,
            config.codec,
            frame_limit,
            measurements,
            status="error",
            error=str(exc),
        )
    else:
        row = average_measurements(
            method_key,
            config.runs,
            processor_workers,
            input_path,
            info,
            config.codec,
            frame_limit,
            measurements,
        )

    all_rows = update_csv(csv_path(output_dir), row)
    write_results_md(
        partial_md_path(output_dir, method_key),
        f"Resultado parcial TP final sepia - {METHOD_LABELS[method_key]}",
        [row],
        env,
        all_rows_for_speedup=all_rows,
    )
    write_results_md(
        output_dir / "resultados_video4k_sepia.md",
        "Resultados TP final sepia",
        all_rows,
        env,
        all_rows_for_speedup=all_rows,
    )

    print(f"[{METHOD_LABELS[method_key]}] CSV actualizado: {csv_path(output_dir)}", flush=True)
    print(f"[{METHOD_LABELS[method_key]}] Markdown parcial: {partial_md_path(output_dir, method_key)}", flush=True)
    print(f"[{METHOD_LABELS[method_key]}] Markdown agregado: {output_dir / 'resultados_video4k_sepia.md'}", flush=True)
    if row.status != "ok":
        print(f"[{METHOD_LABELS[method_key]}] error registrado: {row.error}", flush=True)
    return row


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=default_input_path(), help="Video de entrada.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Carpeta de resultados.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Cantidad de corridas.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Hilos PyTorch CPU. Default: 4.")
    parser.add_argument("--codec", default=DEFAULT_CODEC, help="Codec OpenCV de 4 caracteres. Default: mp4v.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help="Segundos a procesar desde el inicio. Default: 30.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Procesa una cantidad exacta de frames. Tiene prioridad sobre --seconds.",
    )
    parser.add_argument(
        "--merge-audio",
        action="store_true",
        help="Reincorpora audio con ffmpeg al video procesado.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Oculta progreso por frames.")
    parser.add_argument("--progress-every", type=int, default=120, help="Frecuencia de progreso por frames.")


def config_from_args(args: argparse.Namespace) -> ConfigMedicion:
    return ConfigMedicion(
        input_path=args.input,
        output_dir=args.output_dir,
        runs=args.runs,
        workers=args.workers,
        codec=args.codec,
        seconds=args.seconds,
        max_frames=args.max_frames,
        merge_audio=args.merge_audio,
        show_progress=not args.no_progress,
        progress_every=args.progress_every,
    )


def parse_args_medicion(description: str) -> ConfigMedicion:
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(parser)
    return config_from_args(parser.parse_args())


def print_divider(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}", flush=True)
