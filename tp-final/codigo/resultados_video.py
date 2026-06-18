"""Utilidades compartidas para el TP final de video 4K con filtro sepia."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

from sepia_filter import METHOD_LABELS


CSV_FIELDS = [
    "timestamp",
    "method_key",
    "method_label",
    "runs",
    "workers",
    "input_path",
    "output_video_path",
    "output_video_with_audio_path",
    "frames",
    "width",
    "height",
    "fps_original",
    "duration_s",
    "read_s",
    "filter_s",
    "filter_compute_s",
    "write_s",
    "pipeline_total_s",
    "effective_fps",
    "transfer_h2d_s",
    "transfer_d2h_s",
    "transfer_total_s",
    "audio_merge_s",
    "codec",
    "max_frames",
    "peak_rss_mb",
    "checksum",
    "output_hash",
    "status",
    "error",
]


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


@dataclass
class RunMeasurement:
    run_index: int
    frames: int
    read_s: float
    filter_s: float
    filter_compute_s: float
    write_s: float
    pipeline_total_s: float
    effective_fps: float
    transfer_h2d_s: float | None
    transfer_d2h_s: float | None
    transfer_total_s: float | None
    audio_merge_s: float | None
    peak_rss_mb: float | None
    checksum: int
    output_hash: str
    output_video_path: str
    output_video_with_audio_path: str


@dataclass
class SummaryRow:
    timestamp: str
    method_key: str
    method_label: str
    runs: int
    workers: int
    input_path: str
    output_video_path: str
    output_video_with_audio_path: str
    frames: int | None
    width: int | None
    height: int | None
    fps_original: float | None
    duration_s: float | None
    read_s: float | None
    filter_s: float | None
    filter_compute_s: float | None
    write_s: float | None
    pipeline_total_s: float | None
    effective_fps: float | None
    transfer_h2d_s: float | None
    transfer_d2h_s: float | None
    transfer_total_s: float | None
    audio_merge_s: float | None
    codec: str
    max_frames: int | None
    peak_rss_mb: float | None
    checksum: int | None
    output_hash: str | None
    status: str
    error: str


def _parse_fraction(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return float(Fraction(value))
    except Exception:
        try:
            return float(value)
        except Exception:
            return 0.0


def _video_info_ffprobe(path: Path) -> VideoInfo | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
        data = json.loads(result.stdout)
    except Exception:
        return None

    streams = data.get("streams") or []
    if not streams:
        return None

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_fraction(stream.get("avg_frame_rate"))
    duration_s = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nb_frames = stream.get("nb_frames")
    if nb_frames and nb_frames != "N/A":
        frame_count = int(nb_frames)
    else:
        frame_count = int(round(duration_s * fps)) if duration_s and fps else 0

    if width <= 0 or height <= 0 or fps <= 0:
        return None
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
    )


def _video_info_opencv(path: Path) -> VideoInfo:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Falta OpenCV. Instalalo en el entorno de ejecucion.") from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    duration_s = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
    )


def video_info(path: Path) -> VideoInfo:
    """Devuelve informacion basica del video.

    Se intenta usar ffprobe primero porque detecta mejor contenedores WebM/AV1.
    Si no esta disponible, se cae a OpenCV.
    """
    info = _video_info_ffprobe(path)
    if info is not None:
        return info
    return _video_info_opencv(path)


def output_video_path(output_dir: Path, method_key: str) -> Path:
    return output_dir / "videos" / f"sepia_{method_key}_sin_audio.mp4"


def output_video_with_audio_path(output_dir: Path, method_key: str) -> Path:
    return output_dir / "videos" / f"sepia_{method_key}_con_audio.mp4"


def csv_path(output_dir: Path) -> Path:
    return output_dir / "resultados_video4k_sepia.csv"


def partial_md_path(output_dir: Path, method_key: str) -> Path:
    return output_dir / "parciales" / f"resultado_parcial_sepia_{method_key}.md"


def final_md_path(output_dir: Path) -> Path:
    return output_dir / "finales" / "resumen_video4k_sepia.md"


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "no disponible"
    except Exception:
        return "sin dato"


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "no detectado"


def cpu_physical_cores() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return "no detectado"

    pairs: set[tuple[str, str]] = set()
    physical_id = ""
    core_id = ""
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            if physical_id and core_id:
                pairs.add((physical_id, core_id))
            physical_id = ""
            core_id = ""
            continue
        if line.startswith("physical id"):
            physical_id = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core_id = line.split(":", 1)[1].strip()

    if physical_id and core_id:
        pairs.add((physical_id, core_id))
    return str(len(pairs)) if pairs else "no detectado"


def ram_info() -> str:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return "no detectada"
    total_kb = None
    available_kb = None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match(r"^(MemTotal|MemAvailable):\s+(\d+)\s+kB", line)
        if not match:
            continue
        if match.group(1) == "MemTotal":
            total_kb = int(match.group(2))
        elif match.group(1) == "MemAvailable":
            available_kb = int(match.group(2))

    if total_kb is None:
        return "no detectada"
    total_gib = total_kb / (1024 * 1024)
    if available_kb is None:
        return f"{total_gib:.2f} GiB total"
    available_gib = available_kb / (1024 * 1024)
    return f"{total_gib:.2f} GiB total, {available_gib:.2f} GiB disponible"


def pytorch_cuda_info(timeout_s: int = 10) -> str:
    code = (
        "import torch; "
        "print('CUDA no disponible para PyTorch' if not torch.cuda.is_available() "
        "else f'{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda or \"sin version CUDA\"})')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return result.stdout.strip() or "sin dato"
    except subprocess.TimeoutExpired:
        return f"no detectada (import torch excedio {timeout_s} s)"
    except Exception as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        detail = detail.strip().splitlines()[-1] if detail.strip() else type(exc).__name__
        return f"no detectada ({detail})"


def environment_info() -> dict[str, str]:
    return {
        "cpu_model": cpu_model(),
        "cpu_physical_cores": cpu_physical_cores(),
        "cpu_logical_cores": str(os.cpu_count() or "no detectado"),
        "ram": ram_info(),
        "os": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "opencv": package_version("opencv-python"),
        "torch": package_version("torch"),
        "pytorch_cuda": pytorch_cuda_info(),
    }


def peak_rss_mb() -> float | None:
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return None


def merge_audio_with_ffmpeg(input_video: Path, filtered_video: Path, output_video: Path) -> float:
    """Reincorpora audio con ffmpeg y devuelve el tiempo del paso externo."""
    from time import perf_counter

    output_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filtered_video),
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]
    start = perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return perf_counter() - start


def average_measurements(
    method_key: str,
    runs: int,
    workers: int,
    input_path: Path,
    info: VideoInfo,
    codec: str,
    max_frames: int | None,
    measurements: list[RunMeasurement],
    status: str = "ok",
    error: str = "",
) -> SummaryRow:
    timestamp = datetime.now().isoformat(timespec="seconds")
    method_label = METHOD_LABELS[method_key]
    if not measurements:
        return SummaryRow(
            timestamp=timestamp,
            method_key=method_key,
            method_label=method_label,
            runs=runs,
            workers=workers,
            input_path=str(input_path),
            output_video_path="",
            output_video_with_audio_path="",
            frames=None,
            width=info.width,
            height=info.height,
            fps_original=info.fps,
            duration_s=info.duration_s,
            read_s=None,
            filter_s=None,
            filter_compute_s=None,
            write_s=None,
            pipeline_total_s=None,
            effective_fps=None,
            transfer_h2d_s=None,
            transfer_d2h_s=None,
            transfer_total_s=None,
            audio_merge_s=None,
            codec=codec,
            max_frames=max_frames,
            peak_rss_mb=None,
            checksum=None,
            output_hash=None,
            status=status,
            error=error,
        )

    def avg(values: Iterable[float]) -> float:
        values_list = list(values)
        return sum(values_list) / len(values_list)

    def avg_optional(values: Iterable[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        return (sum(present) / len(present)) if present else None

    last = measurements[-1]
    return SummaryRow(
        timestamp=timestamp,
        method_key=method_key,
        method_label=method_label,
        runs=runs,
        workers=workers,
        input_path=str(input_path),
        output_video_path=last.output_video_path,
        output_video_with_audio_path=last.output_video_with_audio_path,
        frames=last.frames,
        width=info.width,
        height=info.height,
        fps_original=info.fps,
        duration_s=info.duration_s,
        read_s=avg(m.read_s for m in measurements),
        filter_s=avg(m.filter_s for m in measurements),
        filter_compute_s=avg(m.filter_compute_s for m in measurements),
        write_s=avg(m.write_s for m in measurements),
        pipeline_total_s=avg(m.pipeline_total_s for m in measurements),
        effective_fps=avg(m.effective_fps for m in measurements),
        transfer_h2d_s=avg_optional(m.transfer_h2d_s for m in measurements),
        transfer_d2h_s=avg_optional(m.transfer_d2h_s for m in measurements),
        transfer_total_s=avg_optional(m.transfer_total_s for m in measurements),
        audio_merge_s=avg_optional(m.audio_merge_s for m in measurements),
        codec=codec,
        max_frames=max_frames,
        peak_rss_mb=avg_optional(m.peak_rss_mb for m in measurements),
        checksum=last.checksum,
        output_hash=last.output_hash,
        status=status,
        error=error,
    )


def _summary_to_csv_row(row: SummaryRow) -> dict[str, str]:
    def f(value: float | None) -> str:
        return f"{value:.9f}" if value is not None else ""

    def i(value: int | None) -> str:
        return str(value) if value is not None else ""

    return {
        "timestamp": row.timestamp,
        "method_key": row.method_key,
        "method_label": row.method_label,
        "runs": str(row.runs),
        "workers": str(row.workers),
        "input_path": row.input_path,
        "output_video_path": row.output_video_path,
        "output_video_with_audio_path": row.output_video_with_audio_path,
        "frames": i(row.frames),
        "width": i(row.width),
        "height": i(row.height),
        "fps_original": f(row.fps_original),
        "duration_s": f(row.duration_s),
        "read_s": f(row.read_s),
        "filter_s": f(row.filter_s),
        "filter_compute_s": f(row.filter_compute_s),
        "write_s": f(row.write_s),
        "pipeline_total_s": f(row.pipeline_total_s),
        "effective_fps": f(row.effective_fps),
        "transfer_h2d_s": f(row.transfer_h2d_s),
        "transfer_d2h_s": f(row.transfer_d2h_s),
        "transfer_total_s": f(row.transfer_total_s),
        "audio_merge_s": f(row.audio_merge_s),
        "codec": row.codec,
        "max_frames": i(row.max_frames),
        "peak_rss_mb": f(row.peak_rss_mb),
        "checksum": i(row.checksum),
        "output_hash": row.output_hash or "",
        "status": row.status,
        "error": row.error,
    }


def _csv_row_to_summary(row: dict[str, str]) -> SummaryRow:
    def f(value: str) -> float | None:
        return float(value) if value.strip() else None

    def i(value: str) -> int | None:
        return int(value) if value.strip() else None

    return SummaryRow(
        timestamp=row.get("timestamp", ""),
        method_key=row["method_key"],
        method_label=row.get("method_label", METHOD_LABELS.get(row["method_key"], row["method_key"])),
        runs=int(row.get("runs", "1") or "1"),
        workers=int(row.get("workers", "1") or "1"),
        input_path=row.get("input_path", ""),
        output_video_path=row.get("output_video_path", ""),
        output_video_with_audio_path=row.get("output_video_with_audio_path", ""),
        frames=i(row.get("frames", "")),
        width=i(row.get("width", "")),
        height=i(row.get("height", "")),
        fps_original=f(row.get("fps_original", "")),
        duration_s=f(row.get("duration_s", "")),
        read_s=f(row.get("read_s", "")),
        filter_s=f(row.get("filter_s", "")),
        filter_compute_s=f(row.get("filter_compute_s", "")),
        write_s=f(row.get("write_s", "")),
        pipeline_total_s=f(row.get("pipeline_total_s", "")),
        effective_fps=f(row.get("effective_fps", "")),
        transfer_h2d_s=f(row.get("transfer_h2d_s", "")),
        transfer_d2h_s=f(row.get("transfer_d2h_s", "")),
        transfer_total_s=f(row.get("transfer_total_s", "")),
        audio_merge_s=f(row.get("audio_merge_s", "")),
        codec=row.get("codec", ""),
        max_frames=i(row.get("max_frames", "")),
        peak_rss_mb=f(row.get("peak_rss_mb", "")),
        checksum=i(row.get("checksum", "")),
        output_hash=row.get("output_hash", "") or None,
        status=row.get("status", "ok"),
        error=row.get("error", ""),
    )


def load_csv_rows(path: Path) -> list[SummaryRow]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [_csv_row_to_summary(row) for row in csv.DictReader(handle)]


def write_csv_rows(path: Path, rows: list[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_summary_to_csv_row(row))


def update_csv(path: Path, new_row: SummaryRow) -> list[SummaryRow]:
    old_rows = load_csv_rows(path)
    kept = [row for row in old_rows if row.method_key != new_row.method_key]
    rows = kept + [new_row]
    order = {"secuencial": 0, "pytorch_cpu": 1, "pytorch_gpu": 2}
    rows.sort(key=lambda row: order.get(row.method_key, 99))
    write_csv_rows(path, rows)
    return rows


def speedup_data(rows: list[SummaryRow]) -> dict[str, float | None]:
    baseline = None
    for row in rows:
        if row.method_key == "secuencial" and row.status == "ok" and row.pipeline_total_s:
            baseline = row.pipeline_total_s
            break

    data: dict[str, float | None] = {}
    for row in rows:
        if baseline and row.status == "ok" and row.pipeline_total_s and row.pipeline_total_s > 0:
            data[row.method_key] = baseline / row.pipeline_total_s
        else:
            data[row.method_key] = None
    return data


def format_float(value: float | None, digits: int = 6) -> str:
    return f"{value:.{digits}f}" if value is not None else ""


def format_int(value: int | None) -> str:
    return str(value) if value is not None else ""


def md_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_results_md(
    path: Path,
    title: str,
    rows: list[SummaryRow],
    env: dict[str, str],
    all_rows_for_speedup: list[SummaryRow] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    speed_rows = all_rows_for_speedup if all_rows_for_speedup is not None else rows
    speedups = speedup_data(speed_rows)

    lines = [
        f"# {title}",
        "",
        "## Entorno",
        "",
        f"- CPU: {env['cpu_model']}",
        f"- Nucleos fisicos: {env['cpu_physical_cores']}",
        f"- Nucleos logicos: {env['cpu_logical_cores']}",
        f"- RAM: {env['ram']}",
        f"- Sistema operativo: {env['os']}",
        f"- Python: {env['python']}",
        f"- OpenCV: {env['opencv']}",
        f"- PyTorch: {env['torch']}",
        f"- PyTorch CUDA: {env['pytorch_cuda']}",
        "",
        "## Filtro",
        "",
        "Filtro elegido: sepia.",
        "",
        "Formula aplicada por pixel, usando canales RGB y salida BGR para OpenCV:",
        "",
        "```text",
        "R' = min(255, (393R + 769G + 189B) // 1000)",
        "G' = min(255, (349R + 686G + 168B) // 1000)",
        "B' = min(255, (272R + 534G + 131B) // 1000)",
        "salida OpenCV = B', G', R'",
        "```",
        "",
        "## Tabla de benchmark",
        "",
        "| metodo | frames | resolucion | fps original | lectura/decodif. (s) | filtrado (s) | escritura/codif. (s) | total pipeline (s) | FPS efectivos | speed-up | memoria pico (MB) | estado |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        resolution = f"{row.width}x{row.height}" if row.width and row.height else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(row.method_label),
                    format_int(row.frames),
                    resolution,
                    format_float(row.fps_original, 3),
                    format_float(row.read_s),
                    format_float(row.filter_s),
                    format_float(row.write_s),
                    format_float(row.pipeline_total_s),
                    format_float(row.effective_fps),
                    format_float(speedups.get(row.method_key)),
                    format_float(row.peak_rss_mb, 2),
                    md_escape(row.status),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Datos de control",
            "",
            "| metodo | corridas | workers | duracion entrada (s) | codec | max_frames | checksum | hash salida | video sin audio | video con audio |",
            "|---|---:|---:|---:|---|---:|---:|---|---|---|",
        ]
    )

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(row.method_label),
                    str(row.runs),
                    str(row.workers),
                    format_float(row.duration_s),
                    md_escape(row.codec),
                    format_int(row.max_frames),
                    format_int(row.checksum),
                    row.output_hash or "",
                    md_escape(row.output_video_path),
                    md_escape(row.output_video_with_audio_path),
                ]
            )
            + " |"
        )

    if any(row.transfer_total_s is not None for row in rows):
        lines.extend(
            [
                "",
                "## Detalle PyTorch GPU",
                "",
                "La columna filtrado de la tabla principal incluye transferencia CPU-GPU, computo GPU y vuelta GPU-CPU, porque el frame se lee y se escribe desde CPU.",
                "",
                "| metodo | transferencia CPU->GPU (s) | computo GPU (s) | transferencia GPU->CPU (s) | transferencia total (s) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row.transfer_total_s is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_escape(row.method_label),
                        format_float(row.transfer_h2d_s),
                        format_float(row.filter_compute_s),
                        format_float(row.transfer_d2h_s),
                        format_float(row.transfer_total_s),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Notas metodologicas",
            "",
            "- El video se procesa como flujo: no se carga completo en memoria.",
            "- El tiempo de pipeline es lectura/decodificacion + filtrado + escritura/codificacion.",
            "- El merge de audio con ffmpeg se mide aparte y no se suma al pipeline de filtrado.",
            "- Speed-up = tiempo total del pipeline secuencial / tiempo total del pipeline del metodo.",
            "- Si falta la fila secuencial, el speed-up queda vacio porque falta la linea base.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def frame_hash() -> hashlib._Hash:
    return hashlib.sha256()
