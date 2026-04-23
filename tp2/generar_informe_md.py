"""Genera matrices/resultados/<nombre_del_csv>.md a partir del CSV de benchmark."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _norm_row(row: dict[str, str]) -> dict[str, str]:
    """Compatibilidad con CSV antiguos (solo speed_up / eficiencia = A·B)."""
    out = dict(row)
    if out.get("speed_up_ab", "").strip() == "" and out.get("speed_up", "").strip() != "":
        out["speed_up_ab"] = out["speed_up"]
        out["eficiencia_ab"] = out.get("eficiencia", "")
    return out


def _title_c(rows: list[dict[str, str]]) -> str:
    cs = sorted({int(r["complejidad_c"]) for r in rows if (r.get("complejidad_c") or "").strip().isdigit()})
    if not cs:
        return "C = ?"
    if len(cs) == 1:
        return f"C = {cs[0]}"
    return "C ∈ {" + ", ".join(str(c) for c in cs) + "}"


def _escape_cell(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def informe_md_path_for_csv(csv_path: Path, matrices_dir: Path) -> Path:
    """
    CSV generado por el benchmark (cualquier nombre) -> matrices/resultados/<mismo_stem>.md
    matrices_dir: directorio matrices/ (padre de resultados/).
    """
    return (matrices_dir / "resultados" / f"{csv_path.resolve().stem}.md").resolve()


def write_informe_md(csv_path: Path, md_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)

    rows = [_norm_row(r) for r in raw_rows]
    title_c = _title_c(rows)

    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# Resultados benchmark — multiplicación de matrices ({title_c})",
        "",
        f"Fuente: `{csv_path.name}` (ruta relativa al directorio `matrices/` según ubicación del CSV).",
        "",
        "Las columnas **speed_up_*** y **eficiencia_*** usan como baseline el tiempo del método **secuencial** (mismo C, perfil y max-val):",
        "- **_ab**: respecto al tiempo del producto **A·B**.",
        "- **_btat**: respecto al tiempo del producto **Bᵀ·Aᵀ**.",
        "",
        "Los **checksum** son la suma de todos los elementos de cada matriz resultado (deben coincidir Σ(A·B) y Σ(Bᵀ·Aᵀ)).",
        "",
        "## Tabla",
        "",
    ]

    headers = [
        ("algoritmo", "Algoritmo"),
        ("complejidad_c", "C"),
        ("perfil", "Perfil"),
        ("max_val", "max-val"),
        ("procesos_workers", "Workers"),
        ("tiempo_segundos_ab", "t A·B (s)"),
        ("tiempo_segundos_btat", "t Bᵀ·Aᵀ (s)"),
        ("speed_up_ab", "Speed-up A·B"),
        ("eficiencia_ab", "Eficiencia A·B"),
        ("speed_up_btat", "Speed-up Bᵀ·Aᵀ"),
        ("eficiencia_btat", "Eficiencia Bᵀ·Aᵀ"),
        ("equipo_cores", "Cores"),
        ("estado", "Estado"),
        ("checksum_ab", "Σ(A·B)"),
        ("checksum_btat", "Σ(Bᵀ·Aᵀ)"),
    ]

    lines.append("| " + " | ".join(h[1] for h in headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for row in rows:
        cells = [_escape_cell(row.get(k, "") or "") for k, _ in headers]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Notas",
            "",
            "- **Eficiencia** = speed-up / número de workers.",
            "- Si falta speed-up/eficiencia para Bᵀ·Aᵀ en un CSV viejo, volvé a ejecutar `benchmark.py` para regenerar columnas.",
            "",
        ]
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Genera resultados/resultados_matrices.md desde el CSV")
    parser.add_argument(
        "--csv",
        type=Path,
        default=base / "resultados_matrices.csv",
        help="Ruta al CSV (por defecto matrices/resultados_matrices.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "resultados" / "resultados_matrices.md",
        help="Ruta al markdown de salida",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    md_path = args.output.resolve()

    try:
        write_informe_md(csv_path, md_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Informe generado: {md_path}")


if __name__ == "__main__":
    main()
