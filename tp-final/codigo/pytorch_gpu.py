"""Medicion individual: filtro sepia con PyTorch GPU."""

from __future__ import annotations

from procesamiento_video import ConfigMedicion, ejecutar_medicion, parse_args_medicion


METHOD_KEY = "pytorch_gpu"


def run(config: ConfigMedicion | None = None):
    return ejecutar_medicion(METHOD_KEY, config or ConfigMedicion())


def main() -> None:
    config = parse_args_medicion("Ejecuta la medicion PyTorch GPU del filtro sepia.")
    row = run(config)
    raise SystemExit(0 if row.status == "ok" else 1)


if __name__ == "__main__":
    main()
