# TP3.2 - Sobel, entrega 2

Alcance: medir el caso nuevo `numba_gpu` usando CUDA y compararlo en el informe contra los resultados de la entrega 1.

## Requisitos

- Python 3.14
- `numpy`
- `numba`
- `pillow`
- GPU compatible con CUDA y driver instalado

Instalación:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verificar CUDA desde Numba:

```bash
python - <<'PY'
from numba import cuda
print(cuda.is_available())
if cuda.is_available():
    print(cuda.get_current_device().name)
PY
```

## Corrida completa

Desde `tp3.2`:

```bash
source .venv/bin/activate
python correr_entrega2.py
```

Esto genera:

- `resultados_sobel_entrega2.csv`
- `resultados_sobel_entrega2.md`
- `informe_sobel_entrega2.md`
- previews en `salidas/`

## Corrida rápida

```bash
python benchmark_entrega2.py --sizes 750 --runs 1 --save-preview
```

Para regenerar solo el informe desde un CSV existente:

```bash
python generar_informe_entrega2.py
```

El benchmark excluye I/O de imágenes. Para GPU, se reportan transferencias CPU<->GPU por separado porque son parte del costo práctico de la versión CUDA.
