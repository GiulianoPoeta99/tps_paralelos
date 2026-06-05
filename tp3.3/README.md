# TP3.3 - Sobel, entrega 3

Alcance: medir los casos nuevos `pytorch_cpu` y `pytorch_gpu`, y generar un informe combinado con los resultados ya obtenidos en `tp3.1` y `tp3.2`.

## Requisitos

- Python 3.14
- `numpy`
- `pillow`
- `torch`
- GPU compatible con CUDA y driver instalado para ejecutar `pytorch_gpu`

Instalación:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verificar CUDA desde PyTorch:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)
PY
```

## Corrida completa

Desde `tp3.3`:

```bash
source .venv/bin/activate
python correr_entrega3.py
```

Esto mide solo:

- `pytorch_cpu`
- `pytorch_gpu`

Y genera:

- `resultados_sobel_entrega3.csv`
- `resultados_sobel_entrega3.md`
- `informe_sobel_entrega3.md`
- previews en `salidas/`

El informe toma como base:

- `../tp3.1/resultados_sobel_entrega1.csv`
- `../tp3.2/resultados_sobel_entrega2.csv`

## Corrida rápida

```bash
python benchmark_entrega3.py --sizes 750 --runs 1 --methods pytorch_cpu --save-preview
```

Si ya se midió `pytorch_cpu` y solo falta completar `pytorch_gpu`:

```bash
python benchmark_entrega3.py --methods pytorch_gpu --runs 5 --save-preview
python generar_informe_entrega3.py
```

Para regenerar solo el informe desde CSV existentes:

```bash
python generar_informe_entrega3.py
```

## Generar DOCX/PDF

Después de correr `python correr_entrega3.py`, generar los documentos finales con:

```bash
pandoc informe_sobel_entrega3.md -o 04-giuliano-poeta-sobel-entrega3.docx
pandoc informe_sobel_entrega3.md -o 04-giuliano-poeta-sobel-entrega3.pdf
```

El benchmark excluye I/O de imágenes. Para PyTorch GPU, se reportan transferencias CPU<->GPU por separado porque son parte del costo práctico de la versión GPU.
