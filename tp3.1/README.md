# TP3 - Sobel, entrega 7 de mayo

Alcance de esta entrega: `secuencial`, `numpy` y `numba_cpu` paralelo. Las versiones GPU/PyTorch quedan para las entregas siguientes del enunciado.

## Preparacion

Desde esta carpeta (`tp3`), activar el entorno:

```bash
source .venv/bin/activate
```

Si hiciera falta reinstalar dependencias:

```bash
pip install -r requirements.txt
```

Las imagenes del ZIP ya se esperan en `imagenes/` con estos nombres:

- `IMG_0358_750x750.jpg`
- `IMG_0358_1500x1500.jpg`
- `IMG_0358_3000x3000.jpg`
- `IMG_0358_6000x6000.jpg`

## Corrida rapida

```bash
python benchmark.py --sizes 750 --runs 1
```

Esto genera dos archivos:

- `resultados_sobel_entrega1.csv`
- `resultados_sobel_entrega1.md`

## Benchmark para la entrega

La forma recomendada es correr el archivo de configuracion:

```bash
python correr_entrega1.py
```

Ese script ejecuta:

- tamanios: `750`, `1500`, `3000`, `6000`
- metodos: `secuencial`, `numpy`, `numba_cpu`
- corridas por caso: `5`
- salidas: `resultados_sobel_entrega1.csv` y `resultados_sobel_entrega1.md`

Tambien se puede correr manualmente con flags:

```bash
python benchmark.py --runs 5 --output resultados_sobel_entrega1.csv
```

Tambien podes elegir explicitamente el Markdown:

```bash
python benchmark.py --runs 5 --output resultados_sobel_entrega1.csv --md-output resultados_sobel_entrega1.md
```

Si queres guardar una imagen de salida por metodo y tamanio:

```bash
python benchmark.py --runs 5 --save-preview
```

## Scripts individuales

```bash
python sobel_secuencial.py --size 750 --runs 5 --output salidas/sobel_secuencial_750.png
python sobel_numpy.py --size 750 --runs 5 --output salidas/sobel_numpy_750.png
python sobel_numba_cpu.py --size 750 --runs 5 --workers 8 --output salidas/sobel_numba_cpu_750.png
```

El benchmark excluye carga y guardado de imagenes. Solo mide conversion `RGB->gris` y Sobel.
