# TP1 — Suma paralela de un vector con Free-Threaded Python

Trabajo práctico de Sistemas Paralelos (UNTDF). Compara el rendimiento de distintas estrategias de concurrencia en Python para sumar un vector de complejidad `c`, aprovechando Free-Threaded Python (sin GIL).

## Requisitos

- Linux (probado en Manjaro)
- [pyenv](https://github.com/pyenv/pyenv) + [uv](https://docs.astral.sh/uv/)

> **Setup completo:** ver [docs/pyenv.md](docs/pyenv.md)

## Estructura del proyecto

```
TP1/
├── sumaParalela/
│   ├── suma_lib.py                   # Funciones compartidas (vector, chunks, impresión)
│   ├── secuential.py                 # Baseline secuencial
│   ├── suma_threadpoolexecutor.py    # ThreadPoolExecutor
│   ├── suma_processpoolexecutor.py   # ProcessPoolExecutor
│   ├── suma_multiprocessing.py       # multiprocessing.Pool
│   ├── suma_threading.py             # threading manual
│   └── benchmark.py                  # Script automático de benchmarking
├── docs/
│   ├── pyenv.md                      # Guía de instalación y configuración
│   ├── enunciado.md                  # Consigna del TP
│   └── estandar-paper.md             # Estructura estándar del informe
└── README.md
```

## Uso

```bash
source .venv/bin/activate
```

Ejecutar variantes individuales:

```bash
cd sumaParalela
python secuential.py --workers 1 --c 1000000
python suma_threadpoolexecutor.py --workers 4 --c 1000000
python suma_processpoolexecutor.py --workers 4 --c 1000000
python suma_multiprocessing.py --workers 4 --c 1000000
python suma_threading.py --workers 4 --c 1000000
```

Para forzar el GIL desactivado explícitamente:

```bash
PYTHON_GIL=0 python src/benchmark.py
# o
python -X gil=0 src/benchmark.py
```

## Benchmark automático

Corre todas las combinaciones y exporta una tabla CSV:

```bash
cd sumaParalela
python benchmark.py
```

Por defecto usa `p = 1,2,4,8,16`, `c = 12,1000000,100000000` y genera `resultados_suma.csv`.

Opciones:

```bash
python benchmark.py --output tabla_tp.csv
python benchmark.py --p-values 1,2,4 --c-values 12,1000000
```

### Carga incremental

`benchmark.py` acumula resultados por defecto: si el CSV ya existe, no repite combinaciones calculadas y agrega solo lo faltante, recalculando `speed_up` y `eficiencia` en todo el archivo.

```bash
# Por partes
python benchmark.py --c-values 12,1000000 --output resultados_suma.csv
python benchmark.py --c-values 100000000 --output resultados_suma.csv

# Rehacer filas existentes
python benchmark.py --rerun-existing --c-values 100000000 --output resultados_suma.csv

# Empezar de cero
python benchmark.py --overwrite --output resultados_suma.csv
```

## Corridas del enunciado

| Parámetro | Valores |
| --- | --- |
| `p` (workers) | 1, 2, 4, 8, 16 |
| `c` (complejidad) | 12, 1 000 000, 100 000 000 |

Métricas a calcular:

- **Speed-up** = T_secuencial / T_paralelo
- **Eficiencia** = speed-up / p

## Nota sobre memoria

Para valores grandes de `c` (ej. 100 000 000) sin agotar RAM, el vector se modela como una secuencia determinística generada por índice. Todas las variantes usan exactamente los mismos valores, garantizando consistencia en la comparación.
