# Ejercicio de suma paralela de un vector

Este directorio replica la estructura del ejercicio de `factorial`, adaptada al problema:

- sumar un vector de complejidad `c` en forma secuencial o paralela,
- comparar tiempos con distintas bibliotecas de concurrencia.

## Archivos

- `suma_lib.py`: funciones compartidas (vector deterministico, division en chunks, impresion de resultados).
- `secuential.py`: baseline secuencial.
- `suma_threadpoolexecutor.py`: paralelo con `ThreadPoolExecutor`.
- `suma_processpoolexecutor.py`: paralelo con `ProcessPoolExecutor`.
- `suma_multiprocessing.py`: paralelo con `multiprocessing.Pool`.
- `suma_threading.py`: paralelo con `threading` manual.

## Nota sobre memoria

Para permitir valores grandes de `c` (por ejemplo `100000000`) sin agotar RAM, el vector se modela como una secuencia deterministica generada por indice. Todas las variantes usan exactamente los mismos valores, por lo que la comparacion de tiempos y resultado es consistente.

## Como ejecutar

Desde la raiz del repo:

```bash
cd sumaParalela
python secuential.py --workers 1 --c 1000000
python suma_threadpoolexecutor.py --workers 4 --c 1000000
python suma_processpoolexecutor.py --workers 4 --c 1000000
python suma_multiprocessing.py --workers 4 --c 1000000
python suma_threading.py --workers 4 --c 1000000
```

## Corridas del enunciado

Ejecutar con:

- `p` en `1, 2, 4, 8, 16`
- `c` en `12, 1000000, 100000000`

y luego calcular en la tabla:

- `speed-up = T_secuencial / T_paralelo`
- `eficiencia = speed-up / p`

## Generar tabla CSV automatica

Incluye un script para correr todas las combinaciones y exportar una tabla lista para el informe:

```bash
python benchmark.py
```

Por defecto usa:

- `p = 1,2,4,8,16`
- `c = 12,1000000,100000000`
- salida: `resultados_suma.csv`

Opciones utiles:

```bash
python benchmark.py --output tabla_tp.csv
python benchmark.py --p-values 1,2,4 --c-values 12,1000000
```

### Carga incremental (recomendado)

`benchmark.py` ahora acumula resultados por defecto:

- si el CSV ya existe, **no repite** combinaciones ya calculadas;
- agrega solo lo faltante;
- recalcula `speed_up` y `eficiencia` en todo el archivo.

Ejemplo por partes:

```bash
# Parte 1: chico y medio
python benchmark.py --c-values 12,1000000 --output resultados_suma.csv

# Parte 2: solo el caso pesado
python benchmark.py --c-values 100000000 --output resultados_suma.csv
```

Opciones extra:

```bash
# Rehacer aunque ya exista (pisando esas filas)
python benchmark.py --rerun-existing --c-values 100000000 --output resultados_suma.csv

# Empezar de cero
python benchmark.py --overwrite --output resultados_suma.csv
```
