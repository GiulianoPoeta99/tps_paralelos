# Resultados benchmark - filtro de Sobel

Fuente CSV: `numba_cpu_750.csv`.

Cada fila usa promedios sobre la cantidad de corridas indicada en la columna `corridas` del CSV.
El speed-up se calcula respecto del tiempo total secuencial del mismo tamanio de imagen.

## Imagen 750x750

| Metodo | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up | Performance (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| numba_cpu | 0.010843878 | 0.000573319 | 0.011417198 | 0.281777778 | 1.000000 | 25.00 |

## Notas

- El tiempo total se mide como conversion `RGB->gris` mas aplicacion de Sobel.
- La carga y el guardado de imagenes quedan fuera de las mediciones.
- `performance (%) = speed-up / workers * 100`; los `workers` estan en el CSV.
