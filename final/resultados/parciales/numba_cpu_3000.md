# Resultados benchmark - filtro de Sobel

Fuente CSV: `numba_cpu_3000.csv`.

Cada fila usa promedios sobre la cantidad de corridas indicada en la columna `corridas` del CSV.
El speed-up se calcula respecto del tiempo total secuencial del mismo tamanio de imagen.

## Imagen 3000x3000

| Metodo | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up | Performance (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| numba_cpu | 0.008683773 | 0.011001686 | 0.019685458 | 0.001366667 | 1.000000 | 25.00 |

## Notas

- El tiempo total se mide como conversion `RGB->gris` mas aplicacion de Sobel.
- La carga y el guardado de imagenes quedan fuera de las mediciones.
- `performance (%) = speed-up / workers * 100`; los `workers` estan en el CSV.
