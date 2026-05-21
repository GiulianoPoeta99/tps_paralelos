# Resultados benchmark - filtro de Sobel

Fuente CSV: `resultados_sobel_entrega1.csv`.

Cada fila usa promedios sobre la cantidad de corridas indicada en la columna `corridas` del CSV.
El speed-up se calcula respecto del tiempo total secuencial del mismo tamanio de imagen.

## Imagen 750x750

| Metodo | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up | Performance (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 0.137598206 | 0.381080508 | 0.518678714 | 0.281777778 | 1.000000 | 100.00 |
| numpy | 0.003584823 | 0.005273254 | 0.008858076 | 0.281777778 | 58.554328 | 5855.43 |
| numba_cpu | 0.008086321 | 0.001361815 | 0.009448136 | 0.281777778 | 54.897466 | 686.22 |

## Imagen 1500x1500

| Metodo | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up | Performance (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 0.589171745 | 1.607831727 | 2.197003472 | 0.059955556 | 1.000000 | 100.00 |
| numpy | 0.013785413 | 0.025855008 | 0.039640422 | 0.059955556 | 55.423312 | 5542.33 |
| numba_cpu | 0.001911381 | 0.005410280 | 0.007321660 | 0.059955556 | 300.069022 | 3750.86 |

## Imagen 3000x3000

| Metodo | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up | Performance (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 2.425503625 | 6.512637727 | 8.938141351 | 0.001366667 | 1.000000 | 100.00 |
| numpy | 0.060008272 | 0.120068416 | 0.180076688 | 0.001366667 | 49.635194 | 4963.52 |
| numba_cpu | 0.006325961 | 0.011174299 | 0.017500260 | 0.001366667 | 510.743340 | 6384.29 |

## Imagen 6000x6000

| Metodo | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up | Performance (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 9.813729013 | 26.460540787 | 36.274269800 | 0.000000000 | 1.000000 | 100.00 |
| numpy | 1.742763535 | 2.402780660 | 4.145544195 | 0.000000000 | 8.750183 | 875.02 |
| numba_cpu | 0.027100780 | 0.029705952 | 0.056806732 | 0.000000000 | 638.555833 | 7981.95 |

## Notas

- El tiempo total se mide como conversion `RGB->gris` mas aplicacion de Sobel.
- La carga y el guardado de imagenes quedan fuera de las mediciones.
- `performance (%) = speed-up / workers * 100`; los `workers` estan en el CSV.
