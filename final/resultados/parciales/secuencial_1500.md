# Resultados benchmark - filtro de Sobel

Fuente CSV: `secuencial_1500.csv`.

Cada fila usa promedios sobre la cantidad de corridas indicada en la columna `corridas` del CSV.
El speed-up se calcula respecto del tiempo total secuencial del mismo tamanio de imagen.

## Imagen 1500x1500

| Metodo | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up | Performance (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 0.538026730 | 1.499926087 | 2.037952817 | 0.059955556 | 1.000000 | 100.00 |

## Notas

- El tiempo total se mide como conversion `RGB->gris` mas aplicacion de Sobel.
- La carga y el guardado de imagenes quedan fuera de las mediciones.
- `performance (%) = speed-up / workers * 100`; los `workers` estan en el CSV.
