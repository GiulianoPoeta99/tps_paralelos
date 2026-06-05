# Resultados benchmark - Sobel entrega 2

Fuente CSV: `numba_gpu_750.csv`.

Esta entrega mide solo el caso nuevo `numba_gpu`. Los resultados de `secuencial`, `numpy` y `numba_cpu` se toman de la entrega 1 para el informe combinado.

| Tamaño | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 750 | 0.001167969 | 0.000624656 | 0.001792625 | 0.281777778 | 0.000662192 | 0.000505777 | 0.000450615 | 0.000174041 | NVIDIA GeForce GTX 1650 |

## Notas

- La carga y el guardado de imágenes quedan fuera de las mediciones.
- En GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.
- En GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.
