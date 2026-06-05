# Resultados benchmark - Sobel entrega 2

Fuente CSV: `numba_gpu_6000.csv`.

Esta entrega mide solo el caso nuevo `numba_gpu`. Los resultados de `secuencial`, `numpy` y `numba_cpu` se toman de la entrega 1 para el informe combinado.

| Tamaño | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 6000 | 0.022455198 | 0.026138397 | 0.048593595 | 0.000000000 | 0.010809212 | 0.011645986 | 0.019370614 | 0.006767782 | NVIDIA GeForce GTX 1650 |

## Notas

- La carga y el guardado de imágenes quedan fuera de las mediciones.
- En GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.
- En GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.
