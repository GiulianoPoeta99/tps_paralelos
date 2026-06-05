# Resultados benchmark - Sobel entrega 2

Fuente CSV: `numba_gpu_3000.csv`.

Esta entrega mide solo el caso nuevo `numba_gpu`. Los resultados de `secuencial`, `numpy` y `numba_cpu` se toman de la entrega 1 para el informe combinado.

| Tamaño | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3000 | 0.007701786 | 0.007390242 | 0.015092028 | 0.001366667 | 0.004192532 | 0.003509254 | 0.005540778 | 0.001849464 | NVIDIA GeForce GTX 1650 |

## Notas

- La carga y el guardado de imágenes quedan fuera de las mediciones.
- En GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.
- En GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.
