# Resultados benchmark - Sobel entrega 2

Fuente CSV: `numba_gpu_1500.csv`.

Esta entrega mide solo el caso nuevo `numba_gpu`. Los resultados de `secuencial`, `numpy` y `numba_cpu` se toman de la entrega 1 para el informe combinado.

| Tamaño | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1500 | 0.002681795 | 0.002221041 | 0.004902836 | 0.059955556 | 0.001496865 | 0.001184929 | 0.001580774 | 0.000640268 | NVIDIA GeForce GTX 1650 |

## Notas

- La carga y el guardado de imágenes quedan fuera de las mediciones.
- En GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.
- En GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.
