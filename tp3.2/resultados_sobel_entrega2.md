# Resultados benchmark - Sobel entrega 2

Fuente CSV: `resultados_sobel_entrega2.csv`.

Esta entrega mide solo el caso nuevo `numba_gpu`. Los resultados de `secuencial`, `numpy` y `numba_cpu` se toman de la entrega 1 para el informe combinado.

| Tamaño | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 750 | 0.001899755 | 0.000691792 | 0.002591548 | 0.281777778 | 0.001061602 | 0.000838153 | 0.000473858 | 0.000217934 | NVIDIA GeForce GTX 1650 |
| 1500 | 0.003172580 | 0.002474146 | 0.005646725 | 0.059955556 | 0.001809637 | 0.001362943 | 0.001944962 | 0.000529183 | NVIDIA GeForce GTX 1650 |
| 3000 | 0.010337396 | 0.007141225 | 0.017478621 | 0.001366667 | 0.004983035 | 0.005354361 | 0.005610899 | 0.001530326 | NVIDIA GeForce GTX 1650 |
| 6000 | 0.039567730 | 0.071842688 | 0.111410418 | 0.000000000 | 0.015295089 | 0.024272641 | 0.023390333 | 0.048452355 | NVIDIA GeForce GTX 1650 |

## Notas

- La carga y el guardado de imágenes quedan fuera de las mediciones.
- En GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.
- En GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.
