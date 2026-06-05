# Resultados benchmark - Sobel entrega 3

Fuente CSV: `pytorch_gpu_750.csv`.

Esta entrega mide solo los casos nuevos `pytorch_cpu` y `pytorch_gpu`. Los resultados de las entregas anteriores se toman desde sus CSV para el informe combinado.

| Tamaño | Método | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | Dispositivo |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 750x750 | pytorch_gpu | 0.010117973 | 0.006472537 | 0.016590509 | 0.281777778 | 0.000849445 | 0.009268528 | 0.006126249 | 0.000346288 | NVIDIA GeForce GTX 1650 |

## Notas

- La carga y el guardado de imágenes quedan fuera de las mediciones.
- En PyTorch CPU, los tensores se preparan fuera de la medición y se mide el cómputo sobre CPU.
- En PyTorch GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.
- En PyTorch GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.
