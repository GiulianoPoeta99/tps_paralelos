# Resultados benchmark - Sobel entrega 3

Fuente CSV: `pytorch_gpu_3000.csv`.

Esta entrega mide solo los casos nuevos `pytorch_cpu` y `pytorch_gpu`. Los resultados de las entregas anteriores se toman desde sus CSV para el informe combinado.

| Tamaño | Método | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | Dispositivo |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3000x3000 | pytorch_gpu | 0.016800519 | 0.009445662 | 0.026246180 | 0.001366667 | 0.009720037 | 0.007080482 | 0.007885758 | 0.001559904 | NVIDIA GeForce GTX 1650 |

## Notas

- La carga y el guardado de imágenes quedan fuera de las mediciones.
- En PyTorch CPU, los tensores se preparan fuera de la medición y se mide el cómputo sobre CPU.
- En PyTorch GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.
- En PyTorch GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.
