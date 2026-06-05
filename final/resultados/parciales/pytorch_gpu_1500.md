# Resultados benchmark - Sobel entrega 3

Fuente CSV: `pytorch_gpu_1500.csv`.

Esta entrega mide solo los casos nuevos `pytorch_cpu` y `pytorch_gpu`. Los resultados de las entregas anteriores se toman desde sus CSV para el informe combinado.

| Tamaño | Método | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | Dispositivo |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1500x1500 | pytorch_gpu | 0.004482590 | 0.002693912 | 0.007176502 | 0.059955556 | 0.002550422 | 0.001932168 | 0.002133532 | 0.000560380 | NVIDIA GeForce GTX 1650 |

## Notas

- La carga y el guardado de imágenes quedan fuera de las mediciones.
- En PyTorch CPU, los tensores se preparan fuera de la medición y se mide el cómputo sobre CPU.
- En PyTorch GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.
- En PyTorch GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.
