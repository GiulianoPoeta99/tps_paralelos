# Resultados benchmark - Sobel entrega 3

Fuente CSV: `pytorch_cpu_1500.csv`.

Esta entrega mide solo los casos nuevos `pytorch_cpu` y `pytorch_gpu`. Los resultados de las entregas anteriores se toman desde sus CSV para el informe combinado.

| Tamaño | Método | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | Dispositivo |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1500x1500 | pytorch_cpu | 0.021680436 | 0.037033428 | 0.058713864 | 0.059955556 | 0.000000000 | 0.021680436 | 0.037033428 | 0.000000000 | CPU |

## Notas

- La carga y el guardado de imágenes quedan fuera de las mediciones.
- En PyTorch CPU, los tensores se preparan fuera de la medición y se mide el cómputo sobre CPU.
- En PyTorch GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.
- En PyTorch GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.
