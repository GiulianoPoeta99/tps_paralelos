# Resultados benchmark - Sobel entrega 3

Fuente CSV: `resultados_sobel_entrega3.csv`.

Esta entrega mide solo los casos nuevos `pytorch_cpu` y `pytorch_gpu`. Los resultados de las entregas anteriores se toman desde sus CSV para el informe combinado.

| Tamaño | Método | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | Dispositivo |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 750x750 | pytorch_cpu | 0.023946776 | 0.018791604 | 0.042738380 | 0.281777778 | 0.000000000 | 0.023946776 | 0.018791604 | 0.000000000 | CPU |
| 750x750 | pytorch_gpu | 0.015180147 | 0.018391323 | 0.033571470 | 0.281777778 | 0.001367908 | 0.013812239 | 0.011459848 | 0.006931475 | NVIDIA GeForce GTX 1650 |
| 1500x1500 | pytorch_cpu | 0.028409258 | 0.084831254 | 0.113240513 | 0.059955556 | 0.000000000 | 0.028409258 | 0.084831254 | 0.000000000 | CPU |
| 1500x1500 | pytorch_gpu | 0.031052971 | 0.017936814 | 0.048989785 | 0.059955556 | 0.003916377 | 0.027136593 | 0.017088616 | 0.000848198 | NVIDIA GeForce GTX 1650 |
| 3000x3000 | pytorch_cpu | 0.080905324 | 0.227373985 | 0.308279309 | 0.001366667 | 0.000000000 | 0.080905324 | 0.227373985 | 0.000000000 | CPU |
| 3000x3000 | pytorch_gpu | 0.015456200 | 0.008582767 | 0.024038967 | 0.001366667 | 0.008904967 | 0.006551233 | 0.006901338 | 0.001681429 | NVIDIA GeForce GTX 1650 |
| 6000x6000 | pytorch_cpu | 0.285784754 | 0.756089119 | 1.041873873 | 0.000000000 | 0.000000000 | 0.285784754 | 0.756089119 | 0.000000000 | CPU |
| 6000x6000 | pytorch_gpu | 0.105016384 | 0.038902479 | 0.143918863 | 0.000000000 | 0.040180978 | 0.064835406 | 0.023611030 | 0.015291449 | NVIDIA GeForce GTX 1650 |

## Notas

- La carga y el guardado de imágenes quedan fuera de las mediciones.
- En PyTorch CPU, los tensores se preparan fuera de la medición y se mide el cómputo sobre CPU.
- En PyTorch GPU, el tiempo RGB->gris incluye transferencia CPU->GPU y kernel de conversión.
- En PyTorch GPU, el tiempo Sobel incluye kernel Sobel y transferencia GPU->CPU del resultado.
