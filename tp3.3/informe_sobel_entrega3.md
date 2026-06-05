# Filtro de Sobel en CPU y GPU: Secuencial, NumPy, Numba y PyTorch

**Materia:** Sistemas Paralelos - Lic. en Sistemas, 5to año  
**Institución:** UNTDF  
**Docente:** MsC. Federico González Brizzio  
**Entrega:** 28 de mayo de 2026  
**Repositorio:** <https://github.com/GiulianoPoeta99/tps_paralelos.git>

---

## Abstract

Este informe integra la evolución completa del trabajo: una versión secuencial, una versión vectorizada con NumPy, una versión paralela en CPU con Numba, una versión GPU con Numba CUDA y dos variantes con PyTorch sobre CPU y GPU. Todas las implementaciones aplican el mismo flujo de procesamiento: conversión RGB->gris por luminancia y filtro Sobel 3x3 usando la magnitud del gradiente. La comparación se organiza por tamaño de imagen y utiliza el tiempo total promedio como referencia para calcular speed-up y mejora porcentual.

---

## 1. Introducción

El operador Sobel calcula un gradiente local para cada píxel usando una vecindad 3x3. El objetivo del trabajo es analizar cómo cambia el rendimiento cuando el mismo algoritmo se expresa con distintos enfoques: Python secuencial, operaciones vectorizadas, compilación/paralelismo en CPU, kernels CUDA y tensores PyTorch sobre CPU/GPU.

---

## 2. Metodología

### 2.1 Equipo

| Propiedad | Valor |
| --- | --- |
| CPU | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz |
| Núcleos físicos | 4 |
| Procesadores lógicos (hilos) | 8 |
| Threads por núcleo | 2 |
| GPU | NVIDIA GeForce GTX 1650 |
| Multiprocesadores CUDA reportados | 14 |
| Python | 3.14.5 |
| NumPy | 2.4.4 |
| Numba | 0.65.1 |
| PyTorch | 2.12.0+cu130 |
| CUDA PyTorch | 13.0 |
| Pillow | 12.2.0 |

### 2.2 Algoritmos incluidos

1. **Secuencial:** implementación base sin librerías vectorizadas.
2. **NumPy:** implementación vectorizada sobre arreglos.
3. **Numba CPU:** implementación compilada y paralela sobre CPU.
4. **Numba GPU:** implementación con kernels CUDA.
5. **PyTorch CPU:** implementación con tensores sobre CPU.
6. **PyTorch GPU:** implementación con tensores sobre GPU.

### 2.3 Parámetros experimentales

| Parámetro | Valor |
| --- | --- |
| Tamaños | 750x750, 1500x1500, 3000x3000, 6000x6000 |
| Corridas por caso | 5 |
| Fuente Secuencial/NumPy/Numba CPU | `/mnt/sda1/code/facu/paralelos/tp3.1/resultados_sobel_entrega1.csv` |
| Fuente Numba GPU | `/mnt/sda1/code/facu/paralelos/tp3.2/resultados_sobel_entrega2.csv` |
| Fuente PyTorch CPU/GPU | `/mnt/sda1/code/facu/paralelos/tp3.3/resultados_sobel_entrega3.csv` |
| Script de corrida PyTorch CPU | `python benchmark_entrega3.py --methods pytorch_cpu --runs 5 --save-preview` |
| Script de corrida PyTorch GPU | `python benchmark_entrega3.py --methods pytorch_gpu --runs 5 --save-preview` |

La carga de imágenes y el guardado de salidas no forman parte de las mediciones.

### 2.4 Métricas

- **Tiempo RGB->gris (s):** tiempo promedio de conversión.
- **Tiempo Sobel (s):** tiempo promedio del filtro Sobel.
- **Tiempo total (s):** suma medida de conversión y Sobel.
- **% blancos:** `(píxeles con valor 255 / píxeles totales) * 100`.
- **Speed-up vs secuencial:** `tiempo_total_secuencial / tiempo_total_metodo`.
- **Mejora vs secuencial (%):** `(1 - tiempo_total_metodo / tiempo_total_secuencial) * 100`.

---

## 3. Comparación combinada

Las tablas siguientes integran todas las implementaciones medidas. La mejora porcentual se calcula siempre respecto del secuencial del mismo tamaño.

### 3.1 Imagen 750x750

| Método | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up vs secuencial | Mejora vs secuencial (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 0.137598206 | 0.381080508 | 0.518678714 | 0.281777778 | 1.000000 | 0.00 |
| numpy | 0.003584823 | 0.005273254 | 0.008858076 | 0.281777778 | 58.554331 | 98.29 |
| numba_cpu | 0.008086321 | 0.001361815 | 0.009448136 | 0.281777778 | 54.897465 | 98.18 |
| numba_gpu | 0.001899755 | 0.000691792 | 0.002591548 | 0.281777778 | 200.142430 | 99.50 |
| pytorch_cpu | 0.023946776 | 0.018791604 | 0.042738380 | 0.281777778 | 12.136134 | 91.76 |
| pytorch_gpu | 0.015180147 | 0.018391323 | 0.033571470 | 0.281777778 | 15.449985 | 93.53 |

### 3.2 Imagen 1500x1500

| Método | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up vs secuencial | Mejora vs secuencial (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 0.589171745 | 1.607831727 | 2.197003472 | 0.059955556 | 1.000000 | 0.00 |
| numpy | 0.013785413 | 0.025855008 | 0.039640422 | 0.059955556 | 55.423312 | 98.20 |
| numba_cpu | 0.001911381 | 0.005410280 | 0.007321660 | 0.059955556 | 300.069038 | 99.67 |
| numba_gpu | 0.003172580 | 0.002474146 | 0.005646725 | 0.059955556 | 389.075698 | 99.74 |
| pytorch_cpu | 0.028409258 | 0.084831254 | 0.113240513 | 0.059955556 | 19.401214 | 94.85 |
| pytorch_gpu | 0.031052971 | 0.017936814 | 0.048989785 | 0.059955556 | 44.846155 | 97.77 |

### 3.3 Imagen 3000x3000

| Método | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up vs secuencial | Mejora vs secuencial (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 2.425503625 | 6.512637727 | 8.938141351 | 0.001366667 | 1.000000 | 0.00 |
| numpy | 0.060008272 | 0.120068416 | 0.180076688 | 0.001366667 | 49.635194 | 97.99 |
| numba_cpu | 0.006325961 | 0.011174299 | 0.017500260 | 0.001366667 | 510.743346 | 99.80 |
| numba_gpu | 0.010337396 | 0.007141225 | 0.017478621 | 0.001366667 | 511.375660 | 99.80 |
| pytorch_cpu | 0.080905324 | 0.227373985 | 0.308279309 | 0.001366667 | 28.993647 | 96.55 |
| pytorch_gpu | 0.015456200 | 0.008582767 | 0.024038967 | 0.001366667 | 371.818862 | 99.73 |

### 3.4 Imagen 6000x6000

| Método | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up vs secuencial | Mejora vs secuencial (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 9.813729013 | 26.460540787 | 36.274269800 | 0.000000000 | 1.000000 | 0.00 |
| numpy | 1.742763535 | 2.402780660 | 4.145544195 | 0.000000000 | 8.750183 | 88.57 |
| numba_cpu | 0.027100780 | 0.029705952 | 0.056806732 | 0.000000000 | 638.555828 | 99.84 |
| numba_gpu | 0.039567730 | 0.071842688 | 0.111410418 | 0.000000000 | 325.591363 | 99.69 |
| pytorch_cpu | 0.285784754 | 0.756089119 | 1.041873873 | 0.000000000 | 34.816373 | 97.13 |
| pytorch_gpu | 0.105016384 | 0.038902479 | 0.143918863 | 0.000000000 | 252.046667 | 99.60 |

---

## 4. Discusión

### 4.1 Mejora porcentual de PyTorch contra secuencial

La mejora de `pytorch_cpu` respecto del secuencial, usando tiempo total, fue: 91.76% en 750x750 (12.14x), 94.85% en 1500x1500 (19.40x), 96.55% en 3000x3000 (28.99x), 97.13% en 6000x6000 (34.82x).

La mejora de `pytorch_gpu` respecto del secuencial, usando tiempo total, fue: 93.53% en 750x750 (15.45x), 97.77% en 1500x1500 (44.85x), 99.73% en 3000x3000 (371.82x), 99.60% en 6000x6000 (252.05x).

Los dos casos PyTorch mejoran claramente al secuencial puro. La versión GPU supera a PyTorch CPU en todos los tamaños medidos, aunque no alcanza a las mejores implementaciones basadas en Numba CPU/GPU.

### 4.2 Comparación PyTorch CPU vs PyTorch GPU

La siguiente tabla compara directamente PyTorch CPU contra PyTorch GPU. `pytorch_gpu` fue más rápido que `pytorch_cpu` en todos los tamaños medidos, considerando el tiempo total con transferencias.

| Tamaño | PyTorch CPU total (s) | PyTorch GPU total (s) | Speed-up GPU vs CPU | Estado GPU |
| --- | ---: | ---: | ---: | --- |
| 750x750 | 0.042738380 | 0.033571470 | 1.273057 | medido |
| 1500x1500 | 0.113240513 | 0.048989785 | 2.311513 | medido |
| 3000x3000 | 0.308279309 | 0.024038967 | 12.824150 | medido |
| 6000x6000 | 1.041873873 | 0.143918863 | 7.239314 | medido |

### 4.3 Comparación contra las mejores implementaciones Numba/NumPy

En 750x750, `pytorch_cpu` tarda 0.04274 s y `pytorch_gpu` tarda 0.03357 s, ambos por encima de `numba_gpu` (0.00259 s). En 1500x1500, `pytorch_gpu` mejora a `pytorch_cpu`, pero sigue por detrás de `numba_gpu` (0.00565 s). En 3000x3000, `pytorch_gpu` queda cerca de `numba_gpu`, aunque todavía por encima de su tiempo total (0.01748 s). En 6000x6000, `pytorch_gpu` alcanza 0.14392 s, mejor que PyTorch CPU y NumPy, pero por detrás de `numba_cpu` y `numba_gpu`.

### 4.4 Transferencias en PyTorch GPU

En PyTorch GPU, el tiempo total incluye transferencia CPU->GPU, kernels y transferencia GPU->CPU. El desglose fue:

| Tamaño | H2D (s) | Kernel RGB->gris (s) | Kernel Sobel (s) | D2H (s) | Total GPU (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 750x750 | 0.001367908 | 0.013812239 | 0.011459848 | 0.006931475 | 0.033571470 |
| 1500x1500 | 0.003916377 | 0.027136593 | 0.017088616 | 0.000848198 | 0.048989785 |
| 3000x3000 | 0.008904967 | 0.006551233 | 0.006901338 | 0.001681429 | 0.024038967 |
| 6000x6000 | 0.040180978 | 0.064835406 | 0.023611030 | 0.015291449 | 0.143918863 |

### 4.5 Consistencia de salida

| Tamaño | Secuencial % blancos | NumPy % blancos | Numba CPU % blancos | Numba GPU % blancos | PyTorch CPU % blancos | PyTorch GPU % blancos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 750x750 | 0.281777778 | 0.281777778 | 0.281777778 | 0.281777778 | 0.281777778 | 0.281777778 |
| 1500x1500 | 0.059955556 | 0.059955556 | 0.059955556 | 0.059955556 | 0.059955556 | 0.059955556 |
| 3000x3000 | 0.001366667 | 0.001366667 | 0.001366667 | 0.001366667 | 0.001366667 | 0.001366667 |
| 6000x6000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |

Los porcentajes coinciden entre los métodos medidos para cada tamaño, por lo que la comparación es de rendimiento y no de diferencia en la métrica de salida.

### 4.6 Síntesis del análisis

**1. ¿Qué diferencias de rendimiento se observan entre PyTorch CPU y PyTorch GPU para cada tamaño de imagen?**

PyTorch GPU fue más rápido que PyTorch CPU en todos los tamaños medidos. La diferencia es moderada en 750x750, con 1.27x de speed-up, y crece en tamaños mayores: 2.31x en 1500x1500, 12.82x en 3000x3000 y 7.24x en 6000x6000. Esto muestra que el uso de GPU se justifica mejor cuando hay más trabajo por corrida, aunque el tiempo total sigue incluyendo las transferencias entre CPU y GPU.

**2. ¿Cómo se comparan PyTorch CPU/PyTorch GPU frente a las mejores implementaciones Numba/NumPy?**

PyTorch CPU no supera a las mejores implementaciones Numba/NumPy en ningún tamaño. PyTorch GPU mejora claramente a PyTorch CPU, pero tampoco supera al mejor resultado global. En 750x750, 1500x1500 y 3000x3000 el mejor método fue `numba_gpu`; PyTorch GPU quedó aproximadamente 12.95x, 8.68x y 1.38x más lento respectivamente. En 6000x6000 el mejor método fue `numba_cpu`, y PyTorch GPU quedó aproximadamente 2.53x más lento. Aun así, PyTorch GPU queda cerca de Numba GPU en 3000x3000 y mejora ampliamente a NumPy y PyTorch CPU en los tamaños grandes.

**3. ¿Las salidas de PyTorch (CPU/GPU) son consistentes con Numba/NumPy en términos de bordes detectados y porcentaje de píxeles blancos, y qué factores podrían explicar diferencias?**

Sí. En los cuatro tamaños, PyTorch CPU y PyTorch GPU dan el mismo porcentaje de píxeles blancos que Secuencial, NumPy, Numba CPU y Numba GPU. Esto indica que las salidas son consistentes para la métrica usada. Si aparecieran diferencias, podrían deberse a cambios en el redondeo al convertir de `float` a `uint8`, diferencias en el manejo de bordes, uso de otra fórmula de magnitud del gradiente, signos distintos en las máscaras Sobel, o diferencias de precisión entre operaciones CPU y GPU.

---

## 5. Conclusión

El informe final integra las seis implementaciones desarrolladas para el filtro Sobel. Todas mantienen la misma salida según el porcentaje de blancos, por lo que la comparación se centra en rendimiento. PyTorch GPU supera a PyTorch CPU en todos los tamaños, especialmente en 3000x3000 y 6000x6000, pero las mejores variantes globales siguen siendo las implementaciones basadas en Numba CPU/GPU según el tamaño de imagen.

---

## Referencias

- *Trabajo Práctico - Filtro de Sobel para detección de bordes*, Sistemas Paralelos, UNTDF, 2026.
- *Introducción a la programación paralela*, material de cátedra, capítulos 7 y 8.
- PyTorch Documentation: Tensors and `torch.nn.functional.conv2d`.
- Numba Documentation: CUDA kernels.
- NVIDIA CUDA Programming Guide.
