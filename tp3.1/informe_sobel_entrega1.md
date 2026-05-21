# Filtro de Sobel en Python: comparacion secuencial, NumPy y Numba CPU

**Materia:** Sistemas Paralelos - Lic. en Sistemas, 5to ano  
**Institucion:** UNTDF  
**Docente:** MsC. Federico Gonzalez Brizzio  
**Entrega:** 7 de mayo de 2026  
**Repositorio:** <https://github.com/GiulianoPoeta99/tps_paralelos.git>

---

## Abstract

Se implemento el filtro de Sobel para deteccion de bordes sobre imagenes RGB convertidas a escala de grises, comparando tres enfoques pedidos para la primera entrega: una version secuencial en Python puro, una version vectorizada con NumPy y una version compilada/paralela en CPU con Numba. El benchmark se ejecuto sobre imagenes de 750x750, 1500x1500, 3000x3000 y 6000x6000 pixeles, con 5 corridas por combinacion. Para cada metodo se midio por separado el tiempo de conversion RGB->gris, el tiempo de aplicacion de Sobel y el tiempo total, excluyendo carga y guardado de archivos. La salida se comparo mediante el porcentaje de pixeles blancos en la imagen Sobel. En todos los tamanios, los tres metodos produjeron el mismo porcentaje de blancos, lo que indica consistencia de salida para esta metrica. NumPy y Numba CPU redujeron fuertemente los tiempos respecto del secuencial; Numba CPU obtuvo los mejores tiempos totales en 1500x1500, 3000x3000 y 6000x6000, mientras que NumPy fue levemente mejor en 750x750.

---

## 1. Introduccion

El operador de Sobel estima cambios locales de intensidad mediante dos mascaras 3x3: una para el gradiente horizontal (`Gx`) y otra para el gradiente vertical (`Gy`). Para cada pixel interior se calcula la magnitud aproximada del gradiente como:

```text
|grad| = sqrt(gx^2 + gy^2)
```

El resultado se recorta al rango `[0, 255]` para obtener una imagen de bordes en escala de grises. Como el calculo se aplica pixel por pixel, el problema es adecuado para comparar el costo de una implementacion secuencial frente a implementaciones optimizadas o paralelas.

Esta entrega se limita a:

1. Version secuencial en Python puro.
2. Version vectorizada con NumPy.
3. Version Numba CPU paralela.

Las versiones Numba GPU, PyTorch CPU y PyTorch GPU corresponden a entregas posteriores.

---

## 2. Metodologia

### 2.1 Equipo

| Propiedad | Valor |
| --- | --- |
| CPU | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz |
| Nucleos fisicos | 4 |
| Nucleos logicos | 8 |
| RAM | 15 GiB |
| Sistema operativo | Manjaro Linux, kernel 6.12.85-1-MANJARO |
| Python | 3.14.4 |
| NumPy | 2.4.4 |
| Numba | 0.65.1 |
| Pillow | 12.2.0 |
| GPU | No usada en esta entrega |

**Nota:** se solicito Python 3.14.3, pero en el equipo estaba disponible Python 3.14.4 como version estable instalada. El entorno virtual usado fue `tp3/.venv`.

### 2.2 Algoritmos evaluados

1. **Secuencial:** conversion RGB->gris y Sobel con bucles de Python puro. No usa NumPy para el computo; usa `bytes`/`bytearray` como almacenamiento para evitar el costo de listas gigantes en 6000x6000.
2. **NumPy:** conversion y Sobel con operaciones vectorizadas sobre arreglos `ndarray`.
3. **Numba CPU:** kernels compilados con `@njit(parallel=True)` y `prange`, usando hasta 8 hilos logicos.

### 2.3 Parametros experimentales

| Parametro | Valor |
| --- | --- |
| Tamanios | 750x750, 1500x1500, 3000x3000, 6000x6000 |
| Corridas por caso | 5 |
| Metodos | `secuencial`, `numpy`, `numba_cpu` |
| Workers Numba CPU | 8 |
| Entrada | Imagenes provistas por la catedra (`IMG_0358_*`) |
| Script de corrida | `python correr_entrega1.py` |
| CSV | `resultados_sobel_entrega1.csv` |
| Tablas Markdown | `resultados_sobel_entrega1.md` |

La carga y el guardado de archivos no forman parte de los tiempos medidos. El cronometro solo cubre:

1. Conversion RGB->gris.
2. Aplicacion del filtro Sobel.

### 2.4 Metricas

- **Tiempo RGB->gris (s):** tiempo promedio de conversion de imagen RGB a escala de grises.
- **Tiempo Sobel (s):** tiempo promedio del filtro Sobel sobre la imagen gris.
- **Tiempo total (s):** suma medida de conversion y Sobel.
- **% blancos:** `(pixeles con valor 255 / pixeles totales) * 100`.
- **Speed-up:** `tiempo_total_secuencial / tiempo_total_metodo`, para el mismo tamanio.
- **Performance (%):** `speed-up / workers * 100`.

La columna `performance (%)` debe leerse como una metrica relativa definida por la consigna. Puede superar el 100% porque compara contra un baseline secuencial de Python, no contra un limite fisico de uso de CPU.

---

## 3. Resultados

Los valores siguientes provienen de `resultados_sobel_entrega1.csv`, con 5 corridas por combinacion.

### 3.1 Imagen 750x750

| Metodo | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up | Performance (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 0.137598206 | 0.381080508 | 0.518678714 | 0.281777778 | 1.000000 | 100.00 |
| numpy | 0.003584823 | 0.005273254 | 0.008858076 | 0.281777778 | 58.554328 | 5855.43 |
| numba_cpu | 0.008086321 | 0.001361815 | 0.009448136 | 0.281777778 | 54.897466 | 686.22 |

### 3.2 Imagen 1500x1500

| Metodo | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up | Performance (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 0.589171745 | 1.607831727 | 2.197003472 | 0.059955556 | 1.000000 | 100.00 |
| numpy | 0.013785413 | 0.025855008 | 0.039640422 | 0.059955556 | 55.423312 | 5542.33 |
| numba_cpu | 0.001911381 | 0.005410280 | 0.007321660 | 0.059955556 | 300.069022 | 3750.86 |

### 3.3 Imagen 3000x3000

| Metodo | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up | Performance (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 2.425503625 | 6.512637727 | 8.938141351 | 0.001366667 | 1.000000 | 100.00 |
| numpy | 0.060008272 | 0.120068416 | 0.180076688 | 0.001366667 | 49.635194 | 4963.52 |
| numba_cpu | 0.006325961 | 0.011174299 | 0.017500260 | 0.001366667 | 510.743340 | 6384.29 |

### 3.4 Imagen 6000x6000

| Metodo | Tiempo RGB->gris (s) | Tiempo Sobel (s) | Tiempo total (s) | % blancos | Speed-up | Performance (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| secuencial | 9.813729013 | 26.460540787 | 36.274269800 | 0.000000000 | 1.000000 | 100.00 |
| numpy | 1.742763535 | 2.402780660 | 4.145544195 | 0.000000000 | 8.750183 | 875.02 |
| numba_cpu | 0.027100780 | 0.029705952 | 0.056806732 | 0.000000000 | 638.555833 | 7981.95 |

## 4. Discusion

### 4.1 Diferencias de tiempo entre secuencial, NumPy y Numba CPU

El metodo secuencial fue el mas lento en todos los tamanios, como era esperable por ejecutar bucles interpretados en Python. El costo crece de forma clara al aumentar la resolucion: pasa de 0.519 s en 750x750 a 36.274 s en 6000x6000.

NumPy reduce fuertemente los tiempos frente al secuencial por vectorizacion. En 750x750 obtuvo el mejor tiempo total (0.00886 s), levemente por debajo de Numba CPU (0.00945 s). En los tamanios mayores, Numba CPU domina: 0.00732 s en 1500x1500, 0.01750 s en 3000x3000 y 0.05681 s en 6000x6000.

En el caso de 6000x6000, NumPy tambien mejora mucho al secuencial (4.146 s contra 36.274 s), pero Numba CPU queda muy por debajo (0.0568 s). Esto muestra que, para este nucleo puntual de Sobel, los bucles compilados y paralelos tienen una ventaja importante frente a la expresion vectorizada que crea arreglos intermedios.

### 4.2 Evolucion del speed-up

El speed-up de NumPy no crece monotonicamente. Fue 58.55x en 750x750, 55.42x en 1500x1500, 49.64x en 3000x3000 y baja a 8.75x en 6000x6000. La caida en el caso mas grande se atribuye al costo de memoria y a los arreglos temporales que genera la version vectorizada para `gx`, `gy` y la magnitud.

Numba CPU muestra un comportamiento mas favorable al aumentar la resolucion: 54.90x en 750x750, 300.07x en 1500x1500, 510.74x en 3000x3000 y 638.56x en 6000x6000. En imagenes chicas, el costo fijo de preparar/ejecutar kernels paralelos pesa mas; en imagenes grandes, ese costo se amortiza y predomina el computo paralelizado.

### 4.3 Equivalencia visual y porcentaje de blancos

Los tres metodos dieron exactamente el mismo porcentaje de pixeles blancos para cada tamanio:

| Tamanio | % blancos |
| --- | ---: |
| 750x750 | 0.281777778 |
| 1500x1500 | 0.059955556 |
| 3000x3000 | 0.001366667 |
| 6000x6000 | 0.000000000 |

Esto indica que las implementaciones son consistentes para la metrica pedida. En una primera version del codigo, Numba podia producir diferencias por desbordamiento al operar directamente con `uint8`; se corrigio casteando los pixeles a enteros con signo antes de calcular `gx` y `gy`.

El porcentaje de blancos cae con la resolucion porque la metrica solo cuenta pixeles exactamente iguales a 255. Aunque la imagen Sobel contenga bordes visibles en otros niveles de gris, esos pixeles no suman como "blancos" para esta consigna.

---

## 5. Conclusion

La implementacion secuencial sirve como baseline y confirma el costo alto de aplicar Sobel pixel por pixel en Python puro. NumPy mejora de forma marcada al vectorizar las operaciones y fue el metodo mas rapido en 750x750, donde el problema es chico y los costos fijos de Numba pesan mas. Para 1500x1500 en adelante, Numba CPU fue el metodo mas rapido, con speed-ups crecientes hasta 638.56x en 6000x6000.

Las tres implementaciones fueron equivalentes en la metrica de salida exigida: el porcentaje de pixeles blancos coincide en todos los tamanios. Por lo tanto, las diferencias observadas son de rendimiento y no de resultado numerico para esta comparacion.

---

## Referencias

- *Trabajo Practico - Filtro de Sobel para deteccion de bordes*, Sistemas Paralelos, UNTDF, 2026.
- NumPy Documentation: arreglos `ndarray` y operaciones vectorizadas.
- Numba Documentation: `njit`, `prange`, `set_num_threads`.
- Python Software Foundation. Documentacion de Python 3.14.
