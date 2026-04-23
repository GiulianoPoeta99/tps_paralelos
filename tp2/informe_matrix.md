# Multiplicación de matrices en Python: benchmarks de paralelismo, transpuesta y Numba

**Sistemas Paralelos — UNTDF**  
**Fecha de entrega (consigna):** 23 de abril de 2026  

> Renombrar este archivo a `apellido_nombre_matrix.md` (o `.pdf`) antes de enviar por correo, según formato de entrega.

---

## Resumen

Se midió el tiempo de cómputo de la multiplicación \(C = A \cdot B\) con matrices de enteros acotados, perfil cúbico (`A` y `B` de tamaño \(c \times c\)), semilla fija `--seed 2026` y cota `--max-val 256`. Se compararon implementaciones secuencial (método transpuesto y triple bucle tradicional), paralelismo con `threading`, `ThreadPoolExecutor` y `ProcessPoolExecutor`, y aceleración con `numba.njit` (serial y paralelo con `prange`). En todos los casos el checksum de suma de elementos de \(A \cdot B\) coincidió entre métodos y con la verificación \((AB)^T = B^T A^T\). Los resultados muestran que el acceso por filas usando \(B^T\) mejora fuerte respecto del bucle clásico en el secuencial; el paralelismo por filas escala de forma sublineal frente al secuencial transpuesto; `threading` y el pool de hilos se comportan de forma parecida en esta carga (CPU-bound en Python puro); `numba` domina por compilación a código nativo incluso sin paralelismo explícito. No se utilizaron NumPy ni PyTorch en la implementación propia del trabajo, según criterio docente.

---

## 1. Introducción

La multiplicación de matrices es un núcleo computacional intensivo en álgebra lineal y en muchas aplicaciones. En Python, el coste de interpretación y el modelo de memoria hacen relevante tanto el **orden de los bucles** (p. ej. uso de la transpuesta de \(B\) para mejorar localidad) como el tipo de **paralelismo** (hilos vs procesos vs JIT). El trabajo reproduce un entorno de benchmarking controlado (misma semilla, mismas dimensiones) y cuantifica tiempos, *speed-up* respecto de un secuencial de referencia y una métrica de **eficiencia** derivada de la grilla de *workers*.

**Contribuciones del informe:** (1) tablas de resultados para las combinaciones ejecutadas; (2) respuestas breves a las preguntas de análisis de la consigna; (3) descripción del entorno de hardware y software en el que se corrieron los experimentos.

---

## 2. Trabajo relacionado (breve)

El curso se basó en estrategias clásicas: paralelismo de tareas con la biblioteca estándar (`threading`, `concurrent.futures`) y atenuación del coste de Python puro vía `numba.njit` (código *nopython* y, opcionalmente, `prange`). La consigna original citaba NumPy y PyTorch; en la práctica implementada se respetó el criterio de no usar esas dependencias en el código de los alumnos, alineado con el dictado y la defensa oral.

---

## 3. Metodología

### 3.1 Definición del problema

Dadas \(A \in \mathbb{Z}^{m \times n}\) y \(B \in \mathbb{Z}^{n \times p}\), se computa \(C = AB\) y la suma de todos los elementos de \(C\) como *checksum* reproducible. Se valida además el cálculo vía \((AB)^T = B^T A^T\) (misma suma *trace* de elementos en el sentido de la implementación). Perfil usado: **cubo** (`--perfil cubo`), es decir \(m = n = p = c\).

### 3.2 Métricas

- **Tiempo (s):** fase `A·B` (y, cuando aplica, fase `B^T A^T` en los scripts), según el benchmark.
- **Speed-up y eficiencia:** calculados por el script `benchmark.py` respecto del tiempo del secuencial **tradicional** como referencia base en la exportación CSV (convención del proyecto; ver columna `speed_up_ab` / `eficiencia_ab`).
- **Checksum:** idéntico entre métodos para la misma `c` (obligación de la consigna).

### 3.3 Configuración experimental

| Parámetro | Valor |
| --- | --- |
| Semilla | `2026` |
| Cota de enteros | `max_val = 256` |
| Perfil | `cubo` |
| *Workers* (paralelos) | 4 y, donde aplica, 8 (véase nota) |
| Código Python | 3.14.3, build *free-threaded* (GIL desactivado) |

**Nota sobre *workers* 10 y Numba:** en esta máquina, `numba.set_num_threads(n)` acepta como máximo el número de hilos admitidos por el runtime (acotado, en el equipo de medición, a **8** hilos lógicos). Por tanto las corridas con `numba` paralelo usan **4 u 8** hilos, no 10. En una CPU con ocho hilos lógicos, fijar diez hilos excede el límite del backend de Numba; se documenta el valor **efectivo** en la tabla. El resto de métodos (`ThreadPoolExecutor`, `threading`, `ProcessPoolExecutor`) se ejecutó con la misma grilla 4/8 alineada a la disponibilidad del equipo. Las tablas reflejen los **datos reales** del CSV agregado al repositorio.

### 3.4 Entorno y hardware (medido en el equipo de ejecución)

| Ítem | Detalle |
| --- | --- |
| CPU | Intel(R) Core(TM) i5-10300H @ 2.50GHz (hasta ~4.5 GHz) |
| Núcleos físicos | 4 |
| Hilos lógicos | 8 (Hyper-Threading) |
| Memoria RAM | 15 GiB totales (orden de magnitud; disponible variable según carga) |
| Sistema operativo | Manjaro Linux, kernel 6.12.77 (x86_64) |
| Python | 3.14.3 (intérprete *free-threaded* / GIL deshabilitado en este build) |
| GPU / PyTorch | No se reportan resultados de PyTorch: no se usó en el código del TP y no fue requerido con GPU según criterio actualizado. |

---

## 4. Experimentos y resultados

Se utilizaron los archivos `resultados_v2_c512.csv` y `resultados_v2_c1024.csv` generados con `benchmark.py` en el directorio del TP2. A continuación se resume el tiempo de la fase **A·B** (`tiempo_segundos_ab`); el *checksum* fue **-110463704** para \(c=512\) y **34818311** para \(c=1024\), con coincidencia `checksum_ab = checksum_btat` en todas las filas.

### 4.1 \(c = 512\)

Tiempos aproximados (s) para **A·B**, por método y *workers*:

| Método | *workers* | Tiempo A·B (s) | Speed-up A·B (vs. sec. trad.) | Eficiencia A·B (↔ CSV) |
| --- | ---: | ---: | ---: | ---: |
| Secuencial (transpuesta) | 1 | 11,66 | 1,15 | 1,15 |
| Secuencial (tradicional) | 1 | 13,42 | 1,00 | 1,00 |
| ThreadPoolExecutor | 1 | 11,53 | 1,16 | 1,16 |
| ThreadPoolExecutor | 4 | 3,63 | 3,70 | 0,92 |
| ThreadPoolExecutor | 8 | 3,37 | 3,98 | 0,50 |
| threading | 1 | 11,43 | 1,17 | 1,17 |
| threading | 4 | 3,56 | 3,77 | 0,94 |
| threading | 8 | 3,26 | 4,12 | 0,52 |
| ProcessPoolExecutor | 1 | 10,07 | 1,33 | 1,33 |
| ProcessPoolExecutor | 4 | 3,18 | 4,22 | 1,05 |
| ProcessPoolExecutor | 8 | 3,07 | 4,37 | 0,55 |
| numba (njit) | 1 (sin `parallel`) | 1,16 | 11,55 | 11,55 |
| numba (njit) | 4 (`parallel`) | 1,11 | 12,15 | 3,04 |
| numba (njit) | 8 (`parallel`) | 1,04 | 12,95 | 1,62 |

*Performance (%)*: puede reportarse como **eficiencia × 100** respecto de la asignación de *workers* (p. ej. ~92% para 4 hilos con ThreadPool si la eficiencia del CSV es ~0,92), con la salvedad de que para Numba con *baseline* no paralelo el valor supera 100% por construcción de la métrica frente al secuencial lento puro (véase sección 5).

### 4.2 \(c = 1024\)

| Método | *workers* | Tiempo A·B (s) | Speed-up A·B (vs. sec. trad.) | Eficiencia A·B (↔ CSV) |
| --- | ---: | ---: | ---: | ---: |
| Secuencial (transpuesta) | 1 | 85,87 | 1,82 | 1,82 |
| Secuencial (tradicional) | 1 | 155,93 | 1,00 | 1,00 |
| ThreadPoolExecutor | 4 | 29,69 | 5,25 | 1,31 |
| ThreadPoolExecutor | 8 | 27,56 | 5,66 | 0,71 |
| threading | 4 | 30,87 | 5,05 | 1,26 |
| threading | 8 | 27,60 | 5,65 | 0,71 |
| ProcessPoolExecutor | 4 | 24,89 | 6,26 | 1,57 |
| ProcessPoolExecutor | 8 | 23,91 | 6,52 | 0,82 |
| numba (njit) | 1 (sin `parallel`) | — | (no listado en CSV; vía 4/8) | — |
| numba (njit) | 4 | 4,98 | 31,33 | 7,83 |
| numba (njit) | 8 | 4,70 | 33,21 | 4,15 |

En el CSV de \(c=1024\) el secuencial transpuesto y Numba con `workers` 1 no aparecen en las filas mostradas; si se requiere el caso Numba *serial* explícito para el análisis “sin paralelismo”, conviene **una corrida adicional** con `matrices_numba.py --workers 1 --no-parallel` y misma `c` y semilla, y añadir la fila a la tabla.

### 4.3 Mejores tiempos (A·B, orden de magnitud)

- **\(c=512\):** el menor tiempo lo aporta **numba** (≈1,0 s con 4–8 hilos); entre métodos *solo Python*, **ProcessPoolExecutor** lidera ligeramente sobre hilos puros a igual *workers*.
- **\(c=1024\):** **numba** sigue siendo claramente el más rápido; entre `threading` y `ThreadPoolExecutor` los tiempos son **muy próximos** (p. ej. ~27,6 s vs ~27,5 s con 8 *workers*); se pueden marcar **ambos** como mejores dentro de esa familia, según la consigna docente.

---

## 5. Análisis (preguntas de la consigna)

**¿Por qué threading no “revienta” el techo de rendimiento en esta carga?**  
El cuerpo del producto fila·columna en Python puro es **intensivo en CPU** y en asignación/trabajo por objeto. Aun con GIL relajado o libre, el coste de interpretación, la granularidad (por fila) y la sincronización hacen que el speed-up se aleje del lineal. Además, al comparar con el secuencial **transpuesto** ya optimizado, el margen de mejora del pool de hilos es menor que respecto del bucle clásico.

**¿Qué explica la mejora de las versiones con matriz transpuesta?**  
El método \(C_{ij} = A_{i:} \cdot (B^T)_{j:}\) mejora la **localidad de acceso** a memoria: se recorren filas consecutivas de \(B^T\) en lugar de saltar columnas de \(B\), reduciendo fallos de caché y el coste de indirección. En \(c=1024\) el secuencial transpuesto (≈86 s) frente al tradicional (≈156 s) lo demuestra de forma clara.

**¿El speed-up de multiprocessing es lineal con los workers? ¿Por qué?**  
**No** de forma completa. Aparece **sobre-ideal** en el CSV (eficiencia > 1 en algunas filas con 4 *workers*) por la definición de *speed-up* frente al secuencial **tradicional** lento, mientras el paralelo aprovecha un camino de código similar al transpuesto. Además, hay **overhead** de *pickling*, arranque de procesos y balanceo de carga. Al pasar a 8 *workers* la eficiencia baja (recursos compartidos, Amdahl, contención), típico de speed-ups sublineales al aumentar `p`.

**¿Qué aporta `numba.njit` incluso con `parallel=False`?**  
Compila el núcleo numérico a **código máquina** (LLVM) con tipos fijos, eliminando el bucle intérprete y permitiendo optimizaciones agresivas. El salto de ~13 s a ~1,2 s en \(c=512\) con un solo hilo ilustra el efecto del JIT **sin** prange.

**¿Se puede superar una eficiencia del 100%? ¿Cómo interpretarlo?**  
Sí, en tablas como la del proyecto: la **eficiencia** se reporta como `speed_up / workers` (u otra fórmula del script) y el *speed-up* se mide frente a un **baseline lento** (tradicional). Si el baseline es artificialmente malo, la “eficiencia” puede > 1 o valores enormes con Numba, sin que ello signifique violar un límite físico: es un **artefacto de la métrica elegida**, no un motor perpetuo. La interpretación correcta es comparar con el secuencial **más representativo** (p. ej. transpuesto) para *speed-up* “honesto”, o fijar explícitamente el baseline en el informe.

---

## 6. Conclusión

Se documenta un benchmark reproducible (misma semilla y checksums alineados) de multiplicación de matrices enteras. El orden transpuesto y Numba ofrecen las mayores reducciones de tiempo; el paralelismo con hilos o procesos mejora frente al triple bucle clásico pero con escalamiento limitado. Las limitaciones incluyen el hardware de 8 hilos, la no inclusión de PyTorch/NumPy en el código entregable y la necesidad de explicitar en el informe el **máximo de hilos Numba** admisible en la máquina de prueba. Como trabajo futuro, podría unificarse el baseline de *speed-up* en el análisis escrito o regenerarse el CSV con una sola referencia acordada.

---

## 7. Referencias

- Consigna: *Trabajo Práctico — Multiplicación de Matrices*, Sistemas Paralelos, UNTDF, 2026.  
- Documentación Python: `concurrent.futures`, `threading`, `multiprocessing`.  
- Documentación Numba: `njit`, `prange`, `set_num_threads`.  

---

## Apéndice: comandos (referencia)

```bash
cd TP2
source .venv/bin/activate
python benchmark.py --c 512  --workers-values 1,4,8 --seed 2026 --perfil cubo --max-val 256 --output resultados_v2_c512.csv  --stream-output
python benchmark.py --c 1024 --workers-values 4,8   --seed 2026 --perfil cubo --max-val 256 --output resultados_v2_c1024.csv --stream-output
```

(Ajustar `workers-values` a la grilla acordada con el cuerpo docente; reemplazar `8` por `10` en métodos no-Numba si el equipo y Numba lo permiten, o acotar Numba a `min(10, tope_hilos)`.)
