# Trabajo práctico con suma paralela de un vector

**Materia:** Sistemas Paralelos - Lic. en Sistemas, 5to año  
**Institución:** UNTDF  
**Docente:** MsC. Federico Gonzalez Brizzio

---

## Consigna

Desarrollar la suma en paralelo de un vector de complejidad "c", utilizando las librerías de multiprocessing analizadas anteriormente. Utilizar el código base provisto.

## Informe (una página)

1. **Tabla de resultados** con las siguientes columnas:
   - Algoritmo (threads, procesos, secuencial)
   - Complejidad
   - Procesos
   - Tiempo
   - Speed-up
   - Eficiencia
   - Equipo (cores)

2. **Ejecuciones requeridas:**
   - `p = 1, 2, 4, 8, 16`
   - `c = 12, 1000000, 100000000`
   - Destacar el mayor speed-up y eficiencia para cada bloque de ejecución por equipo.

3. **Reflexión** sobre los resultados. Detallar los equipos utilizados, particularmente la cantidad de cores disponibles.

## Código base

```python
from time import time
import numpy as np

def sum_elements(v):
    res = 0
    for e in v:
        res += e
    return res

if __name__ == '__main__':
    c = 100000000
    arr = np.random.rand(c)
    init = time()
    result = sum_elements(arr)
    end = time() - init
    print("Result:", result, "Time:", end, "With C =", c)
```
