# Resultados benchmark — multiplicación de matrices (C = 1024)

Fuente: `resultados_v2_c1024.csv` (ruta relativa al directorio `matrices/` según ubicación del CSV).

Las columnas **speed_up_*** y **eficiencia_*** usan como baseline el tiempo del método **secuencial** (mismo C, perfil y max-val):
- **_ab**: respecto al tiempo del producto **A·B**.
- **_btat**: respecto al tiempo del producto **Bᵀ·Aᵀ**.

Los **checksum** son la suma de todos los elementos de cada matriz resultado (deben coincidir Σ(A·B) y Σ(Bᵀ·Aᵀ)).

## Tabla

| Algoritmo | C | Perfil | max-val | Workers | t A·B (s) | t Bᵀ·Aᵀ (s) | Speed-up A·B | Eficiencia A·B | Speed-up Bᵀ·Aᵀ | Eficiencia Bᵀ·Aᵀ | Cores | Estado | Σ(A·B) | Σ(Bᵀ·Aᵀ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| secuencial | 1024 | cubo | 256 | 1 | 85.870545 | 83.596886 | 1.815835 | 1.815835 | 1.874450 | 1.874450 | 8 | ok | 34818311 | 34818311 |
| secuencial (tradicional) | 1024 | cubo | 256 | 1 | 155.926739 | 156.698163 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 8 | ok | 34818311 | 34818311 |
| concurrent.futures.ThreadPoolExecutor | 1024 | cubo | 256 | 4 | 29.686793 | 30.575668 | 5.252394 | 1.313099 | 5.124930 | 1.281233 | 8 | ok | 34818311 | 34818311 |
| threading | 1024 | cubo | 256 | 4 | 30.872268 | 31.042364 | 5.050706 | 1.262676 | 5.047881 | 1.261970 | 8 | ok | 34818311 | 34818311 |
| concurrent.futures.ProcessPoolExecutor | 1024 | cubo | 256 | 4 | 24.894975 | 25.305801 | 6.263382 | 1.565846 | 6.192183 | 1.548046 | 8 | ok | 34818311 | 34818311 |
| numba (njit) | 1024 | cubo | 256 | 4 | 4.977233 | 5.044623 | 31.327997 | 7.831999 | 31.062413 | 7.765603 | 8 | ok | 34818311 | 34818311 |
| concurrent.futures.ThreadPoolExecutor | 1024 | cubo | 256 | 8 | 27.555593 | 27.622861 | 5.658624 | 0.707328 | 5.672771 | 0.709096 | 8 | ok | 34818311 | 34818311 |
| threading | 1024 | cubo | 256 | 8 | 27.604374 | 28.554723 | 5.648624 | 0.706078 | 5.487644 | 0.685956 | 8 | ok | 34818311 | 34818311 |
| concurrent.futures.ProcessPoolExecutor | 1024 | cubo | 256 | 8 | 23.911540 | 23.711855 | 6.520983 | 0.815123 | 6.608431 | 0.826054 | 8 | ok | 34818311 | 34818311 |
| numba (njit) | 1024 | cubo | 256 | 8 | 4.695137 | 4.671435 | 33.210264 | 4.151283 | 33.543903 | 4.192988 | 8 | ok | 34818311 | 34818311 |

## Notas

- **Eficiencia** = speed-up / número de workers.
- Si falta speed-up/eficiencia para Bᵀ·Aᵀ en un CSV viejo, volvé a ejecutar `benchmark.py` para regenerar columnas.
