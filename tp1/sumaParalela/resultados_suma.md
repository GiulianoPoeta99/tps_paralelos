# Resultados de suma paralela (Markdown)

Fuente: `resultados_suma.csv` (estado actual).

## Tabla completa


| algoritmo                              | c   | workers | tiempo (s) | speed-up | eficiencia | cores | estado | suma final |
| -------------------------------------- | --- | ------- | ---------- | -------- | ---------- | ----- | ------ | ---------- |
| secuencial                             | 12  | 1       | 0.000011   | 1.000000 | 1.000000   | 22    | ok     | 4.984695   |
| concurrent.futures.ThreadPoolExecutor  | 12  | 1       | 0.000726   | 0.015152 | 0.015152   | 22    | ok     | 4.984695   |
| concurrent.futures.ProcessPoolExecutor | 12  | 1       | 0.006258   | 0.001758 | 0.001758   | 22    | ok     | 4.984695   |
| multiprocessing.Pool                   | 12  | 1       | 0.014928   | 0.000737 | 0.000737   | 22    | ok     | 4.984695   |
| threading                              | 12  | 1       | 0.000192   | 0.057292 | 0.057292   | 22    | ok     | 4.984695   |
| concurrent.futures.ThreadPoolExecutor  | 12  | 2       | 0.000697   | 0.015782 | 0.007891   | 22    | ok     | 4.984695   |
| concurrent.futures.ProcessPoolExecutor | 12  | 2       | 0.006707   | 0.001640 | 0.000820   | 22    | ok     | 4.984695   |
| multiprocessing.Pool                   | 12  | 2       | 0.014012   | 0.000785 | 0.000393   | 22    | ok     | 4.984695   |
| threading                              | 12  | 2       | 0.000173   | 0.063584 | 0.031792   | 22    | ok     | 4.984695   |
| concurrent.futures.ThreadPoolExecutor  | 12  | 4       | 0.004233   | 0.002599 | 0.000650   | 22    | ok     | 4.984695   |
| concurrent.futures.ProcessPoolExecutor | 12  | 4       | 0.009929   | 0.001108 | 0.000277   | 22    | ok     | 4.984695   |
| multiprocessing.Pool                   | 12  | 4       | 0.024246   | 0.000454 | 0.000113   | 22    | ok     | 4.984695   |
| threading                              | 12  | 4       | 0.000599   | 0.018364 | 0.004591   | 22    | ok     | 4.984695   |
| concurrent.futures.ThreadPoolExecutor  | 12  | 8       | 0.002410   | 0.004564 | 0.000571   | 22    | ok     | 4.984695   |
| concurrent.futures.ProcessPoolExecutor | 12  | 8       | 0.014081   | 0.000781 | 0.000098   | 22    | ok     | 4.984695   |
| multiprocessing.Pool                   | 12  | 8       | 0.020767   | 0.000530 | 0.000066   | 22    | ok     | 4.984695   |
| threading                              | 12  | 8       | 0.000713   | 0.015428 | 0.001928   | 22    | ok     | 4.984695   |
| concurrent.futures.ThreadPoolExecutor  | 12  | 16      | 0.003628   | 0.003032 | 0.000189   | 22    | ok     | 4.984695   |
| concurrent.futures.ProcessPoolExecutor | 12  | 16      | 0.021638   | 0.000508 | 0.000032   | 22    | ok     | 4.984695   |
| multiprocessing.Pool                   | 12  | 16      | 0.028092   | 0.000392 | 0.000024   | 22    | ok     | 4.984695   |
| threading                              | 12  | 16      | 0.001896   | 0.005802 | 0.000363   | 22    | ok     | 4.984695   |


## Resumen rapido (mejor speed-up por algoritmo, c=12)


| algoritmo                              | mejor p | mejor speed-up | eficiencia asociada |
| -------------------------------------- | ------- | -------------- | ------------------- |
| concurrent.futures.ThreadPoolExecutor  | 2       | 0.015782       | 0.007891            |
| concurrent.futures.ProcessPoolExecutor | 1       | 0.001758       | 0.001758            |
| multiprocessing.Pool                   | 2       | 0.000785       | 0.000393            |
| threading                              | 2       | 0.063584       | 0.031792            |


> Nota: para `c=12`, el overhead de paralelizar domina el tiempo total.

