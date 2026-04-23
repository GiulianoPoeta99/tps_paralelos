# Resultados de suma paralela (CSV 2 + CSV 3)

Fuente:

- `resultados_suma2.csv`
- `resultados_suma3.csv`

## Tabla completa (`c = 1000000`)

| algoritmo | c | workers | tiempo (s) | speed-up | eficiencia | cores | estado | suma final |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| secuencial | 1000000 | 1 | 0.204124 | 1.000000 | 1.000000 | 22 | ok | 499998.460826 |
| concurrent.futures.ThreadPoolExecutor | 1000000 | 1 | 0.197512 | 1.033476 | 1.033476 | 22 | ok | 499998.460826 |
| concurrent.futures.ProcessPoolExecutor | 1000000 | 1 | 0.206578 | 0.988121 | 0.988121 | 22 | ok | 499998.460826 |
| multiprocessing.Pool | 1000000 | 1 | 0.185514 | 1.100316 | 1.100316 | 22 | ok | 499998.460826 |
| threading | 1000000 | 1 | 0.198378 | 1.028965 | 1.028965 | 22 | ok | 499998.460826 |
| concurrent.futures.ThreadPoolExecutor | 1000000 | 2 | 0.249538 | 0.818008 | 0.409004 | 22 | ok | 499998.460826 |
| concurrent.futures.ProcessPoolExecutor | 1000000 | 2 | 0.118626 | 1.720736 | 0.860368 | 22 | ok | 499998.460826 |
| multiprocessing.Pool | 1000000 | 2 | 0.119088 | 1.714060 | 0.857030 | 22 | ok | 499998.460826 |
| threading | 1000000 | 2 | 0.241440 | 0.845444 | 0.422722 | 22 | ok | 499998.460826 |
| concurrent.futures.ThreadPoolExecutor | 1000000 | 4 | 0.252064 | 0.809810 | 0.202453 | 22 | ok | 499998.460826 |
| concurrent.futures.ProcessPoolExecutor | 1000000 | 4 | 0.071966 | 2.836395 | 0.709099 | 22 | ok | 499998.460826 |
| multiprocessing.Pool | 1000000 | 4 | 0.076067 | 2.683476 | 0.670869 | 22 | ok | 499998.460826 |
| threading | 1000000 | 4 | 0.249851 | 0.816983 | 0.204246 | 22 | ok | 499998.460826 |
| concurrent.futures.ThreadPoolExecutor | 1000000 | 8 | 0.244534 | 0.834747 | 0.104343 | 22 | ok | 499998.460826 |
| concurrent.futures.ProcessPoolExecutor | 1000000 | 8 | 0.046610 | 4.379404 | 0.547425 | 22 | ok | 499998.460826 |
| multiprocessing.Pool | 1000000 | 8 | 0.053078 | 3.845736 | 0.480717 | 22 | ok | 499998.460826 |
| threading | 1000000 | 8 | 0.255507 | 0.798898 | 0.099862 | 22 | ok | 499998.460826 |
| concurrent.futures.ThreadPoolExecutor | 1000000 | 16 | 0.243063 | 0.839799 | 0.052487 | 22 | ok | 499998.460826 |
| concurrent.futures.ProcessPoolExecutor | 1000000 | 16 | 0.045292 | 4.506844 | 0.281678 | 22 | ok | 499998.460826 |
| multiprocessing.Pool | 1000000 | 16 | 0.052623 | 3.878988 | 0.242437 | 22 | ok | 499998.460826 |
| threading | 1000000 | 16 | 0.252631 | 0.807993 | 0.050500 | 22 | ok | 499998.460826 |

## Tabla completa (`c = 100000000`)

| algoritmo | c | workers | tiempo (s) | speed-up | eficiencia | cores | estado | suma final |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| secuencial | 100000000 | 1 | 10.925741 | 1.000000 | 1.000000 | 22 | ok | 50000000.706162 |
| concurrent.futures.ThreadPoolExecutor | 100000000 | 1 | 11.378959 | 0.960171 | 0.960171 | 22 | ok | 50000000.706162 |
| concurrent.futures.ProcessPoolExecutor | 100000000 | 1 | 11.029577 | 0.990586 | 0.990586 | 22 | ok | 50000000.706162 |
| multiprocessing.Pool | 100000000 | 1 | 11.605306 | 0.941444 | 0.941444 | 22 | ok | 50000000.706162 |
| threading | 100000000 | 1 | 11.241798 | 0.971886 | 0.971886 | 22 | ok | 50000000.706162 |
| concurrent.futures.ThreadPoolExecutor | 100000000 | 2 | 15.762657 | 0.693141 | 0.346570 | 22 | ok | 50000000.706162 |
| concurrent.futures.ProcessPoolExecutor | 100000000 | 2 | 6.197966 | 1.762795 | 0.881397 | 22 | ok | 50000000.706162 |
| multiprocessing.Pool | 100000000 | 2 | 6.105219 | 1.789574 | 0.894787 | 22 | ok | 50000000.706162 |
| threading | 100000000 | 2 | 19.657007 | 0.555819 | 0.277910 | 22 | ok | 50000000.706162 |
| concurrent.futures.ThreadPoolExecutor | 100000000 | 4 | 20.511757 | 0.532657 | 0.133164 | 22 | ok | 50000000.706162 |
| concurrent.futures.ProcessPoolExecutor | 100000000 | 4 | 3.476167 | 3.143043 | 0.785761 | 22 | ok | 50000000.706162 |
| multiprocessing.Pool | 100000000 | 4 | 3.327934 | 3.283040 | 0.820760 | 22 | ok | 50000000.706162 |
| threading | 100000000 | 4 | 21.273273 | 0.513590 | 0.128398 | 22 | ok | 50000000.706162 |
| concurrent.futures.ThreadPoolExecutor | 100000000 | 8 | 22.867227 | 0.477790 | 0.059724 | 22 | ok | 50000000.706162 |
| concurrent.futures.ProcessPoolExecutor | 100000000 | 8 | 2.124283 | 5.143261 | 0.642908 | 22 | ok | 50000000.706162 |
| multiprocessing.Pool | 100000000 | 8 | 2.120283 | 5.152964 | 0.644120 | 22 | ok | 50000000.706162 |
| threading | 100000000 | 8 | 21.659920 | 0.504422 | 0.063053 | 22 | ok | 50000000.706162 |
| concurrent.futures.ThreadPoolExecutor | 100000000 | 16 | 22.835965 | 0.478444 | 0.029903 | 22 | ok | 50000000.706162 |
| concurrent.futures.ProcessPoolExecutor | 100000000 | 16 | 1.386949 | 7.877536 | 0.492346 | 22 | ok | 50000000.706162 |
| multiprocessing.Pool | 100000000 | 16 | 1.407404 | 7.763045 | 0.485190 | 22 | ok | 50000000.706162 |
| threading | 100000000 | 16 | 22.391425 | 0.487943 | 0.030496 | 22 | ok | 50000000.706162 |
