# Resultado parcial TP final sepia - PyTorch CPU

## Entorno

- CPU: Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz
- Nucleos fisicos: 4
- Nucleos logicos: 8
- RAM: 15.45 GiB total, 2.30 GiB disponible
- Sistema operativo: Linux-6.12.91-1-MANJARO-x86_64-with-glibc2.43
- Python: 3.12.11 (main, Aug  8 2025, 17:06:48) [Clang 20.1.4 ]
- OpenCV: 4.13.0.92
- PyTorch: 2.11.0+cu128
- PyTorch CUDA: no detectada (import torch excedio 10 s)

## Filtro

Filtro elegido: sepia.

Formula aplicada por pixel, usando canales RGB y salida BGR para OpenCV:

```text
R' = min(255, (393R + 769G + 189B) // 1000)
G' = min(255, (349R + 686G + 168B) // 1000)
B' = min(255, (272R + 534G + 131B) // 1000)
salida OpenCV = B', G', R'
```

## Tabla de benchmark

| metodo | frames | resolucion | fps original | lectura/decodif. (s) | filtrado (s) | escritura/codif. (s) | total pipeline (s) | FPS efectivos | speed-up | memoria pico (MB) | estado |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PyTorch CPU | 1799 | 3840x2160 | 59.940 | 13.422055 | 447.948625 | 91.757641 | 553.128321 | 3.268387 |  | 1378.95 | ok |

## Datos de control

| metodo | corridas | workers | duracion entrada (s) | codec | max_frames | checksum | hash salida | video sin audio | video con audio |
|---|---:|---:|---:|---|---:|---:|---|---|---|
| PyTorch CPU | 3 | 4 | 30.013315 | mp4v | 1799 | 3788871516465 | 7685a7d3304f5e10 | /mnt/sda1/code/facu/paralelos/tp-final/resultados/videos/sepia_pytorch_cpu_sin_audio.mp4 | /mnt/sda1/code/facu/paralelos/tp-final/resultados/videos/sepia_pytorch_cpu_con_audio.mp4 |

## Notas metodologicas

- El video se procesa como flujo: no se carga completo en memoria.
- El tiempo de pipeline es lectura/decodificacion + filtrado + escritura/codificacion.
- El merge de audio con ffmpeg se mide aparte y no se suma al pipeline de filtrado.
- Speed-up = tiempo total del pipeline secuencial / tiempo total del pipeline del metodo.
- Si falta la fila secuencial, el speed-up queda vacio porque falta la linea base.
