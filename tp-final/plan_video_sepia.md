# Plan de trabajo - TP final video 4K con filtro sepia

## Objetivo

Implementar y comparar el procesamiento de un video 4K de 30 segundos aplicando un filtro
sepia. La comparacion debe incluir:

- Python secuencial como linea base.
- PyTorch CPU.
- PyTorch GPU, si CUDA esta disponible.

El video procesado debe conservar el audio original.

## Consigna tomada del libro

La seccion 10.5 del libro `tp-final/sistemas-paralelos-book.pdf` pide:

- Procesar un video 4K de aproximadamente 30 segundos.
- Aplicar un filtro elegido por el estudiante.
- Procesar frame por frame o en lotes chicos.
- No cargar el video completo en memoria.
- Reconstruir el video final.
- Reincorporar el audio original si el video de entrada lo tiene.
- Medir lectura/decodificacion, filtrado, escritura/codificacion y tiempo total del pipeline.
- Reportar frames procesados, resolucion, FPS original, FPS efectivos, codec/herramienta usada,
  memoria aproximada y speed-up respecto del secuencial.
- Sincronizar GPU antes de detener temporizadores cuando se mida PyTorch GPU.

## Video de entrada

Archivo:

```text
tp-final/video_facultad.webm
```

Datos detectados con `ffprobe`:

- Resolucion: `3840x2160`.
- FPS: `59.94`.
- Duracion total: `00:50:15.94`.
- Codec de video: `AV1`.
- Codec de audio: `Opus`.
- El video tiene audio, por lo tanto hay que reincorporarlo en la salida final.

Para la entrega se deben procesar solo los primeros 30 segundos. A `59.94 FPS`, eso equivale
aproximadamente a `1798` frames.

## Hardware y criterio de workers

CPU detectada:

```text
Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz
```

Caracteristicas:

- Nucleos fisicos: `4`.
- Hilos logicos: `8`.
- Threads por core: `2`.

Para este trabajo se debe usar la cantidad de nucleos fisicos cuando se configure paralelismo en
CPU. Por lo tanto:

```text
workers = 4
```

No se deben usar `8` workers para PyTorch CPU, porque esos son hilos logicos y mezclan el analisis
con efectos de concurrencia/SMT.

En PyTorch GPU no corresponde hablar de "4 workers GPU". La version GPU usa el dispositivo CUDA.
Lo que si puede limitarse a `4` son los hilos CPU auxiliares si el codigo los configura, pero el
trabajo computacional principal de esa variante se describe como ejecucion en GPU.

## Ejemplo de referencia

El ejemplo mas importante para no inventar estructura esta en:

```text
SistemasParalelos/video4k/
```

Archivos relevantes:

```text
SistemasParalelos/video4k/codigo/posterize_filter.py
SistemasParalelos/video4k/codigo/benchmark_video4k.py
SistemasParalelos/video4k/codigo/video_lib.py
SistemasParalelos/video4k/codigo/generar_resumen_video4k.py
SistemasParalelos/video4k/codigo/Documentatio.md
SistemasParalelos/video4k/informe/trabajo_practico_4.md
```

Ese ejemplo ya implementa:

- Procesamiento frame por frame.
- Version secuencial.
- Version PyTorch CPU.
- Version PyTorch GPU.
- Medicion por etapas.
- CSV de resultados.
- Markdown parcial por metodo.
- Markdown agregado.
- Reincorporacion de audio con `ffmpeg`.

Hay que adaptarlo al filtro sepia y corregir el criterio de workers para esta maquina.

## Interpretacion de "Python puro secuencial"

La version secuencial debe aplicar el filtro con bucles Python, sin usar PyTorch, NumPy vectorizado
ni otra biblioteca para el calculo pixel a pixel.

Para leer y escribir video se puede usar OpenCV o `ffmpeg`, porque decodificar/codificar video sin
herramientas externas no es el objetivo del TP. El punto importante es que el filtro secuencial sea
la linea base de computo puro.

## Filtro sepia

El filtro elegido es sepia. Para cada pixel se transforma el color original a una tonalidad sepia.

Como OpenCV trabaja los frames en formato BGR, hay que tener cuidado con el orden de canales:

```text
frame_bgr[:, :, 0] = B
frame_bgr[:, :, 1] = G
frame_bgr[:, :, 2] = R
```

Formula propuesta, usando enteros para mantener equivalencia entre implementaciones:

```text
R' = min(255, (393R + 769G + 189B) // 1000)
G' = min(255, (349R + 686G + 168B) // 1000)
B' = min(255, (272R + 534G + 131B) // 1000)
```

La salida debe escribirse nuevamente en formato BGR:

```text
B', G', R'
```

Usar enteros evita diferencias de redondeo entre la version secuencial y PyTorch.

## Estructura propuesta

Crear la siguiente estructura:

```text
tp-final/
  codigo/
    sepia_filter.py
    resultados_video.py
    procesamiento_video.py
    secuencial.py
    pytorch_cpu.py
    pytorch_gpu.py
    main.py
    generar_resumen_video4k.py
  resultados/
    parciales/
    videos/
    finales/
```

## Archivos a implementar

### 1. `sepia_filter.py`

Debe contener:

- Etiquetas de metodos.
- Clase o funcion de tiempos del filtro.
- Validacion de frames.
- Funcion secuencial pura del filtro sepia.
- Implementacion `SequentialSepia`.
- Implementacion `TorchSepiaCPU`.
- Implementacion `TorchSepiaGPU`.
- Funcion `build_processor`.

Clases esperadas:

```text
SequentialSepia
TorchSepiaCPU
TorchSepiaGPU
```

La version CPU de PyTorch debe usar:

```python
torch.set_num_threads(4)
```

o recibir `workers=4` desde el benchmark.

La version GPU debe usar:

```python
torch.cuda.synchronize()
```

antes de cerrar tiempos de transferencia y computo.

### 2. `resultados_video.py`

Debe adaptar la version de `SistemasParalelos/video4k/codigo/video_lib.py`.

Responsabilidades:

- Detectar informacion del video.
- Detectar entorno de ejecucion.
- Calcular memoria pico aproximada.
- Guardar CSV.
- Guardar Markdown parcial por metodo.
- Guardar Markdown agregado.
- Calcular speed-up respecto del secuencial.
- Reincorporar audio con `ffmpeg`.

Importante: los nombres de salida deben decir `sepia`, no `posterize`.

### 3. `procesamiento_video.py`

Debe contener la logica compartida de medicion para evitar duplicar codigo entre metodos.

Responsabilidades:

- Abrir el video.
- Calcular cuantos frames procesar segun `--seconds 30` o `--max-frames`.
- Leer frame por frame.
- Aplicar el procesador recibido.
- Escribir el video filtrado.
- Calcular tiempos de lectura, filtrado, escritura y pipeline total.
- Calcular checksum y hash de salida.
- Guardar CSV y Markdown.
- Reincorporar audio si se pide `--merge-audio`.

Este archivo no representa un metodo por si mismo: solo concentra la infraestructura comun.

### 4. `secuencial.py`

Debe ejecutar solo la version secuencial en Python.

Usa:

```text
SequentialSepia
```

Debe poder ejecutarse de forma individual y tambien ser llamado desde `main.py`.

### 5. `pytorch_cpu.py`

Debe ejecutar solo la version PyTorch CPU.

Usa:

```text
TorchSepiaCPU
```

Debe usar `workers=4` por defecto, porque la maquina tiene 4 nucleos fisicos.

Debe poder ejecutarse de forma individual y tambien ser llamado desde `main.py`.

### 6. `pytorch_gpu.py`

Debe ejecutar solo la version PyTorch GPU.

Usa:

```text
TorchSepiaGPU
```

Debe sincronizar CUDA antes de cerrar las mediciones internas de transferencia y computo.

Debe poder ejecutarse de forma individual y tambien ser llamado desde `main.py`.

### 7. `main.py`

Debe ejecutar las mediciones en orden, mostrando mensajes informativos y divisores:

```text
==============================
Medicion secuencial
==============================
...

==============================
Medicion PyTorch CPU
==============================
...

==============================
Medicion PyTorch GPU
==============================
...
```

Si una variante falla, debe registrar el error y continuar con las siguientes cuando sea posible.

Debe procesar el video como flujo:

```text
leer frame -> aplicar filtro -> escribir frame -> liberar referencias -> continuar
```

No debe cargar el video completo en memoria.

Los scripts individuales (`secuencial.py`, `pytorch_cpu.py` y `pytorch_gpu.py`) deben permitir:

- `--runs`
- `--workers`, con default `4`
- `--max-frames` para pruebas cortas
- `--seconds 30` o mecanismo equivalente para cortar en los primeros 30 segundos
- `--merge-audio`
- `--codec`
- `--output-dir`

La medicion final debe usar `30` segundos, no el video completo de 50 minutos.

El `main` debe permitir ejecutar todos los metodos en orden o seleccionar una lista con:

```text
--methods secuencial pytorch_cpu pytorch_gpu
```

La idea principal es que cada metodo tenga su archivo propio y que el `main` solo coordine la
ejecucion.

### 8. `generar_resumen_video4k.py`

Debe leer el CSV acumulado y generar un resumen Markdown final con todos los metodos ejecutados.

## Salidas esperadas

Resultados:

```text
tp-final/resultados/resultados_video4k_sepia.csv
tp-final/resultados/resultados_video4k_sepia.md
tp-final/resultados/parciales/resultado_parcial_sepia_secuencial.md
tp-final/resultados/parciales/resultado_parcial_sepia_pytorch_cpu.md
tp-final/resultados/parciales/resultado_parcial_sepia_pytorch_gpu.md
```

Videos:

```text
tp-final/resultados/videos/sepia_secuencial_sin_audio.mp4
tp-final/resultados/videos/sepia_secuencial_con_audio.mp4
tp-final/resultados/videos/sepia_pytorch_cpu_sin_audio.mp4
tp-final/resultados/videos/sepia_pytorch_cpu_con_audio.mp4
tp-final/resultados/videos/sepia_pytorch_gpu_sin_audio.mp4
tp-final/resultados/videos/sepia_pytorch_gpu_con_audio.mp4
```

Como el video original tiene audio, para la entrega interesan especialmente las versiones
`*_con_audio.mp4`.

## Metricas a registrar

Por metodo:

- Frames procesados.
- Resolucion.
- FPS original.
- Tiempo de lectura/decodificacion.
- Tiempo de filtrado.
- Tiempo de computo del filtro.
- Tiempo de escritura/codificacion.
- Tiempo total del pipeline.
- FPS efectivos.
- Speed-up respecto del secuencial.
- Memoria pico aproximada.
- Codec usado.
- Checksum.
- Hash de salida.
- Estado de la corrida.
- Error, si corresponde.

Para PyTorch GPU, ademas:

- Transferencia CPU -> GPU.
- Computo GPU.
- Transferencia GPU -> CPU.
- Transferencia total.

## Informe final

El informe debe incluir:

- Abstract.
- Introduccion.
- Metodologia.
- Resultados.
- Analisis.
- Manejo de memoria.
- Reconstruccion de video y audio.
- Conclusiones.
- Codigo utilizado.

La version secuencial debe ser siempre la linea base para calcular speed-up.

Tambien conviene incluir una comparacion complementaria:

- PyTorch CPU vs secuencial.
- PyTorch GPU vs secuencial.
- PyTorch GPU vs PyTorch CPU.
- Tiempo de computo GPU separado de transferencias.
- Impacto de lectura/escritura de video sobre el pipeline total.

## Orden de trabajo propuesto

1. Crear estructura de carpetas en `tp-final/`.
2. Implementar solo `sepia_filter.py`.
3. Probar el filtro con un frame sintetico chico.
4. Adaptar `resultados_video.py`.
5. Implementar `procesamiento_video.py`.
6. Implementar `secuencial.py`.
7. Implementar `pytorch_cpu.py`.
8. Implementar `pytorch_gpu.py`.
9. Implementar `main.py`.
10. Hacer una prueba corta con pocos frames.
11. Resolver o documentar el entorno PyTorch/OpenCV necesario.
12. Ejecutar medicion secuencial de 30 segundos.
13. Ejecutar medicion PyTorch CPU con `workers=4`.
14. Ejecutar medicion PyTorch GPU si CUDA funciona.
15. Generar Markdown agregado.
16. Redactar informe final.

Los comandos de preparacion del entorno y ejecucion estan en:

```text
tp-final/instrucciones.md
```

## Pendientes tecnicos antes de correr mediciones

- Verificar que el entorno virtual elegido tenga OpenCV.
- Verificar que el entorno virtual elegido tenga PyTorch.
- Revisar por que `import torch` en `tp3.3/.venv` queda colgado.
- No modificar los entornos usados para entregas anteriores sin decidirlo antes.
- Generar primero un clip H.264/MP4 de 30 segundos, porque OpenCV no puede decodificar
  correctamente los frames AV1 del `video_facultad.webm` original en este entorno.

## Decision recomendada sobre el clip de 30 segundos

Para evitar procesar accidentalmente los 50 minutos completos, conviene crear o usar una entrada
de 30 segundos antes de la medicion final, por ejemplo:

```text
tp-final/entrada/video_facultad_30s_h264.mp4
```

El script tambien puede aceptar `--seconds 30`, pero tener un archivo corto hace mas dificil cometer
errores durante las corridas largas.

Ademas, en este entorno OpenCV detecta la metadata del archivo `video_facultad.webm`, pero no
puede leer sus frames AV1. Por eso la entrada de trabajo debe generarse con `ffmpeg` en H.264/MP4,
manteniendo el audio del tramo original.
