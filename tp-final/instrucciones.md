# Instrucciones de ejecucion

Estas instrucciones se ejecutan desde la carpeta `tp-final/`, tomada como proyecto Python
independiente.

No ejecutar estos comandos desde `/mnt/sda1/code/facu/paralelos`, porque eso puede crear rutas
anidadas incorrectas.

## 1. Entrar al proyecto

```bash
cd /mnt/sda1/code/facu/paralelos/tp-final
pwd
```

El `pwd` debe mostrar:

```text
/mnt/sda1/code/facu/paralelos/tp-final
```

## 2. Crear y activar el entorno virtual

El entorno virtual debe quedar en:

```text
/mnt/sda1/code/facu/paralelos/tp-final/.venv
```

En esta maquina conviene usar Python 3.12 para PyTorch:

```bash
/home/gip/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

La version esperada es:

```text
Python 3.12.11
```

Actualizar `pip`:

```bash
python -m pip install --upgrade pip
```

## 3. Instalar dependencias

Como el TP requiere PyTorch CPU y PyTorch GPU, se instala PyTorch con CUDA. Si el driver NVIDIA
no esta operativo, PyTorch CPU funciona igual, pero PyTorch GPU va a quedar no disponible.

```bash
python -m pip install opencv-python
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verificar:

```bash
python - <<'PY'
import cv2
import torch

print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

En esta maquina, al momento de preparar estas instrucciones, `nvidia-smi` fallaba. Si sigue
fallando, correr el TP con `--skip-gpu` y documentar que CUDA no estaba operativo.

```bash
nvidia-smi
```

## 4. Preparar el video de entrada de 30 segundos

El archivo original `video_facultad.webm` esta en AV1. OpenCV puede detectar metadata del archivo,
pero en esta maquina no puede decodificar sus frames correctamente. Por eso se crea primero un
clip de 30 segundos en H.264/MP4, manteniendo el audio del tramo original.

```bash
mkdir -p entrada
ffmpeg -y \
  -t 30 \
  -i video_facultad.webm \
  -map 0:v:0 \
  -map '0:a?' \
  -c:v libx264 \
  -preset veryfast \
  -crf 23 \
  -pix_fmt yuv420p \
  -c:a aac \
  -b:a 192k \
  entrada/video_facultad_30s_h264.mp4
```

Verificar el clip:

```bash
ffprobe -hide_banner entrada/video_facultad_30s_h264.mp4
```

Verificar que OpenCV puede leer frames:

```bash
python - <<'PY'
import cv2

cap = cv2.VideoCapture("entrada/video_facultad_30s_h264.mp4")
print("opened:", cap.isOpened())
print("width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("fps:", cap.get(cv2.CAP_PROP_FPS))
ok, frame = cap.read()
print("read:", ok, None if frame is None else frame.shape)
cap.release()
PY
```

La lectura debe mostrar:

```text
opened: True
read: True
```

## 5. Compilar el codigo

```bash
python -m py_compile \
  codigo/sepia_filter.py \
  codigo/resultados_video.py \
  codigo/procesamiento_video.py \
  codigo/secuencial.py \
  codigo/pytorch_cpu.py \
  codigo/pytorch_gpu.py \
  codigo/main.py
```

## 6. Prueba corta

Ejecutar pocos frames antes de medir el clip completo.

Secuencial:

```bash
python codigo/secuencial.py \
  --input entrada/video_facultad_30s_h264.mp4 \
  --runs 1 \
  --max-frames 3 \
  --no-progress \
  --output-dir resultados/pruebas
```

PyTorch CPU:

```bash
python codigo/pytorch_cpu.py \
  --input entrada/video_facultad_30s_h264.mp4 \
  --runs 1 \
  --workers 4 \
  --max-frames 3 \
  --no-progress \
  --output-dir resultados/pruebas
```

PyTorch GPU, solo si `torch.cuda.is_available()` devuelve `True`:

```bash
python codigo/pytorch_gpu.py \
  --input entrada/video_facultad_30s_h264.mp4 \
  --runs 1 \
  --workers 4 \
  --max-frames 3 \
  --no-progress \
  --output-dir resultados/pruebas
```

Prueba del flujo completo sin GPU:

```bash
python codigo/main.py \
  --input entrada/video_facultad_30s_h264.mp4 \
  --runs 1 \
  --workers 4 \
  --max-frames 3 \
  --no-progress \
  --skip-gpu \
  --output-dir resultados/pruebas
```

Prueba del flujo completo con GPU:

```bash
python codigo/main.py \
  --input entrada/video_facultad_30s_h264.mp4 \
  --runs 1 \
  --workers 4 \
  --max-frames 3 \
  --no-progress \
  --output-dir resultados/pruebas
```

## 7. Mediciones finales

Usar siempre `--workers 4`, porque la CPU tiene 4 nucleos fisicos.

### Final sin GPU

```bash
python codigo/main.py \
  --input entrada/video_facultad_30s_h264.mp4 \
  --runs 3 \
  --workers 4 \
  --seconds 30 \
  --merge-audio \
  --skip-gpu \
  --output-dir resultados
```

### Final con GPU

```bash
python codigo/main.py \
  --input entrada/video_facultad_30s_h264.mp4 \
  --runs 3 \
  --workers 4 \
  --seconds 30 \
  --merge-audio \
  --output-dir resultados
```

## 8. Ejecutar metodo por metodo

Secuencial:

```bash
python codigo/secuencial.py \
  --input entrada/video_facultad_30s_h264.mp4 \
  --runs 3 \
  --workers 4 \
  --seconds 30 \
  --merge-audio \
  --output-dir resultados
```

PyTorch CPU:

```bash
python codigo/pytorch_cpu.py \
  --input entrada/video_facultad_30s_h264.mp4 \
  --runs 3 \
  --workers 4 \
  --seconds 30 \
  --merge-audio \
  --output-dir resultados
```

PyTorch GPU:

```bash
python codigo/pytorch_gpu.py \
  --input entrada/video_facultad_30s_h264.mp4 \
  --runs 3 \
  --workers 4 \
  --seconds 30 \
  --merge-audio \
  --output-dir resultados
```

## 9. Salidas esperadas

Resultados:

```text
resultados/resultados_video4k_sepia.csv
resultados/resultados_video4k_sepia.md
resultados/parciales/resultado_parcial_sepia_secuencial.md
resultados/parciales/resultado_parcial_sepia_pytorch_cpu.md
resultados/parciales/resultado_parcial_sepia_pytorch_gpu.md
```

Videos:

```text
resultados/videos/sepia_secuencial_sin_audio.mp4
resultados/videos/sepia_secuencial_con_audio.mp4
resultados/videos/sepia_pytorch_cpu_sin_audio.mp4
resultados/videos/sepia_pytorch_cpu_con_audio.mp4
resultados/videos/sepia_pytorch_gpu_sin_audio.mp4
resultados/videos/sepia_pytorch_gpu_con_audio.mp4
```

## 10. Notas

- `--workers 4` corresponde a nucleos fisicos.
- No usar `--workers 8` para la comparacion principal.
- PyTorch GPU no usa "4 workers GPU"; usa CUDA.
- El merge de audio se mide aparte y no se suma al tiempo del pipeline.
- El clip H.264 de entrada conserva el audio del tramo original, transcodificado a AAC para MP4.
- La version secuencial puede tardar mucho porque recorre pixels 4K con bucles Python.
- En una prueba minima de 1 frame 4K, la version secuencial tardo varios segundos. Procesar 30
  segundos completos puede llevar horas, especialmente si se usan 3 corridas.
- No hay una implementacion NumPy del filtro. OpenCV entrega los frames como arreglos, pero la
  version secuencial calcula el sepia con bucles Python y PyTorch convierte esos frames a tensores
  en sus variantes CPU/GPU.
