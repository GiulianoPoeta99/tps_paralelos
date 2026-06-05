import math
from time import perf_counter

import numpy as np
from PIL import Image

from numba import cuda


gx_kernel = (
    (-1, 0, 1),
    (-2, 0, 2),
    (-1, 0, 1),
)
gy_kernel = (
    (1, 2, 1),
    (0, 0, 0),
    (-1, -2, -1),
)

# rgb es la matriz con la imagen color; gray sera la matriz resultante
@cuda.jit
def rgb_to_gray_cuda(rgb: np.ndarray, gray: np.ndarray) -> None:

    # Cada hilo obtiene una coordenada (y, x) unica en una grilla 2D.
    y, x = cuda.grid(2)

    # Un hilo procesa un solo pixel: no hay for en Python porque
    # el recorrido completo lo hacen miles de hilos en paralelo.
    r = float(rgb[y, x, 0]) # x,y = [r,g,b]
    g = float(rgb[y, x, 1])
    b = float(rgb[y, x, 2])

    i = int(0.299 * r + 0.587 * g + 0.114 * b)  # (r+g+b)/3 = promedio == gris
    #i = 0 if i < 0 else 255 if i > 255 else i   # chequea que no se vaya de rango (0,255)

    # i tiene el valor gris del pixel
    gray[y, x] = i


@cuda.jit
def sobel_cuda(gray: np.ndarray, out: np.ndarray) -> None:
    # Igual que en rgb_to_gray_cuda: cada hilo trabaja sobre un pixel (y, x).
    y, x = cuda.grid(2)

    # iniciamos la gradiente en cero
    gx = 0
    gy = 0

    # Estos for NO recorren toda la imagen: solo la vecindad 3x3
    # del pixel actual asignado a este hilo.
    for ky in range(3):
        for kx in range(3):
            p = int(gray[y + ky - 1, x + kx - 1])
            gx += p * gx_kernel[ky][kx]
            gy += p * gy_kernel[ky][kx]

    mag = int(math.sqrt(gx * gx + gy * gy))
    out[y, x] = 255 if mag > 255 else mag


def white_percentage(gray: np.ndarray) -> float:
    total_pixels = gray.size
    white_pixels = np.count_nonzero(gray == 255)
    return float(white_pixels) * 100.0 / float(total_pixels)


input_path = "image.jpg"
#input_path = "laguna-esmeralda-small.jpg"
output_path = "sobel-image.jpg"

image = Image.open(input_path).convert("RGB")
width, height = image.size

# Creamos las variables donde trabajaremos
# La asignacion de uint8 es importante
rgb_host = np.asarray(image, dtype=np.uint8) # cargamos la imagen en un array en la CPU
rgb_device = cuda.to_device(rgb_host)  # llevamos la imagen a CUDA (GPU)

# reservamos dos arrays para las imagenes (gray y sobel)
gray_device = cuda.device_array((height, width), dtype=np.uint8)
sobel_device = cuda.device_array((height, width), dtype=np.uint8)

# Leer aqui por que https://chatgpt.com/share/69fd7f9e-9808-83eb-9177-a080206c3b53
THREADS_PER_BLOCK = (16, 16) # 256

# con 6000px => blockspergrid_y = (6000 + 16 - 1) // 16 = 375
# con 750px  => blockspergrid_y = (6000 + 16 - 1) // 16 = 47
blockspergrid_y = (height + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0]
blockspergrid_x = (width + THREADS_PER_BLOCK[1] - 1) // THREADS_PER_BLOCK[1]
blockspergrid = (blockspergrid_y, blockspergrid_x)

# El lanzamiento del kernel define el "for" implicito sobre toda la imagen:
# se crea una grilla de bloques/hilos y cada hilo ejecuta rgb_to_gray_cuda.

# Warmup para evitar contabilizar overhead inicial del primer kernel.
rgb_to_gray_cuda[blockspergrid, THREADS_PER_BLOCK](rgb_device, gray_device)
sobel_cuda[blockspergrid, THREADS_PER_BLOCK](gray_device, sobel_device)
cuda.synchronize()

# Conversion a GRIS y medicion de tiempo
start_gray = perf_counter()
# Mismo kernel, ahora medido: cada hilo toma su (y, x) con cuda.grid(2).
rgb_to_gray_cuda[blockspergrid, THREADS_PER_BLOCK](rgb_device, gray_device)
cuda.synchronize()
elapsed_gray = perf_counter() - start_gray

# Conversion a SOBEL y medicion de tiempo
start_sobel = perf_counter()
# El recorrido global de la imagen sigue siendo implicito por lanzamiento
# del kernel: muchos hilos ejecutan sobel_cuda en paralelo.
sobel_cuda[blockspergrid, THREADS_PER_BLOCK](gray_device, sobel_device)
cuda.synchronize()
elapsed_sobel = perf_counter() - start_sobel

# Movemos el resultado a la CPU
sobel_host = sobel_device.copy_to_host()

Image.fromarray(sobel_host).save(output_path)

total = elapsed_gray + elapsed_sobel
white_pct = white_percentage(sobel_host)

print("Modo: sobel_numba_gpu")
print(f"Input: {input_path}")
print(f"Output: {output_path}")
print(f"Ancho: {width}")
print(f"Alto: {height}")
print(f"Threads por bloque (y, x): {THREADS_PER_BLOCK}")
print(f"Porcentaje de blancos (%): {white_pct:.6f}")
print(f"Tiempo RGB->gris (segundos): {elapsed_gray:.6f}")
print(f"Tiempo Sobel (segundos): {elapsed_sobel:.6f}")
print(f"Tiempo total (segundos): {total:.6f}")  # 0.017078


from PIL import Image
import matplotlib.pyplot as plt

sobel_image = Image.open(output_path).convert("RGB")

plt.imshow(rgb_host, cmap="gray")
plt.show()

plt.imshow(gray_device, cmap="gray")
plt.show()

plt.imshow(sobel_image)
plt.show()


from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


input_path = "image.jpg"
output_path = "sobel-image.jpg"

device = torch.device("cuda")

image = Image.open(input_path).convert("L")
width, height = image.size

# Imagen en CPU como numpy, luego pasamos al tensor en GPU
gray_host = np.asarray(image, dtype=np.uint8)

# Tensor float32 en GPU con forma (1, 1, H, W) requerida por conv2d
img = torch.tensor(gray_host, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

kernel_x = torch.tensor(
    [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
    dtype=torch.float32,
    device=device,
)
kernel_y = torch.tensor(
    [[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]],
    dtype=torch.float32,
    device=device,
)

start = perf_counter()
gx = F.conv2d(img, kernel_x, padding=1)
gy = F.conv2d(img, kernel_y, padding=1)
magnitude = torch.sqrt(gx * gx + gy * gy).clamp(0.0, 255.0)
torch.cuda.synchronize()
elapsed = perf_counter() - start

# Mover resultado a CPU
sobel_host = magnitude.squeeze().to(torch.uint8).cpu().numpy()

Image.fromarray(sobel_host).save(output_path)

white_pct = float(np.count_nonzero(sobel_host == 255)) * 100.0 / sobel_host.size

print("Modo: sobel_pytorch_cuda")
print(f"Input: {input_path}")
print(f"Output: {output_path}")
print(f"Ancho: {width}")
print(f"Alto: {height}")
print(f"Porcentaje de blancos (%): {white_pct:.6f}")
print(f"Tiempo Sobel (segundos): {elapsed:.6f}")


# //////////////////////////////


from PIL import Image
import matplotlib.pyplot as plt

image = Image.open(output_path).convert("RGB")

plt.imshow(image)
