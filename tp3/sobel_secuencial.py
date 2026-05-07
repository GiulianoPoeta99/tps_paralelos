"""Version secuencial pura del filtro de Sobel para la entrega del 7 de mayo."""

from __future__ import annotations

import argparse
import math

from sobel_lib import (
    GrayImagePython,
    RgbImagePython,
    add_common_cli_args,
    load_rgb_image_python,
    run_single_method_cli,
)


def rgb_to_gray_sequential(rgb: RgbImagePython) -> GrayImagePython:
    gray = bytearray(rgb.width * rgb.height)
    data = rgb.data

    for index in range(rgb.width * rgb.height):
        base = index * 3
        r = data[base]
        g = data[base + 1]
        b = data[base + 2]
        value = int(0.299 * r + 0.587 * g + 0.114 * b)
        if value < 0:
            value = 0
        elif value > 255:
            value = 255
        gray[index] = value

    return GrayImagePython(width=rgb.width, height=rgb.height, data=gray)


def sobel_sequential(gray: GrayImagePython) -> GrayImagePython:
    width = gray.width
    height = gray.height
    data = gray.data
    edges = bytearray(width * height)

    for y in range(1, height - 1):
        row = y * width
        previous_row = (y - 1) * width
        next_row = (y + 1) * width
        for x in range(1, width - 1):
            p00 = data[previous_row + x - 1]
            p01 = data[previous_row + x]
            p02 = data[previous_row + x + 1]
            p10 = data[row + x - 1]
            p12 = data[row + x + 1]
            p20 = data[next_row + x - 1]
            p21 = data[next_row + x]
            p22 = data[next_row + x + 1]

            gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22
            gy = p00 + 2 * p01 + p02 - p20 - 2 * p21 - p22
            magnitude = int(math.sqrt(gx * gx + gy * gy))
            edges[row + x] = 255 if magnitude > 255 else magnitude

    return GrayImagePython(width=width, height=height, data=edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_args(parser)
    run_single_method_cli(
        parser=parser,
        method="secuencial",
        rgb_to_gray=rgb_to_gray_sequential,
        sobel=sobel_sequential,
        workers=1,
        load_rgb=load_rgb_image_python,
    )


if __name__ == "__main__":
    main()
