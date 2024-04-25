#! /usr/bin/env python3
# import os
# import sys
import numpy as np

from PIL import Image

IN_IMG_PATH  = './fig.png'
OUT_IMG_PATH = './out.png'

def main():
  # white: (255, 255,255), black: (0, 0, 0)
  input_image = Image.open(IN_IMG_PATH)
  image_array = np.array(input_image)

  for i in range(len(image_array)):
      for j in range(len(image_array[i])):
        # print(image_array[i])
        r = image_array[i][j][0]
        g = image_array[i][j][1]
        b = image_array[i][j][2]
        a = image_array[i][j][3]

        r = 255 if r > 255//2 else 0
        g = 255 if g > 255//2 else 0
        b = 255 if b > 255//2 else 0

        image_array[i][j][0] = r
        image_array[i][j][1] = g
        image_array[i][j][2] = b
        image_array[i][j][3] = a

  output_image = Image.fromarray(image_array)
  output_image.save(OUT_IMG_PATH)

if __name__ == "__main__": main()