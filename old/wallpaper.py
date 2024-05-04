#! /usr/bin/env python3
import numpy as np
from PIL import Image

IN_IMG_PATH  = './tile.png'
OUT_IMG_PATH = './wallpaper.png'
C = 32

def main(): 
  input_image = Image.open(IN_IMG_PATH)
  input_array = np.array(input_image)
  height, width, channels = input_array.shape
  output_height = C * height
  output_width = C * width
  output_array = np.zeros((output_height, output_width, channels), dtype=np.uint8)

  for i in range(output_height):
    for j in range(output_width):
      pixel = input_array[i % height][j % width]
      output_array[i][j] = pixel

  output_image = Image.fromarray(output_array)
  output_image.save(OUT_IMG_PATH)



if __name__ == '__main__': main()