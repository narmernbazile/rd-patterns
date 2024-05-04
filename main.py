#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys



def lap5(f, h2):
  """
  Use a five-point stencil with periodic boundary conditions to approximate
  the Laplacian. The corresponding array slices for each component of the
  stencil are noted.
  """
  f = np.pad(f, 1, mode='wrap')

  left   = f[1:-1, :-2]   # shift left for f(x - h, y)
  right  = f[1:-1, 2:]    # shift right for f(x + h, y)
  down   = f[2:, 1:-1]    # shift down for f(x, y - h)
  up     = f[:-2, 1:-1]   # shift up for f(x, y + h)
  center = f[1:-1, 1:-1]  # center for f(x, y)

  fxy = (left + right + down + up - 4 * center) / h2

  return fxy


def reaction_diffusion(da: float, # diffusion rate for A
                       db: float, # diffusion rale for B
                        F: float, # feed rate
                        k: int,   # kill rate
                        n: int   = 64,         # number of cells; nxn
                        h: int   = 2,     # approximation interval
                       nt: int   = 20000, # number of timesteps to sumulate
                       dt: float = 1      # magnitude of each timestep
                        ):
  
  # initalized concentrations of chemicals A and B represented as nxn matrices
  A = np.ones((n, n), dtype='float64')
  B = np.zeros((n, n), dtype='float64')

  # initial concentrations at center 3x3 grid
  low = (n // 2) - 1
  high = (n // 2) + 2
  A[low:high, low:high] = 0.50 + np.random.uniform(0, 0.1, (3, 3))
  B[low:high, low:high] = 0.25 + np.random.uniform(0, 0.1, (3, 3))

  # iterate nt timesteps
  for n in range(nt):
    # print(f'Running {n + 1:,}/{nt:,}', end='\r')
    ABB = A * B * B
    A += (da * lap5(A, h*h) - ABB + F * (1 - A)) * dt
    B += (db * lap5(B, h*h) + ABB - B * (F + k)) * dt

  return A, B


def save_tile(matrix, out_img_path, _cmap):
  fig, ax = plt.subplots(tight_layout=True)
  plt.axis('off')
  ax.imshow(matrix, interpolation='lanczos', cmap=_cmap)
  fig.savefig(out_img_path, bbox_inches='tight', pad_inches=0)


def contrast_img(in_img_path, out_img_path, light_color, dark_color):
  input_image = Image.open(in_img_path)
  image_array = np.array(input_image)

  for i in range(len(image_array)):
    for j in range(len(image_array[i])):
      r = image_array[i][j][0]
      g = image_array[i][j][1]
      b = image_array[i][j][2]
      a = image_array[i][j][3]

      r = light_color[0] if r > 255//2 else dark_color[0]
      g = light_color[1] if g > 255//2 else dark_color[1]
      b = light_color[2] if b > 255//2 else dark_color[2]

      image_array[i][j][0] = r
      image_array[i][j][1] = g
      image_array[i][j][2] = b
      image_array[i][j][3] = a

  output_image = Image.fromarray(image_array)
  output_image.save(out_img_path)

def merge_and_contrast_img(in_imgA_path, in_imgB_path, out_img_path):
  input_imageA = Image.open(in_imgA_path)
  input_imageB = Image.open(in_imgB_path)
  imageA_array = np.array(input_imageA)
  imageB_array = np.array(input_imageB)
  if not len(imageA_array) == len(imageB_array):
    print("error: image A and image B are different dimensions")
    sys.exit()
  N = len(imageA_array)
  output_array = np.zeros((N,N,4))
  for i in range(N):
    for j in range(N):
      output_array[i][j][0] = int(imageA_array[i][j][0])  # adopt the 'reds' from image A
      output_array[i][j][1] = 0 # don't use the green channel
      output_array[i][j][2] = int(imageB_array[i][j][2])
      output_array[i][j][3] = 255 # we want this pixel to be fully opaque 

  output_image = Image.fromarray(output_array, 'RGBA')
  output_image.save(out_img_path)


def tile_img(in_img_path, out_img_path, scale_factor):
  input_image = Image.open(in_img_path)
  input_array = np.array(input_image)
  height, width, channels = input_array.shape
  output_height = scale_factor * height
  output_width = scale_factor * width
  output_array = np.zeros((output_height, output_width, channels), dtype=np.uint8)

  for i in range(output_height):
    for j in range(output_width):
      pixel = input_array[i % height][j % width]
      output_array[i][j] = pixel

  output_image = Image.fromarray(output_array)
  output_image.save(out_img_path)


def eggs(): 
  return reaction_diffusion(0.30, 0.10, 0.027, 0.040)

def standard(): 
  return reaction_diffusion(0.20, 0.10, 0.025, 0.056)

def circles(): 
  return reaction_diffusion(0.10, 0.20, 0.025, 0.056)

def texture(): 
  return reaction_diffusion(0.55, 0.08, 0.027, 0.040)
  
def big_eggs():
  return reaction_diffusion(0.10, 0.20, 0.025, 0.040)

def main():
  WHITE  = (255, 255, 255)
  BLACK  = (0, 0, 0)
  GREEN  = (0, 128, 0)
  RED    = (128, 0, 0)
  YELLOW = (255, 255, 0)
  PINK   = (255, 192, 203)

  patterns = [standard, texture, circles, eggs, big_eggs]
  for idx, pattern in enumerate(patterns):
    A, B = pattern()
    save_tile(A, f"./.rd/_tile{idx}.png", "binary")
    contrast_img(f"./.rd/_tile{idx}.png", f"./.rd/_processed_tile{idx}.png", WHITE, BLACK)
    tile_img(f"./.rd/_processed_tile{idx}.png", f"./.rd/wallpaper{idx}.png", 9)
  
  # save_tile(A, "./_tileA.png", "Reds")
  # save_tile(B, "./_tileB.png", "Blues")
  # merge_and_contrast_img("./_tileA.png", "./_tileB.png", "_processed_tile.png")
  # save_tile(A, "./_tile.png", "binary")
  # contrast_img("./_tile.png", "_processed_tile.png", PINK, RED)
  # tile_img("./_processed_tile.png", "./wallpaper.png", 8)


if __name__ == "__main__": main()