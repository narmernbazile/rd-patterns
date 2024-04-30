#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("MacOSX") 


def laplace2(matrix):
  pass

def lap5(f, h2):
    """
    Use a five-point stencil with periodic boundary conditions to approximate
    the Laplacian. The corresponding array slices for each component of the
    stencil are noted.
    """
    f = np.pad(f, 1, mode='wrap')

    left   = f[1:-1, :-2]     # shift left for f(x - h, y)
    right  = f[1:-1, 2:]     # shift right for f(x + h, y)
    down   = f[2:, 1:-1]      # shift down for f(x, y - h)
    up     = f[:-2, 1:-1]       # shift up for f(x, y + h)
    center = f[1:-1, 1:-1]  # center for f(x, y)

    fxy = (left + right + down + up - 4 * center) / h2

    return fxy

def plot(matrix):
  fig, ax = plt.subplots(tight_layout=True)
  ax.imshow(matrix)
  fig.savefig('./fig.png')
  plt.show()

def main():
  da = 0.255  # 0.2
  db = 0.100  # 0.1
  F  = 0.030  # 0.025
  k  = 0.055  # 0.056

  n = 64
  h = 2
  h2 = h * h

  nt = 20000 # number of iterations (time-steps)
  dt = 1     # magnitude of each time-step
  
  # initalized concentrations of chemicals A and B represented as nxn matrices
  A = np.ones((n, n), dtype='float64')
  B = np.zeros((n, n), dtype='float64')

  # # inital condition - randomly initlization cells in region (0,0) to (z, z) of A and B at t = 0
  # z = 4
  # A[0:z, 0:z] = np.random.uniform(0, 0.1, (z, z))
  # B[0:z, 0:z] = np.random.uniform(0, 0.1, (z, z))

  # initial concentrations at center
  low = (n // 2) - 1
  high = (n // 2) + 2
  A[low:high, low:high] = 0.50 + np.random.uniform(0, 0.1, (3, 3))
  B[low:high, low:high] = 0.25 + np.random.uniform(0, 0.1, (3, 3))

  # iterate n timesteps
  for n in range(nt):
    # print(f'Running {n + 1:,}/{nt:,}', end='\r')
    ABB = A * B * B
    A += (da * lap5(A, h2) - ABB + F * (1 - A)) * dt
    B += (db * lap5(B, h2) + ABB - B * (F + k)) * dt

  # plot the final time step and save figure to file
  # plt.imshow(A, interpolation='lanczos', cmap="binary") # type: ignore
  plt.imshow(B, interpolation='lanczos', cmap="binary") # type: ignore
  
  plt.show()



if __name__ == "__main__": main()