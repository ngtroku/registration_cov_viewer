
from scipy.spatial.transform import Rotation
import small_gicp
import functions
import basic_registration
import numpy as np

if __name__ == "__main__":
  binFileName = './data/000000.bin'
  rotate_params = (0, 0, 0)
  translation_params = (0, 1, 0)

  source = functions.bin_to_numpy(binFileName)
  target = functions.translation(source, rotate_params, translation_params)

  result = basic_registration.example_numpy1(target, source)
  
  downsampling_resolution = float(0.25)
  num_neighbors = int(10)
  num_threads = int(4)

  target, target_tree = small_gicp.preprocess_points(target, downsampling_resolution, num_neighbors, num_threads)
  print(target_tree)