from pyridescence import *
from scipy.spatial.transform import Rotation
import small_gicp
import functions
import basic_registration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_csv(pointcloud, eigenvalue, output_name):
  xyz = pd.DataFrame(pointcloud, columns = ["x","y","z"])
  value = pd.DataFrame(eigenvalue, columns=["eigen_value"])
  output_df = pd.concat([xyz, value], axis=1)
  output_df.to_csv(output_name)

def preprocess(file_name, sampling_rate, scale_rotation, scale_translation, file_format, lidar_model):
  if file_format == "csv":
    original_points = functions.csv_to_numpy(file_name, lidar_model)
  elif file_format == "bin":
    original_points = functions.bin_to_numpy(file_name)
    
  sampled_cloud1, sampled_cloud2 = functions.random_sampling(original_points, sampling_rate)
 
  source_points, rotation_x, rotation_y, rotation_z, noise_x1, noise_y1 = functions.points_noise(sampled_cloud1, 0, 0)
  target_points, rotation_x, rotation_y, rotation_z, noise_x2, noise_y2 = functions.points_noise(sampled_cloud2, scale_rotation, scale_translation)
  
  return source_points, target_points, rotation_x, rotation_y, rotation_z, noise_x2, noise_y2

def preprocess_manual(csv_file_name, sampling_rate, rotation_x, rotation_y, rotation_z, noise_x, noise_y, lidar_model="AT128"):
  original_points = functions.csv_to_numpy(csv_file_name, lidar_model)
  sampled_cloud1, sampled_cloud2 = functions.random_sampling(original_points, sampling_rate)

  source_points = functions.points_noise_manual(sampled_cloud1, 0, 0, 0, 0, 0)
  target_points = functions.points_noise_manual(sampled_cloud2, rotation_x, rotation_y, rotation_z, noise_x, noise_y)

  return source_points, target_points

def calc_factor(source_points, target_points):
  target, target_tree = small_gicp.preprocess_points(target_points, downsampling_resolution=0.25)
  source, source_tree = small_gicp.preprocess_points(source_points, downsampling_resolution=0.25)

  result = small_gicp.align(target, source, target_tree)
  result = small_gicp.align(target, source, target_tree, result.T_target_source)

  factors = [small_gicp.GICPFactor()]
  rejector = small_gicp.DistanceRejector()

  sum_H = np.zeros((6, 6))
  sum_b = np.zeros(6)
  sum_e = 0.0

  matrix_cov = np.linalg.pinv(result.H) # 全体共分散行列

  # 固有値と固有ベクトルを求める
  eigen_value, eigen_vector = np.linalg.eig(matrix_cov)
  min_eigen_value = np.argmin(eigen_value)
  min_eigen_vector = eigen_vector[min_eigen_value] # 拘束が弱い方向を指す

  list_xyz = []
  list_cov_eigen_value = []

  for i in range(source.size()):
    succ, H, b, e = factors[0].linearize(target, source, target_tree, result.T_target_source, i, rejector)
    if succ:

      point_cov = np.linalg.pinv(H)
      point_eigen_value, point_eigen_vector = np.linalg.eig(point_cov) #各点の固有値、固有ベクトルを求める
      max_eigen_value = np.argmax(eigen_value) 
      max_eigen_vector = point_eigen_vector[max_eigen_value]

      naiseki = np.dot(max_eigen_vector, min_eigen_vector)
      list_cov_eigen_value.append((naiseki.real + 1)/2)
      list_xyz.append(source_points[i])

      sum_H += H
      sum_b += b
      sum_e += e

  assert np.max(np.abs(result.H - sum_H) / result.H) < 0.05 
  return list_xyz, list_cov_eigen_value

def postprocess(list_xyz, list_cov_eigen_value, list_save_name):

  for xyz, cov_eigen_value, save_name in zip(list_xyz, list_cov_eigen_value, list_save_name):
    save_csv(np.array(xyz), np.array(cov_eigen_value), save_name)




