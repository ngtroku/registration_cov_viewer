#!/usr/bin/python3
import numpy as np
from scipy.spatial.transform import Rotation
from pyridescence import *

import basic_registration
import functions

# Define a callback for UI rendering
def ui_callback():
  # In the callback, you can call ImGui commands to create your UI.
  # Here, we use "DragFloat" and "Button" to create a simple UI.

  global angle
  _, angle = imgui.drag_float('angle', angle, 0.01)

  if imgui.button('close'):
    viewer.close()

# Create a viewer instance (global singleton)
viewer = guik.LightViewer.instance()

angle = 0.0

# Register a callback for UI rendering
viewer.register_ui_callback('ui', ui_callback)

# Set pointcloud
binFileName = './data/000000.bin'
sampling_rate = 0.5 #点群全体から抽出する点数の割合を決定
scale_x = 2 #正規分布の分散
scale_y = 2

original_points = functions.bin_to_numpy(binFileName) #バイナリからnumpy.array形式で点群データを取得
sampled_cloud1, sampled_cloud2 = functions.random_sampling(original_points, sampling_rate) #ランダムサンプリングされた点群を生成

noised_1 = functions.points_noise(sampled_cloud1, scale_x, scale_y)
noised_2 = functions.points_noise(sampled_cloud2, scale_x, scale_y)

registration = basic_registration.example_numpy1(noised_1, noised_2)

reg_h = registration.H
inv_reg_h = np.linalg.inv(reg_h)

# Set covariance
sphere_matrix = np.identity(4)
sphere_matrix[0:3, 0:3] = inv_reg_h[0:3, 0:3] * 5e7
sphere_matrix[:, 3] = registration.T_target_source[:, 3]

# calculate eigenvalue and eigenvector
eigen = np.linalg.eig(inv_reg_h)
eigen_value = eigen[0]
eigen_vector = eigen[1]

min_eigen_value = np.argmin(eigen_value)
min_eigen_vector = eigen_vector[min_eigen_value]

line_vector = min_eigen_vector[3:5]

while viewer.spin_once():
  viewer.update_drawable('wire_sphere', glk.primitives.wire_sphere(), guik.FlatColor(0.1, 0.7, 1.0, 1.0, sphere_matrix))
  viewer.update_points("pc1", noised_1, guik.FlatBlue())
  viewer.update_points("pc2", noised_2, guik.FlatRed())
  viewer.update_thin_lines("min_eigenvector", line_vector, false, guik.FlatGreen())
