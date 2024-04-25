
from scipy.spatial.transform import Rotation
import small_gicp
import functions
import basic_registration
import numpy as np

binFileName = './data/000000.bin'
sampling_rate = 0.7 #点群全体から抽出する点数の割合を決定
scale_x = 1 #正規分布の分散
scale_y = 1

original_points = functions.bin_to_numpy(binFileName) #バイナリからnumpy.array形式で点群データを取得
sampled_cloud1, sampled_cloud2 = functions.random_sampling(original_points, sampling_rate) #ランダムサンプリングされた点群を生成

source_raw_numpy, noise_x1, noise_y1 = functions.points_noise(sampled_cloud1, 0, 0)
target_raw_numpy, noise_x2, noise_y2 = functions.points_noise(sampled_cloud2, scale_x, scale_y)
print("noise x:{} noise y:{}".format(noise_x2[0], noise_y2[0]))

target, target_tree = small_gicp.preprocess_points(target_raw_numpy, downsampling_resolution=0.25)
source, source_tree = small_gicp.preprocess_points(source_raw_numpy, downsampling_resolution=0.25)

result = small_gicp.align(target, source, target_tree)
result = small_gicp.align(target, source, target_tree, result.T_target_source)
print(result)

factors = [small_gicp.GICPFactor()]
rejector = small_gicp.DistanceRejector()

sum_H = np.zeros((6, 6))
sum_b = np.zeros(6)
sum_e = 0.0

for i in range(source.size()):
  succ, H, b, e = factors[0].linearize(target, source, target_tree, result.T_target_source, i, rejector)
  if succ:
    sum_H += H
    sum_b += b
    sum_e += e

assert np.max(np.abs(result.H - sum_H) / result.H) < 0.05 # test


