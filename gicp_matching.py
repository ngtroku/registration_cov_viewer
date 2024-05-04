from pyridescence import *
from scipy.spatial.transform import Rotation

import small_gicp
import functions
import basic_registration
import factor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    file_name1 = "./data/000101.bin"
    csv_file_name2 = "./data/14_rouka_person_at128_2.csv"
    sampling_rate = 0.5 #点群全体から抽出する点数の割合を決定
    scale_rotation = 3 #回転移動の分散
    scale_translation = 1 #並行移動の分散

    source_raw_numpy, target_raw_numpy, rotation_x, rotation_y, rotation_z, noise_x, noise_y = factor.preprocess(file_name1, sampling_rate, scale_rotation, scale_translation, file_format = "bin", lidar_model="VLP-16")
    points_source1, importance_factor1 = factor.calc_factor(source_raw_numpy, target_raw_numpy)

    print("rotate x:{} rotate y:{} rotate z:{} noise x:{} noise y:{}".format(rotation_x[0], rotation_y[0], rotation_z[0], noise_x[0], noise_y[0]))

    #source_raw_numpy, target_raw_numpy = factor.preprocess_manual(csv_file_name2, sampling_rate, rotation_x[0], rotation_y[0], rotation_z[0], noise_x[0], noise_y[0])
    #points_source2, importance_factor2 = factor.calc_factor(source_raw_numpy, target_raw_numpy)

    factor.postprocess([points_source1], [importance_factor1], ["./result_101.csv"])

    #print(importance_factor1, importance_factor2)

