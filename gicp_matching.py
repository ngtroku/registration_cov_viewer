
import functions

import numpy as np
import sys

if __name__ == "__main__":
    file_name1 = input("Please set data file(.csv or .bin) >> ")
    file_name1 = str(file_name1) #文字列型に変換
    
    sampling_rate = 0.5 #点群全体から抽出する点数の割合を決定
    scale_rotation = 3 #回転移動の分散
    scale_translation = 1 #並行移動の分散

    # read file
    if file_name1[-4:] == ".bin":
        raw_points = functions.bin_to_numpy(file_name1)

    elif file_name1[-4:] == ".csv":
        model = input("Please set csv file format. type VLP-16 or AT128 or pcd >> ")
        raw_points = functions.csv_to_numpy(file_name1, model)

    else:
        print("set .csv file or .bin file")
        sys.exit()

    # random sampling
    sampled_points1, sampled_points2 = functions.random_sampling(raw_points, sampling_rate)

    # matching simulation
    num_matching = input("How many times do you simulate matching? >> ")
    save_name = input("Please set save path (.csv) >> ")
    counter = 1

    while counter <= int(num_matching):

        if counter == 1:
            print("iteration:{}/{}".format(counter, num_matching))
            source, target = sampled_points1, functions.points_noise(sampled_points2, scale_rotation, scale_translation)
            coordinate_origin, dot_eigen_value_origin = functions.calc_factor(source, target)
            counter += 1

        else:
            print("iteration:{}/{}".format(counter, num_matching))
            source, target = sampled_points1, functions.points_noise(sampled_points2, scale_rotation, scale_translation)
            coordinate, dot_eigen_value = functions.calc_factor(source, target)
            coordinate_origin = np.concatenate([coordinate_origin, coordinate])
            dot_eigen_value_origin = np.concatenate([dot_eigen_value_origin, dot_eigen_value])
            counter += 1

    if save_name[-4:] == ".csv":
        functions.postprocess([coordinate_origin], [dot_eigen_value_origin], [str(save_name)])
    else:
        print("save format is .csv only")
            


