
import factor
import sys

if __name__ == "__main__":
    file_name1 = input("Please set data file(.csv or .bin) >> ")
    file_name1 = str(file_name1) #文字列型に変換
    
    sampling_rate = 0.5 #点群全体から抽出する点数の割合を決定
    scale_rotation = 3 #回転移動の分散
    scale_translation = 1 #並行移動の分散

    if file_name1[-4:] == ".bin":
        source_raw_numpy, target_raw_numpy, rotation_x, rotation_y, rotation_z, noise_x, noise_y = factor.preprocess(file_name1, sampling_rate, scale_rotation, scale_translation, file_format = "bin", lidar_model="VLP-16")
    elif file_name1[-4:] == ".csv":
        source_raw_numpy, target_raw_numpy, rotation_x, rotation_y, rotation_z, noise_x, noise_y = factor.preprocess(file_name1, sampling_rate, scale_rotation, scale_translation, file_format = "bin", lidar_model="VLP-16")
    else:
        print("set .csv file or .bin file")
        sys.exit()

    points_source1, importance_factor1 = factor.calc_factor(source_raw_numpy, target_raw_numpy)

    print("rotate x:{:.3f} rotate y:{:.3f} rotate z:{:.3f} noise x:{:.3f} noise y:{:.3f}".format(rotation_x[0], rotation_y[0], rotation_z[0], noise_x[0], noise_y[0]))
    save_name = input("Please set save path (.csv) >> ")

    if save_name[-4:] == ".csv":
        factor.postprocess([points_source1], [importance_factor1], [str(save_name)])
    else:
        print("save format is .csv only")

    

