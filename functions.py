import numpy as np
import pandas as pd
import open3d as o3d
import small_gicp

def bin_to_numpy(bin_file): # .bin file to numpy array (N x 3)
    data = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4)).astype(np.float64)
    return data[:, 0:3]

def csv_to_numpy(file_name, lidar_type): # .csv file to numpy array (N x 3)
    df = pd.read_csv(file_name)
    if lidar_type == "VLP-16":
        point_cloud = df[["Points_0", "Points_1", "Points_2"]]
        return np.array(point_cloud)
    elif lidar_type == "AT128":
        point_cloud = df[["x(m)", "y(m)", "z(m)"]]
        return np.array(point_cloud)
    elif lidar_type == "pcd":
        point_cloud = df[["x", "y", "z"]]
        return np.array(point_cloud)

def translation(array, rotate_params, translation_params): # 回転移動, 並進移動
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(array)

    R = point_cloud.get_rotation_matrix_from_xyz(rotate_params)
    point_cloud.rotate(R, center=(0, 0, 0))
    
    point_cloud.translate(translation_params)
    new_cloud = np.asarray(point_cloud.points)

    return new_cloud

def spoofing_sim(array, min_spoofing, max_spoofing, delay_m, mode): # spoofing simulation (support removal, relay attack)
    angle = np.degrees(np.arctan2(array[:, 1], array[:, 0])) + 180

    if mode == "remove":
        mask_index = np.where(((angle > min_spoofing) & (angle < max_spoofing)))
        removed_x = np.delete(array[:, 0], list(mask_index[0]))
        removed_y = np.delete(array[:, 1], list(mask_index[0]))
        removed_z = np.delete(array[:, 2], list(mask_index[0]))
        removed_array = np.vstack((removed_x, removed_y, removed_z)).T

        return removed_array

    elif mode == "relay":
        mask = ((angle > min_spoofing) & (angle < max_spoofing))
        c = np.cos(np.radians(angle))
        s = np.sin(np.radians(angle))
        x = array[:, 0]
        y = array[:, 1]

        temp_x = x[mask] - delay_m * c[mask]
        array[mask, 0] = temp_x

        temp_y = y[mask] - delay_m * s[mask]
        array[mask, 1] = temp_y

        return array
    
    else:
        return array

def random_sampling(array, sample_rate): # random sampling

    num_sample = int(array.shape[0] * sample_rate)

    indices_1 = np.random.choice(array.shape[0], num_sample, replace=False)
    sampled_array1 = array[indices_1]

    indices_2 = np.random.choice(array.shape[0], num_sample, replace=False)
    sampled_array2 = array[indices_2]

    return sampled_array1, sampled_array2

def points_noise(array, scale_translation):
    rng = np.random.default_rng()

    # x, y, z 軸方向の各点に正規分布からサンプリングされたノイズを定義
    noise_x = rng.normal(0, scale_translation, array.shape[0])
    noise_y = rng.normal(0, scale_translation, array.shape[0])
    noise_z = rng.normal(0, scale_translation, array.shape[0])

    # ノイズを与える
    array_noised = array.copy()
    array_noised[:, 0] += noise_x
    array_noised[:, 1] += noise_y
    array_noised[:, 2] += noise_z

    return array_noised

def points_noise_manual(array, rotation_x, rotation_y, rotation_z, noise_x, noise_y): # 自分で設定した値で剛体変換
    array_noised = translation(array, (rotation_x, rotation_y, rotation_z), (noise_x, noise_y, 0))
    return array_noised

def calc_factor(source_points, target_points):
    # update : ヘッセ行列を直接使用(2024 5/8)

    source, source_tree = small_gicp.preprocess_points(source_points, downsampling_resolution=0.4)
    target, target_tree = small_gicp.preprocess_points(target_points, downsampling_resolution=0.4)

    result = small_gicp.align(target, source, target_tree)
    result = small_gicp.align(target, source, target_tree, result.T_target_source)

    factors = [small_gicp.GICPFactor()]
    rejector = small_gicp.DistanceRejector()

    # initialize
    sum_H = np.zeros((6, 6))
    sum_b = np.zeros(6)
    sum_e = 0.0
    
    eigen_value, eigen_vector = np.linalg.eig(result.H)
    global_max_value = np.argmax(eigen_value)
    global_max_vector = eigen_vector[:, global_max_value] # 最も拘束が弱い方向

    list_xyz = []
    list_cov_eigen_value = []

    for i in range(source.size()):
        succ, H, b, e = factors[0].linearize(target, source, target_tree, result.T_target_source, i, rejector)
        if succ:
            point_eigen_value, point_eigen_vector = np.linalg.eig(H) #各点の固有値、固有ベクトルを求める
            local_min_value = np.argmin(eigen_value) 
            local_min_vector = point_eigen_vector[:, local_min_value] # 最も拘束が強い固有ベクトル
            naiseki = np.dot(global_max_vector, local_min_vector)
            list_cov_eigen_value.append((naiseki.real + 1) / 2) # 値を0から1に範囲に制限
            list_xyz.append(source_points[i])

            sum_H += H
            sum_b += b
            sum_e += e
    
    return np.array(list_xyz), np.array(list_cov_eigen_value)

def save_csv(pointcloud, eigenvalue, output_name):
  xyz = pd.DataFrame(pointcloud, columns = ["x","y","z"])
  value = pd.DataFrame(eigenvalue, columns=["eigen_value"])
  output_df = pd.concat([xyz, value], axis=1)
  output_df.to_csv(output_name)

def postprocess(list_xyz, list_cov_eigen_value, list_save_name):
  for xyz, cov_eigen_value, save_name in zip(list_xyz, list_cov_eigen_value, list_save_name):
    save_csv(np.array(xyz), np.array(cov_eigen_value), save_name)
    
        
