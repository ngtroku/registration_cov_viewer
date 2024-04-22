import numpy as np
import open3d as o3d

def bin_to_numpy(bin_file): # .bin file to numpy array (N x 3)
    data = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4)).astype(np.float64)
    return data[:, 0:3]

def translation(array, rotate_params, translation_params): 
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

def points_noise(array, scale_x, scale_y):
    rng = np.random.default_rng()
    noise_x = rng.normal(0, scale_x, 1)
    noise_y = rng.normal(0, scale_y, 1)

    array_noised = translation(array, (0, 0, 0), (noise_x[0], noise_y[0], 0))
    return array_noised