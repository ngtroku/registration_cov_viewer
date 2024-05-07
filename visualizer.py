# どの点が拘束しているかを可視化
import open3d as o3d
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import itertools

def csv_2_array(file_name):
    df = pd.read_csv(file_name)
    pointcloud = df[["x", "y", "z"]]
    eigenvalue = df[["eigen_value"]]

    array_points, array_values = np.array(pointcloud), np.array(eigenvalue)

    return array_points, array_values

def calc_angle(xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    angle = np.degrees(np.arctan2(y, x)) + 180
    return angle

def count_eigen_score(angle_array, eigen_array, step):
    list_angle, list_score = [], []
    for i in range(int(360 / step)):
        mask = ((angle_array >= step * i) & (angle_array < step * (i+1)))
        score = np.sum(eigen_array[mask])
        list_score.append(score)
        list_angle.append(step * (i + 0.5))
    
    return list_angle, list_score

def cmap(value):
    colors_cmap = matplotlib.colormaps["jet"]
    colors = colors_cmap(value)
    colors = colors[:, :, 0:3].tolist()
    a = list(itertools.chain.from_iterable(colors))
    b = [i for i in a]
    return b

def show_plot(eigen_value, list_angle, list_score):
    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(wspace=0.25)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(eigen_value, bins=40)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xticks([0.1 * i for i in range(11)])
    ax1.set_xlabel("eigen value score")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(list_angle, list_score, marker="o")
    ax2.set_xticks([60 * i for i in range(7)])
    ax2.set_xlabel("angle(deg)")
    ax2.set_ylabel("score")
    plt.show()

if __name__ == "__main__":
    file_name = input("Please set visualize file path >> ")
    xyz, eigenvalue = csv_2_array(str(file_name))
    points_angle = calc_angle(xyz)

    list_angle, list_score = count_eigen_score(points_angle, eigenvalue, 10) # 角度情報, 重要度情報, 水平方位角範囲の広さ(deg)

    colors = cmap(eigenvalue)

    pointcloud1 = o3d.geometry.PointCloud()
    pointcloud1.points = o3d.utility.Vector3dVector(xyz)
    pointcloud1.colors = o3d.utility.Vector3dVector(colors)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0]) # 原点にxyz軸を表す矢印を表示
    o3d.visualization.draw_geometries([pointcloud1, mesh_frame])

    show_plot(eigenvalue, list_angle, list_score)
