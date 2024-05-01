# どの点が拘束しているかを可視化
import open3d as o3d
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib as plt
import itertools

def csv_2_array(file_name):
    df = pd.read_csv(file_name)
    pointcloud = df[["x", "y", "z"]]
    eigenvalue = df[["eigen_value"]]
    return np.array(pointcloud), np.array(eigenvalue)

def cmap(value):
    colors_cmap = matplotlib.colormaps["Blues"]
    colors = colors_cmap(value)
    colors = colors[:, :, 0:3].tolist()
    a = list(itertools.chain.from_iterable(colors))
    b = [i for i in a]
    return b
    """
    colors = []
    for i in range(value.shape[0]):
        colors.append([0, 0, value[i][0]/2])
    return colors
    """

xyz, eigenvalue = csv_2_array("./visualize_rouka_person.csv")
colors = cmap(eigenvalue)

pointcloud1 = o3d.geometry.PointCloud()
pointcloud1.points = o3d.utility.Vector3dVector(xyz)
pointcloud1.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pointcloud1])
