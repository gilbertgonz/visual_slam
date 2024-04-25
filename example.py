import open3d as o3d
import numpy as np
import time

# Global settings.
dt = 3e-2 # to add new points each dt secs.
t_total = 10 # total time to run this script.
n_new = 10 # number of points that will be added each iteration.

#---
# 1st, using extend. Run non-blocking visualization.

# create visualizer and window.
vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=640)

# initialize pointcloud instance.
pcd = o3d.geometry.PointCloud()
# *optionally* add initial points
points = np.random.rand(10, 3)
pcd.points = o3d.utility.Vector3dVector(points)
# include it in the visualizer before non-blocking visualization.
vis.add_geometry(pcd)

exec_times = []

current_t = time.time()
t0 = current_t

while current_t - t0 < t_total:

    previous_t = time.time()

    while current_t - previous_t < dt:
        s = time.time()

        # Options (uncomment each to try it out):
        # 1) extend with ndarrays.
        pcd.points.extend(np.random.rand(n_new, 3))

        # 2) extend with Vector3dVector instances.
        # pcd.points.extend(
        #     o3d.utility.Vector3dVector(np.random.rand(n_new, 3)))

        # 3) other iterables, e.g
        # pcd.points.extend(np.random.rand(n_new, 3).tolist())

        vis.update_geometry(pcd)

        current_t = time.time()
        exec_times.append(time.time() - s)

    vis.poll_events()
    vis.update_renderer()

print(f"Using extend\t\t\t# Points: {len(pcd.points)},\n"
      f"\t\t\t\t\t\tMean execution time:{np.mean(exec_times):.5f}")

vis.destroy_window()

# ---
# 2nd, using remove + create + add PointCloud. Run non-blocking visualization.

# create visualizer and window.
vis = o3d.visualization.Visualizer()
vis.create_window(height=480, width=640)

# initialize pointcloud instance.
pcd = o3d.geometry.PointCloud()
points = np.random.rand(10, 3)
pcd.points = o3d.utility.Vector3dVector(points)
vis.add_geometry(pcd)

exec_times = []

current_t = time.time()
t0 = current_t
previous_t = current_t

while current_t - t0 < t_total:

    previous_t = time.time()

    while current_t - previous_t < dt:
        s = time.time()

        # remove, create and add new geometry.
        vis.remove_geometry(pcd)
        pcd = o3d.geometry.PointCloud()
        points = np.concatenate((points, np.random.rand(n_new, 3)))
        pcd.points = o3d.utility.Vector3dVector(points)
        vis.add_geometry(pcd)

        current_t = time.time()
        exec_times.append(time.time() - s)

    current_t = time.time()

    vis.poll_events()
    vis.update_renderer()

print(f"Without using extend\t# Points: {len(pcd.points)},\n"
      f"\t\t\t\t\t\tMean execution time:{np.mean(exec_times):.5f}")

vis.destroy_window()