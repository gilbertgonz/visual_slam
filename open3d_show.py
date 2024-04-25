import numpy as np
import open3d as o3d

# Read the 3D points from the file
file_path = "3d_pts.txt"
points = np.loadtxt(file_path)

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Initialize and show
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)
vis.get_render_option().background_color = [0, 0, 0]
vis.add_geometry(pcd)
vis.run()
