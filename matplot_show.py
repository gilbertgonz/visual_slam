import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the file and extract 3D points
file_path = "3d_pts.txt"
points = []
with open(file_path, 'r') as file:
    for line in file:
        x, y, z = map(float, line.split())
        points.append([x, y, z])

# Convert list of points to NumPy array
points = np.array(points)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot 3D points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()