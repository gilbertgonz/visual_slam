import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plotter():
    def __init__(self, q):
        self.q = q

        # Plot figure
        fig = plt.figure(figsize=(8, 12))
        self.ax_2d = fig.add_subplot(2, 1, 1)
        self.ax_3d = fig.add_subplot(2, 1, 2, projection='3d')


    def plot(self):
        while True:
            ## 2D Plot
            # Clear axis
            self.ax_2d.clear()
            
            while not self.q.empty():
                est, gt = self.q.get()

            # Extract coordinates of estimated_path
            x_est = [point[0] for point in est]
            y_est = [point[2] for point in est]
            self.ax_2d.plot(x_est, y_est, color='blue')

            # Extract coordinates of ground truth path if exists
            if gt:
                x_gt = [point[0] for point in gt]
                y_gt = [point[2] for point in gt]

                self.ax_2d.plot(x_gt, y_gt, color='red')
                self.ax_2d.plot([x_est, x_gt], [y_est, y_gt], linestyle='--', color='purple', linewidth=0.2) # plot error
            
            self.ax_2d.legend(['Estimated path', 'Ground Truth', 'Error'])
            self.ax_2d.set_title('Visual Odometry')
            self.ax_2d.set_xlabel('X (meters)')
            self.ax_2d.set_ylabel('Y (meters)')
            self.ax_2d.grid(True)

            ## 3D Plot
            if len(est) > 0:
                # Extract latest X, Y, Z coordinates from est
                X = est[-1][0]
                Y = est[-1][2]
                Z = est[-1][1]

                if gt:
                    # Extract latest X, Y, Z coordinates from gt
                    X2 = gt[-1][0]
                    Y2 = gt[-1][2]
                    Z2 = gt[-1][1]

                # Plot points
                self.ax_3d.scatter(X, Y, Z, c='b', marker='o')
                self.ax_3d.scatter(X2, Y2, Z2, c='r', marker='o')

                self.ax_3d.set_xlabel('X')
                self.ax_3d.set_ylabel('Y')
                self.ax_3d.set_zlabel('Z')
                self.ax_3d.set_title('3D Plot')

            # Update plot
            plt.pause(0.1)