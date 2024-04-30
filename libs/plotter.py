import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
                # Extract latest X, Y, Z coordinates from est and plot
                X = est[-1][0]
                Y = est[-1][2]
                Z = est[-1][1]
                self.ax_3d.scatter(X, Y, Z, c='b', marker='o')

                if gt:
                    # Extract latest X, Y, Z coordinates from gt and plot
                    X2 = gt[-1][0]
                    Y2 = gt[-1][2]
                    Z2 = gt[-1][1]
                    self.ax_3d.scatter(X2, Y2, Z2, c='r', marker='o')

                self.ax_3d.set_xlabel('X')
                self.ax_3d.set_ylabel('Y')
                self.ax_3d.set_zlabel('Z')
                self.ax_3d.set_title('3D Plot')

            # Update plot
            plt.pause(0.1)

    def plot_opencv(self):
        while True:
            ## 2D Plot
            while not self.q.empty():
                est, gt = self.q.get()

            height, width = 800, 800  # Define image dimensions
            cx , cy = width/2, height/2

            img = np.zeros((height, width, 3), dtype=np.uint8)  # Create black image

            # Extract estimated path coordinates
            while not self.q.empty():
                est, gt = self.q.get()

            # Extract coordinates of estimated path
            if est:
                for i in range(len(est)):
                    cv2.circle(img, (int(est[i][0] + cx), int(est[i][2] + cy/2)), 3, (255, 0, 0), -1)

                cv2.putText(img, f"Est Pos: X: {round(est[-1][0], 3)}, Y: {round(est[-1][2], 3)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1, cv2.LINE_AA)

            # Extract coordinates of ground truth path if exists
            if gt:
                for i in range(len(gt)):
                    cv2.circle(img, (int(gt[i][0] + cx), int(gt[i][2] + cy/2)), 3, (0, 0, 255), -1)

                cv2.putText(img, f"GT Pos: X: {round(gt[-1][0], 3)}, Y: {round(gt[-1][2], 3)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)     

                diff = np.linalg.norm(np.array(gt) - np.array(est), axis=1)  
                cv2.putText(img, f"Avg error: {round(np.mean(diff), 3)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA) 
            
            cv2.imshow('Visual Odometry', img)
            cv2.waitKey(1)  # Wait for a short time to refresh the window