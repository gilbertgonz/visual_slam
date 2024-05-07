import cv2
import numpy as np

import OpenGL.GL as gl
import pangolin

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Plotter():
    def __init__(self, q):
        self.q = q

        # MatPlot figure
        fig = plt.figure(figsize=(8, 12))
        self.ax_2d = fig.add_subplot(2, 1, 1)
        self.ax_3d = fig.add_subplot(2, 1, 2, projection='3d')

        # Pangolin params
        window_w, window_h = 800, 1000
        
        pangolin.CreateWindowAndBind('Visual Odometry', window_h, window_w)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(window_h, window_w, 420, 420, window_h/2, window_w/2, 0.2, 1000),
            pangolin.ModelViewLookAt(15, 50, 0, 0, 0, 25, pangolin.AxisDirection.AxisY))
        handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -window_h/window_w)
        self.dcam.SetHandler(handler)

        # Panel
        panel = pangolin.CreatePanel('ui')
        panel.SetBounds(0.0, 0.12, 0.0, 130/640.)

        self.pts_button = pangolin.VarBool('ui.Show Points', value=True, toggle=False)
        self.show_pts = False
        self.gt_button = pangolin.VarBool('ui.Show GT', value=True, toggle=False)
        self.show_gt = False
        self.exit_button = pangolin.VarBool('ui.Exit', value=False, toggle=False)

    def plot(self):
        while True:
            ## 2D Plot
            # Clear axis
            self.ax_2d.clear()
            
            while not self.q.empty():
                est, gt, Q, cur_pose, img = self.q.get()

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
                est, gt, Q, cur_pose, img = self.q.get()

            height, width = 800, 800  # Define image dimensions
            cx , cy = width/2, height/2

            img = np.zeros((height, width, 3), dtype=np.uint8)  # Create black image

            # Extract coordinates of estimated path
            if est:
                for i in range(len(est)):
                    cv2.circle(img, (int(est[i][0] + cx), int((est[i][2])*-1 + cy)), 3, (255, 0, 0), -1)

                cv2.putText(img, f"Est Pos: X: {round(est[-1][0], 3)}, Y: {round(est[-1][2], 3)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1, cv2.LINE_AA)

            # Extract coordinates of ground truth path if exists
            if gt:
                for i in range(len(gt)):
                    cv2.circle(img, (int(gt[i][0] + cx), int((gt[i][2])*-1 + cy)), 3, (0, 0, 255), -1)

                cv2.putText(img, f"GT Pos: X: {round(gt[-1][0], 3)}, Y: {round(gt[-1][2], 3)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)     
                
                diff = np.linalg.norm(np.array(gt) - np.array(est), axis=1)  
                cv2.putText(img, f"Avg error: {round(np.mean(diff), 3)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA) 
            else:
                cv2.putText(img, f"No GT available", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv2.LINE_AA)     

            cv2.imshow('Visual Odometry', img)
            cv2.waitKey(1)  # Wait for a short time to refresh the window

    def plot_pang(self):        
        while True:
            while not self.q.empty():
                est, gt, Q, poses, img = self.q.get() 

            flipped_img = cv2.flip(img, 0) 
            Q_reshaped = np.array(Q).reshape(-1, 3)*-1
            if gt:
                gt_reshaped = np.array(gt).reshape(-1, 3)
                gt_reshaped[:, 0] *= -1

            # Clear plot
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            self.dcam.Activate(self.scam) 
            # self.update_model_view(self.scam.GetModelViewMatrix(), cur_pose)
            
            # Inversing pose to better match image perspective
            new_poses = []
            for i, pose in enumerate(poses):
                R = pose[:3, :3]
                
                R_inv = np.linalg.inv(R)
                
                cur_pose_inv = np.eye(4)
                cur_pose_inv[:3, :3] = R_inv
                cur_pose_inv[:3, 3] = pose[:3, 3]
                
                cur_pose_inv[0, 3] = pose[0, 3] * -1
                
                new_poses.append(cur_pose_inv)

            # Draw camera pose
            for pose in new_poses:
                gl.glLineWidth(1)
                gl.glColor3f(0.0, 1.0, 1.0)
                pangolin.DrawCamera(pose, 0.5, 0.75, 0.8)

            # Handle points toggle
            if pangolin.Pushed(self.pts_button):
                self.show_pts = not self.show_pts
            if self.show_pts:
                # Draw 3d points
                gl.glPointSize(1)
                gl.glColor3f(0.3099, 0.3099,0.184314)
                pangolin.DrawPoints(Q_reshaped)

            # Handle gt toggle
            if pangolin.Pushed(self.gt_button):
                self.show_gt = not self.show_gt
            if self.show_gt:
                if gt:
                    # Draw gt line
                    gl.glLineWidth(2)
                    gl.glColor3f(0.0, 1.0, 0.0)
                    pangolin.DrawLine(gt_reshaped)

            # Handle exit toggle
            if pangolin.Pushed(self.exit_button):
                print("\nThanks for watching!\n")
                exit()

            pangolin.FinishFrame()

    # Function to update the model view matrix based on camera pose
    def update_model_view(self, model_view_matrix, camera_pose):
        # Extract camera position and orientation from the pose
        camera_position = camera_pose[0][:3]
        look_at_point = camera_position + camera_pose[0][3:]  # Assuming camera pose includes orientation
        print(f"{look_at_point[0] = }")

        # Update the model view matrix to match the camera pose
        model_view_matrix.Load(pangolin.ModelViewLookAt(
            camera_position[0], camera_position[1], camera_position[2],  # camera position
            0, 0, 0,  # look at point
            pangolin.AxisDirection.AxisY))  # camera up direction