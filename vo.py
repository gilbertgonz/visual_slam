#!/usr/bin/env python3

import os
import numpy as np
import cv2
import open3d as o3d

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import time

## TODO:
# - better user experience
# - keep digging for better estimation algorithms (loop closure?) (BA?)

class VisualOdometry():
    def __init__(self, data_dir):
        # Params
        self.K, self.P = self.load_calib(os.path.join(data_dir, 'calib.txt'))
        self.D = np.zeros(5)
        self.gt_poses = self.load_poses(os.path.join(data_dir, 'poses.txt'))
        self.image_paths = self.load_image_paths(os.path.join(data_dir, 'image_0'))

        # ORB
        self.orb = cv2.ORB_create(3000)
        
        # SIFT
        self.sift = cv2.SIFT_create()

        # FAST
        self.fast =cv2.FastFeatureDetector_create(threshold = 25, nonmaxSuppression = True)

        # Matchers
        self.bf = cv2.BFMatcher()

        index_params = dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1)
        search_params = dict(checks = 50)
        self.flann_matcher = cv2.FlannBasedMatcher(indexParams = index_params, searchParams = search_params) #indexParams = index_params, searchParams = search_params

    @staticmethod
    def load_calib(filepath):
        """
        Load calibration of camera
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('P0:'): # only read left camera parameters
                    params = np.fromstring(line.strip().split(': ')[1], dtype=np.float64, sep=' ')
                    P = np.reshape(params, (3, 4))
                    K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def load_poses(filepath):
        """
        Load GT poses
        """
        if not os.path.exists(filepath):
            return None
        
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def load_image_paths(filepath):
        """
        Load image paths
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return image_paths

    @staticmethod
    def form_transf(R, t):
        """
        Form transformation matrix from R and t
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i, detector, frame, prev_frame, ratio=0.5):
        """
        Detect and compute keypoints and descriptors from the i-1'th and i'th img
        """
        if detector == "ORB": # faster (less computational overhead), but less accurate
            print(detector)
            if frame is not None:
                kps1, features1 = self.orb.detectAndCompute(prev_frame, None)
                kps2, features2 = self.orb.detectAndCompute(frame, None) 
            else:
                kps1, features1 = self.orb.detectAndCompute(cv2.imread(self.image_paths[i - 1], cv2.IMREAD_GRAYSCALE), None)
                kps2, features2 = self.orb.detectAndCompute(cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE), None) 

            # FLANN matcher (faster, but higher chance of inaccuracies)
            matches = self.flann_matcher.knnMatch(features1, features2, k=2)

            # Filter good matches
            good = []
            for m,n in matches:
                if m.distance < ratio*n.distance:
                    good.append(m)

            # Extract filtered keypoints
            q1 = np.float32([kps1[m.queryIdx].pt for m in good ])
            q2 = np.float32([kps2[m.trainIdx].pt for m in good ])

            return q1, q2
        if detector == "SIFT": # slower (more computational overhead), but more accurate
            print(detector)
            if frame is not None:
                kps1, features1 = self.sift.detectAndCompute(prev_frame, None)
                kps2, features2 = self.sift.detectAndCompute(frame, None)
            else:
                kps1, features1 = self.sift.detectAndCompute(cv2.imread(self.image_paths[i - 1], cv2.IMREAD_GRAYSCALE), None)
                kps2, features2 = self.sift.detectAndCompute(cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE), None)

            # Brute Force matcher (slower, but lower chance of inaccuracies)
            matches = self.bf.knnMatch(features1, features2, k=2)

            # Filter good matches
            good = []
            for m,n in matches:
                if m.distance < ratio*n.distance:
                    good.append(m)
            
            # Extract filtered keypoints
            q1 = np.float32([kps1[m.queryIdx].pt for m in good ])
            q2 = np.float32([kps2[m.trainIdx].pt for m in good ])

            return q1, q2
        if detector == "FAST": # not working at the moment
            print(detector)
            if frame is not None:
                kps1 = self.fast.detect(prev_frame)
                kps1 = np.array([x.pt for x in kps1], dtype=np.float32).reshape(-1, 1, 2)
            else:
                kps1 = self.fast.detect(cv2.imread(self.image_paths[i - 1], cv2.IMREAD_GRAYSCALE))
                kps1 = np.array([x.pt for x in kps1], dtype=np.float32).reshape(-1, 1, 2)

            # Calculate optical flow between frames
            kps2, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, kps1, None, winSize = (15,15), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))#, **self.lk_params)

            # Extract filtered keypoints
            q1 = kps1[status == 1]
            q2 = kps2[status == 1]

            return q1, q2
            
    def get_pose(self, q1, q2): 
        """
        Calculate transformation from keypoints
        """

        E, mask = cv2.findEssentialMat(q1, q2, self.K, cv2.RANSAC, 0.999, 1.0, None)

        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K, mask)

        return self.form_transf(R, t.squeeze())
        
    def plot(self, ax_2d, estimated_path, gt_path):
        # Clear axis
        ax_2d.clear()

        # Extract coordinates of estimated_path
        x_est = [point[0] for point in estimated_path]
        y_est = [point[1] for point in estimated_path]
        ax_2d.plot(x_est, y_est, color='blue')

        # Extract coordinates of ground truth path if exists
        if gt_path:
            x_gt = [point[0] for point in gt_path]
            y_gt = [point[1] for point in gt_path]

            ax_2d.plot(x_gt, y_gt, color='red')
            ax_2d.plot([x_est, x_gt], [y_est, y_gt], linestyle='--', color='purple', linewidth=0.2) # plot error
        
        ax_2d.legend(['Estimated path', 'Ground Truth', 'Error'])
        ax_2d.set_title('Visual Odometry')
        ax_2d.set_xlabel('X (meters)')
        ax_2d.set_ylabel('Y (meters)')
        ax_2d.grid(True)

    def plot_3d(self, ax_3d, Q):
        # Clear axis
        ax_3d.clear()

        # Extract X, Y, Z coordinates from Q
        X = Q[:, 0]
        Y = Q[:, 1]
        Z = Q[:, 2]

        # Plot points
        ax_3d.scatter(X, Y, Z, c='b', marker='o')

        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('3D Plot')

    def bundle_adjustment(self, cam_params, Qs, cam_idxs, Q_idxs, qs):
        """
        Preforms bundle adjustment

        Parameters
        ----------
        cam_params (ndarray): Initial parameters for cameras
        Qs (ndarray): The 3D points
        cam_idxs (list): Indices of cameras for image points
        Q_idxs (list): Indices of 3D points for image points
        qs (ndarray): The image points

        Returns
        -------
        residual_init (ndarray): Initial residuals
        residuals_solu (ndarray): Residuals at the solution
        solu (ndarray): Solution
        """
        # Use least_squares() from scipy.optimize to minimize the objective function
        # Stack cam_params and Qs after using ravel() on them to create a one dimensional array of the parameters
        # save the initial residuals by manually calling the objective function
        # residual_init = objective()
        # res = least_squares(.....)

        # Stack the camera parameters and the 3D points
        params = np.hstack((cam_params.ravel(), Qs.ravel()))

        # Save the initial residuals
        residual_init = self.objective(params, cam_params, cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)

        # Perform the least_squares optimization
        res = least_squares(self.objective, params, verbose=2, x_scale='jac', ftol=1e-4, method='trf', max_nfev=50,
                            args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))

        # Get the residuals at the solution and the solution
        residuals_solu = res.fun
        solu = res.x
        # normalized_cost = res.cost / res.x.size()
        # print ("\nNormalized cost with reduced points: " +  str(normalized_cost))
        return residual_init, residuals_solu, solu

    def triangulate(self, pts1, pts2, P1, P2):       
        P1 = P1[:3, :]
        P2 = P2[:3, :]

        P1 = self.K @ P1
        P2 = self.K @ P2

        # Triangulate the 3D points
        points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
        points_3D = points_4D / points_4D[3]  # Convert from homogeneous to Cartesian coordinates
        points_3D = points_3D[:3, :].T

        return points_3D.flatten()

    def generate_point_cloud(self, Q):
        # Create an Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(Q)

        # Return the point cloud object
        return point_cloud

    def update_point_cloud(self, vis, Q):
        # Generate the updated point cloud
        point_cloud = self.generate_point_cloud(Q)
        
        # Update the visualization
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

def main():
    data_dir = '/home/gilbertogonzalez/Downloads/KITTI_data_gray/dataset/sequences/14/'
    '''
    Sequences for demo:
        - sequence 09: 0-257
        - sequence 02: 0-300
    '''
    vo = VisualOdometry(data_dir)

    gt_path = []
    estimated_path = []
    time_list = []
    Q = []

    vid = None #cv2.VideoCapture('/home/gilbertogonzalez/Downloads/test.MOV')
    vid_frame = None
    prev_frame = None

    # Plot figure
    fig = plt.figure(figsize=(8, 12))
    ax_2d = fig.add_subplot(2, 1, 1)
    ax_3d = fig.add_subplot(2, 1, 2, projection='3d')

    # Adjust layout and display the plots
    fig.tight_layout()

    counter = 0
    m = 2
    while True:
        print(f"\nframe: {counter}")
        
        start = time.time()
        # Check if playing video
        if vid:
            ret, vid_frame = vid.read()
            if not ret:
                break 
            scale = 0.5
            vid_frame = cv2.resize(vid_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)                 

        # Initialize pose at start
        if counter == 0:
            cur_pose = np.eye(4)
            prev_pose = np.eye(4)
            if vid:
                prev_frame = vid_frame            
        else:
            # Detect viable matches between frames
            q1, q2 = vo.get_matches(counter, "SIFT", vid_frame, prev_frame)

            # Compute transformation between frames
            transf = np.nan_to_num(vo.get_pose(q1, q2), neginf=0, posinf=0)

            # Update current pose by multiplying inverse transformation
            cur_pose = cur_pose @ np.linalg.inv(transf)

            Q_local = []
            for u_q1, u_q2 in zip(q1, q2):
                Q_local.append(vo.triangulate(u_q1, u_q2, prev_pose, cur_pose))
                Q.append(vo.triangulate(u_q1, u_q2, prev_pose, cur_pose))
            Q_local_arr = np.array(Q_local)
            Q_local_arr_downsampled = Q_local_arr[::3]
            
            vo.plot_3d(ax_3d, Q_local_arr_downsampled)

            # Saving all 3d points to txt file
            Q_arr = np.array(Q)
            Q_arr_downsampled = Q_arr[::3]
            with open("3d_pts.txt", 'w') as file:
                np.savetxt(file, Q_arr, fmt='%f')
           


            '''
            **PENDING BUNDLE ADJUSTMENT**

            # rotation_vector, _ = cv2.Rodrigues(transf[:3, :3])
            # translation_vector = transf[:3, 3].flatten()
            # # print(rotation_vector[0][0], rotation_vector[1][0], rotation_vector[2][0], translation_vector[0], translation_vector[1], translation_vector[2],vo.K[0, 0], vo.K[0, 2], vo.K[1, 2])

            # cam_params = np.array([rotation_vector[0][0], rotation_vector[1][0], rotation_vector[2][0], 
            #                         translation_vector[0], translation_vector[1], translation_vector[2], 
            #                         vo.K[0, 0], vo.K[0, 2], vo.K[1, 2]])
            # cam_params = cam_params.reshape((1, 9))

            # residual_init, residual_minimized, opt_params = vo.bundle_adjustment(cam_params, Q, 2, (np.empty(len(Q), dtype=int)), q2)
            # print(opt_params)

            '''


            cur_x_est = cur_pose[0,3]
            cur_y_est = cur_pose[2,3]
            
            # Update estimated path 
            estimated_path.append((cur_x_est, cur_y_est))

            # Update ground truth path if exists in current data sequence
            if vo.gt_poses and vid is None:
                cur_x_gt = vo.gt_poses[counter][0,3]
                cur_y_gt = vo.gt_poses[counter][2,3]
                gt_path.append((cur_x_gt, cur_y_gt))
            
            # Plot paths
            vo.plot(ax_2d, estimated_path, gt_path)
            plt.pause(0.1)
            
            # # Scalar error value between each point, can be used for histogram or something similar
            # gt_path_arr = np.array(gt_path)
            # estimated_path_arr = np.array(estimated_path)
            # diff = np.linalg.norm(gt_path_arr - estimated_path_arr, axis=1)
            # diff_arr = np.array([diff])
            # print(f"average error: {np.mean(diff_arr)}")

            # Extract keypoints coordinates
            q1x = [q1_point[0] for q1_point in q1]
            q1y = [q1_point[1] for q1_point in q1]

            q2x = [q2_point[0] for q2_point in q2]
            q2y = [q2_point[1] for q2_point in q2]

            # Show optical flow
            if vid is None:
                frame = cv2.cvtColor(cv2.imread(vo.image_paths[counter], cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
                
                # # Draw frame axis (for debugging)
                # rvec, _ = cv2.Rodrigues(cur_pose[:3, :3])
                # tvec = np.zeros((3, 1)) # cur_pose[:3, 3]
                # cv2.drawFrameAxes(frame, vo.K, vo.D, rvec, tvec, 0.5, 2)
            else:
                frame = vid_frame
            for i in range(len(q2)):
                cv2.circle(frame, (int(q2x[i]), int(q2y[i])), 2, (0, 255, 0), -1)
                cv2.line(frame, (int(q1x[i]), int(q1y[i])), (int(q2x[i]), int(q2y[i])), (0, 0, 255), 1)
            
            cv2.imshow("VO", frame)

            # Update previous frame ever m frames
            if counter % m == 0:
                prev_frame = vid_frame

            # Update previous pose
            prev_pose = cur_pose

            # Break loop if no more images in data sequence
            if vid is None:
                if counter == (len(vo.image_paths) - 1):
                    break

            key = cv2.waitKey(1)
            if key == 27: # ESC
                break
        
        # Update counter
        counter += 1

        # Time info
        time_list.append(time.time() - start)
        time_arr = np.array([time_list])
        print(f"avg time: {np.mean(time_arr)}")

    # Clean up
    if vid:
        vid.release()
    cv2.destroyWindow("VO")

if __name__ == "__main__":
    main()