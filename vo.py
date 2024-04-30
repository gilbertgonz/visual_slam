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

    def get_matches(self, i, detector, frame, prev_frame, ratio=0.45):
        """
        Detect and compute matching keypoints and descriptors from the i-1'th and i'th img
        """
        if detector == "ORB": # faster (less computational overhead), but less accurate
            if debug:
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
            if debug:
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
            if debug:
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

        _, R, tvec, _ = cv2.recoverPose(E, q1, q2, self.K, mask)

        return self.form_transf(R, tvec.squeeze())
    
    def refine_pose(self, init_3dpoints, img_pts):
        ret, rvec, tvec = cv2.solvePnP(init_3dpoints, img_pts, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE) #cv2.SOLVEPNP_EPNP
        if ret:
            R, _ = cv2.Rodrigues(rvec)
            
            return self.form_transf(R, tvec.squeeze())
        else:
            if debug:
                print("\n\nCound not refine pose")
            exit()
             
        
    def plot(self, ax_2d, estimated_path, gt_path):
        # Clear axis
        ax_2d.clear()

        # Extract coordinates of estimated_path
        x_est = [point[0] for point in estimated_path]
        y_est = [point[2] for point in estimated_path]
        ax_2d.plot(x_est, y_est, color='blue')

        # Extract coordinates of ground truth path if exists
        if gt_path:
            x_gt = [point[0] for point in gt_path]
            y_gt = [point[2] for point in gt_path]

            ax_2d.plot(x_gt, y_gt, color='red')
            ax_2d.plot([x_est, x_gt], [y_est, y_gt], linestyle='--', color='purple', linewidth=0.2) # plot error
        
        ax_2d.legend(['Estimated path', 'Ground Truth', 'Error'])
        ax_2d.set_title('Visual Odometry')
        ax_2d.set_xlabel('X (meters)')
        ax_2d.set_ylabel('Y (meters)')
        ax_2d.grid(True)

    def plot_3d(self, ax_3d, est, gt):
        # Clear axis
        # ax_3d.clear()

        # Extract X, Y, Z coordinates from est
        X = est[0]
        Y = est[2]
        Z = est[1]

        # Extract X, Y, Z coordinates from gt
        X2 = gt[0]
        Y2 = gt[2]
        Z2 = gt[1]

        # Plot points
        ax_3d.scatter(X, Y, Z, c='b', marker='o')
        ax_3d.scatter(X2, Y2, Z2, c='r', marker='o')

        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('3D Plot')

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

    def calc_reprojection_error(self, K, D, rvec, tvec, objp, imgpoints):
        """Calculates reprojection error using 2D Euclidean distance

        Parameters
        ----------
            K: 3x3 intrinsics matrix
            D: distortion coefficients
            rvecs: rotation vectors
            tvecs: translation vectors
            objpoints: numpy array of 3d object points
            imgpoints: numpy array of 2d points

        Returns
        -------
            reproj_err: list of numpy vector of size len(corners)
            reproj_points: list of numpy array of size len(corners)x2
        """

        reproj_err = []

        proj_pts, _ = cv2.projectPoints(objp, rvec, tvec, K, D)

        proj_pts = proj_pts.squeeze()
        img_pts = imgpoints.squeeze()

        # 2D Euclidean distance
        err_sq = np.power(proj_pts - img_pts, 2)
        err_sq = np.sum(err_sq, axis=1)
        err = np.sqrt(err_sq)

        reproj_err.append(err)

        return np.array(reproj_err).ravel()
    
    def bundle_adjustment_residuals(self, params, K, D, imgpoints):
        """
        Residual function for bundle adjustment between consecutive frames.

        Parameters
        ----------
        params: numpy array
            Flattened array containing the current camera pose (rotation vector and translation vector)
            and the 3D point coordinates from the previous frame to be optimized.
        K: numpy array
            3x3 camera intrinsic matrix.
        D: numpy array
            Distortion coefficients.
        cur_imgpoints: numpy array
            2D image points observed in the current frame.

        Returns
        -------
        residuals: numpy array
            Array of residuals (reprojection errors) for all observed image points in the current frame.
        """
        # Extract the current camera pose and 3D points from the parameter vector
        rvec, tvec = params[:3], params[3:6]
        points3d = params[6:].reshape((-1, 3))

        reproj_errors = self.calc_reprojection_error(K, D, rvec, tvec, points3d, imgpoints)

        return reproj_errors

def main():
    data_dir = '/home/gilberto/Downloads/KITTI_data_gray/dataset/sequences/02/'
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
    optimized_Q = []

    vid = None #cv2.VideoCapture('/home/gilberto/Downloads/test.MOV')
    vid_frame = None
    prev_frame = None

    # Plot figure
    fig = plt.figure(figsize=(8, 12))
    ax_2d = fig.add_subplot(2, 1, 1)
    ax_3d = fig.add_subplot(2, 1, 2, projection='3d')

    # # Initialize Open3d visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=True)
    # vis.get_render_option().background_color = [0, 0, 0]

    # pcd_gt = o3d.geometry.PointCloud()
    # pcd_gt.paint_uniform_color([1, 0, 0])

    # pcd_est = o3d.geometry.PointCloud()
    # pcd_est.paint_uniform_color([0, 0, 1])

    counter = 0
    m = 2
    while True:
        if debug:
            print(f"\nframe: {counter}")
            if ba:
                print("running BA") 
        
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
            q1, q2 = vo.get_matches(counter, "ORB", vid_frame, prev_frame)

            # Compute transformation between frames
            transf = np.nan_to_num(vo.get_pose(q1, q2), neginf=0, posinf=0)

            # Refine transformation
            init_3dpts = []
            for u_q1, u_q2 in zip(q1, q2):
                init_3dpts.append(vo.triangulate(u_q1, u_q2, prev_pose, cur_pose))
            init_3dpts = np.array(init_3dpts)

            transf = np.nan_to_num(vo.refine_pose(init_3dpts, q2), neginf=0, posinf=0)
            print("")

            # Update current pose by multiplying inverse transformation
            cur_pose = cur_pose @ np.linalg.inv(transf)

            # Update extrinsic vectors vector
            rvec_tf, _ = cv2.Rodrigues(transf[:3, :3])
            tvec_tf = transf[:3, 3]

            rvec_pose, _ = cv2.Rodrigues(cur_pose[:3, :3])
            tvec_pose = cur_pose[:3, 3]
            # print(tvec_pose)

            Q_local = []
            for u_q1, u_q2 in zip(q1, q2):
                Q_local.append(vo.triangulate(u_q1, u_q2, prev_pose, cur_pose))
                Q.append(vo.triangulate(u_q1, u_q2, prev_pose, cur_pose))
            Q_local_arr = np.array(Q_local)
            Q_local_arr_downsampled = Q_local_arr[::3]

            if debug:
                reprj_err = vo.calc_reprojection_error(vo.K, vo.D, rvec_pose, tvec_pose, Q_local_arr, q2)
                print(f"* reprj_err: {np.mean(reprj_err)}")

            # Bundle adjustment using least squares
            if ba:       
                camera_params = np.array([rvec_pose[0][0], rvec_pose[1][0], rvec_pose[2][0], 
                                        tvec_pose[0], tvec_pose[1], tvec_pose[2]]).reshape((1, 6))

                initial_params = np.hstack((camera_params.ravel(), Q_local_arr.ravel()))

                result = least_squares(vo.bundle_adjustment_residuals, initial_params, jac_sparsity=None, verbose=2, 
                                       x_scale='jac', ftol=1e-3, method='trf', args=(vo.K, vo.D, q2))
                
                # Extract the optimized camera pose and 3D points from the result
                optimized_params = result.x
                optimized_cam_pose = optimized_params[:6]
                optimized_points3d = optimized_params[6:].reshape((-1, 3))

                tvec_pose = optimized_cam_pose[3:]
                optimized_Q.append(optimized_points3d)
                optimized_Q_arr = np.concatenate(optimized_Q, axis=0)

                reprj_err = vo.calc_reprojection_error(vo.K, vo.D, optimized_cam_pose[:3], optimized_cam_pose[3:], np.array(optimized_points3d), q2)

                if debug:
                    print(f"* post_ba_reprj_err: {np.mean(reprj_err)}")
                    print(f"- prev pose: {cur_pose[:3, 3]}")
                    print(f"- new pose : {optimized_cam_pose[3:]}")

            if output_txt:
                # Saving all 3d points to txt file
                Q_arr = np.array(Q)
                Q_arr_downsampled = Q_arr[::3]
                with open("3d_pts.txt", 'w') as file:
                    np.savetxt(file, optimized_Q_arr, fmt='%f')
            
            # Update estimated path 
            estimated_path.append(tvec_pose)

            # Update ground truth path if exists in current data sequence
            if vo.gt_poses and vid is None:
                gt_path.append(vo.gt_poses[counter][:3, 3])
            
            # Plot
            vo.plot(ax_2d, estimated_path, gt_path)
            vo.plot_3d(ax_3d, estimated_path[-1], gt_path[-1])

                               

            '''
            **DRAWING IN OPEN3D**

            # Open3D points
            pcd_est.points.extend(np.array(estimated_path))
            pcd_gt.points.extend(np.array(gt_path))

            vis.add_geometry(pcd_est)
            vis.add_geometry(pcd_gt)

            vis.update_geometry(pcd_est)
            vis.update_geometry(pcd_gt)  

            # Get the standard camera parameters
            standardCameraParametersObj = vis.get_view_control().convert_to_pinhole_camera_parameters()
            standardCameraParametersObj.extrinsic = cur_pose

            # Create camera lines visualization
            cameraLines = o3d.geometry.LineSet.create_camera_visualization(
                intrinsic=standardCameraParametersObj.intrinsic,
                extrinsic=standardCameraParametersObj.extrinsic
            )
            vis.clear_geometries()
            vis.add_geometry(cameraLines)
            '''           
            
            
            # Scalar error value between each point
            diff = np.linalg.norm(np.array(gt_path) - np.array(estimated_path), axis=1)
            if debug:
                print(f"gt error: {np.mean(diff)}")

            # Show optical flow
            if vid is None:
                frame = cv2.cvtColor(cv2.imread(vo.image_paths[counter], cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
                # cv2.drawFrameAxes(frame, vo.K, vo.D, rvec, tvec, 0.5, 2)
            else:
                frame = vid_frame

            for i in range(len(q2)):
                cv2.circle(frame, (int(q2[i][0]), int(q2[i][1])), 2, (0, 255, 0), -1)
                cv2.line(frame, (int(q1[i][0]), int(q1[i][1])), (int(q2[i][0]), int(q2[i][1])), (0, 0, 255), 1)
            
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
        
        # Update plot
        plt.pause(0.1)

        # # Update open3d render
        # vis.poll_events()
        # vis.update_renderer()
        
        # Update counter
        counter += 1

        # Time info
        time_list.append(time.time() - start)
        time_arr = np.array([time_list])
        if debug:
            print(f"avg time: {np.mean(time_arr)}")

    # Clean up
    if vid:
        vid.release()
    # vis.destroy_window()
    cv2.destroyWindow("VO")

if __name__ == "__main__":
    debug = False
    output_txt = False
    ba = False

    main()