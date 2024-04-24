#!/usr/bin/env python3

import os
import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import time

## TODO:
# - better user experience
# - keep digging for better estimation algorithms (loop closure?) (BA?)

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.D = np.zeros(5)
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.image_paths = self._load_image_paths(os.path.join(data_dir, 'image_0'))

        # ORB
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        # SIFT
        self.descriptor = cv2.SIFT_create()
        self.matcher = cv2.DescriptorMatcher_create("BruteForce")

    @staticmethod
    def _load_calib(filepath):
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
    def _load_poses(filepath):
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
    def _load_image_paths(filepath):
        """
        Load image paths
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return image_paths

    @staticmethod
    def _form_transf(R, t):
        """
        Form transformation matrix from R and t
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i, detector, frame, prev_frame, ratio=0.4):
        """
        Detect and compute keypoints and descriptors from the i-1'th and i'th img
        """
        if detector == "ORB": # faster (less computational overhead), less accurate
            if frame is not None:
                kps1, features1 = self.orb.detectAndCompute(prev_frame, None)
                kps2, features2 = self.orb.detectAndCompute(frame, None) 
            else:
                kps1, features1 = self.orb.detectAndCompute(cv2.imread(self.image_paths[i - 1], cv2.IMREAD_GRAYSCALE), None)
                kps2, features2 = self.orb.detectAndCompute(cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE), None) 

            matches = self.flann.knnMatch(features1, features2, k=2)
            # Filter good matches
            good = []
            for m,n in matches:
                if m.distance < ratio*n.distance:
                    good.append(m)

            # Extract filtered keypoints
            q1 = np.float32([kps1[m.queryIdx].pt for m in good ])
            q2 = np.float32([kps2[m.trainIdx].pt for m in good ])

            return q1, q2
        if detector == "SIFT": # slower (more computational overhead), more accurate
            if frame is not None:
                kps1, features1 = self.descriptor.detectAndCompute(prev_frame, None)
                kps2, features2 = self.descriptor.detectAndCompute(frame, None)
            else:
                kps1, features1 = self.descriptor.detectAndCompute(cv2.imread(self.image_paths[i - 1], cv2.IMREAD_GRAYSCALE), None)
                kps2, features2 = self.descriptor.detectAndCompute(cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE), None)

            matches = self.matcher.knnMatch(features1, features2, 2)

            # Filter good matches
            good = []
            for m,n in matches:
                if m.distance < ratio*n.distance:
                    good.append(m)
            
            # Extract filtered keypoints
            q1 = np.float32([kps1[m.queryIdx].pt for m in good ])
            q2 = np.float32([kps2[m.trainIdx].pt for m in good ])

            return q1, q2
            
    def get_pose(self, q1, q2): 
        """
        Calculate transformation from keypoints
        """

        E, mask = cv2.findEssentialMat(q1, q2, self.K, cv2.RANSAC, 0.999, 1.0, None)

        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K, mask)

        return self._form_transf(R, t.squeeze())
        
    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals
    
    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
            """
            Estimates the transformation matrix

            Parameters
            ----------
            q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
            q2 (ndarray): Feature points in i'th image. Shape (n, 2)
            Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
            Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
            max_iter (int): The maximum number of iterations

            Returns
            -------
            transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
            """
            early_termination_threshold = 5

            # Initialize the min_error and early_termination counter
            min_error = float('inf')
            early_termination = 0

            for _ in range(max_iter):
                # Choose 6 random feature points
                sample_idx = np.random.choice(range(q1.shape[0]), 6)
                sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

                # Make the start guess
                in_guess = np.zeros(6)
                # Perform least squares optimization
                opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                        args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

                # Calculate the error for the optimized transformation
                error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
                error = error.reshape((Q1.shape[0] * 2, 2))
                error = np.sum(np.linalg.norm(error, axis=1))

                # Check if the error is less the the current min error. Save the result if it is
                if error < min_error:
                    min_error = error
                    out_pose = opt_res.x
                    early_termination = 0
                else:
                    early_termination += 1
                if early_termination == early_termination_threshold:
                    # If we have not fund any better result in early_termination_threshold iterations
                    break

            # Get the rotation vector
            r = out_pose[:3]
            # Make the rotation matrix
            R, _ = cv2.Rodrigues(r)
            # Get the translation vector
            t = out_pose[3:]
            # Make the transformation matrix
            transformation_matrix = self._form_transf(R, t)
            return transformation_matrix

    def plot(self, estimated_path, gt_path):
        # Extract coordinates of estimated_path
        x_est = [point[0] for point in estimated_path]
        y_est = [point[1] for point in estimated_path]
        plt.plot(x_est, y_est, color='blue')

        # Extract coordinates of ground truth path if exists
        if gt_path:
            x_gt = [point[0] for point in gt_path]
            y_gt = [point[1] for point in gt_path]

            plt.plot(x_gt, y_gt, color='red')
            plt.plot([x_est, x_gt], [y_est, y_gt], linestyle='--', color='purple', linewidth=0.25) # plot error
        
        plt.legend(['Estimated path', 'Ground Truth', 'Error'])
        plt.title('Visual Odometry')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.grid(True)
        plt.pause(0.1)
        plt.clf() 

    def reindex(self, idxs):
        keys = np.sort(np.unique(idxs))
        key_dict = {key: value for key, value in zip(keys, range(keys.shape[0]))}
        return [key_dict[idx] for idx in idxs]

    def shrink_problem(self, n, cam_params, Qs, cam_idxs, Q_idxs, qs):
        """
        Shrinks the problem to be n points

        Parameters
        ----------
        n (int): Number of points the shrink problem should be
        cam_params (ndarray): Shape (n_cameras, 9) contains initial estimates of parameters for all cameras. First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), next 3 components form a translation vector, then a focal distance and two distortion parameters.
        Qs (ndarray): Shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
        cam_idxs (ndarray): Shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
        Q_idxs (ndarray): Shape (n_observations,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
        qs (ndarray): Shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.

        Returns
        -------
        cam_params (ndarray): Shape (n_cameras, 9) contains initial estimates of parameters for all cameras. First 3 components in each row form a rotation vector (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), next 3 components form a translation vector, then a focal distance and two distortion parameters.
        Qs (ndarray): Shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
        cam_idxs (ndarray): Shape (n,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
        Q_idxs (ndarray): Shape (n,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
        qs (ndarray): Shape (n, 2) contains measured 2-D coordinates of points projected on images in each observations.
        """
        cam_idxs = cam_idxs[:n]
        Q_idxs = Q_idxs[:n]
        qs = qs[:n]
        cam_params = cam_params[np.isin(np.arange(cam_params.shape[0]), cam_idxs)]
        Qs = Qs[np.isin(np.arange(Qs.shape[0]), Q_idxs)]

        cam_idxs = self.reindex(cam_idxs)
        Q_idxs = self.reindex(Q_idxs)
        return cam_params, Qs, cam_idxs, Q_idxs, qs

    def sparsity_matrix(self, n_cams, n_Qs, cam_idxs, Q_idxs):
        """
        Create the sparsity matrix

        Parameters
        ----------
        n_cams (int): Number of cameras
        n_Qs (int): Number of points
        cam_idxs (list): Indices of cameras for image points
        Q_idxs (list): Indices of 3D points for image points

        Returns
        -------
        sparse_mat (ndarray): The sparsity matrix
        """
        
        # m = cam_idxs.size * 2  # number of residuals
        # n = n_cams * 9 + n_Qs * 3  # number of parameters
        # print("m:\n" + str(m) + "\nn:\n" + str(n))
        # sparse_mat = lil_matrix((m, n), dtype=int)
        # # Fill the sparse matrix with 1 at the locations where the parameters affects the residuals

        # i = np.arange(cam_idxs.size)
        # # Sparsity from camera parameters
        # for s in range(9):
        #     sparse_mat[2 * i, cam_idxs * 9 + s] = 1
        #     sparse_mat[2 * i + 1, cam_idxs * 9 + s] = 1
        # #print (sparse_mat)
        # # Sparsity from 3D points
        # for s in range(3):
        #     sparse_mat[2 * i, n_cams * 9 + Q_idxs * 3 + s] = 1
        #     sparse_mat[2 * i + 1, n_cams * 9 + Q_idxs * 3 + s] = 1


        m = cam_idxs.size * 2  # number of residuals
        n = n_Qs * 3  # number of parameters
        print("m:\n" + str(m) + "\nn:\n" + str(n))
        sparse_mat = lil_matrix((m, n), dtype=int)
        # Fill the sparse matrix with 1 at the locations where the parameters affects the residuals

        i = np.arange(cam_idxs.size)
        # Sparsity from camera parameters
        for s in range(9):
            sparse_mat[2 * i, cam_idxs * 9 + s] = 1
            sparse_mat[2 * i + 1, cam_idxs * 9 + s] = 1
        #print (sparse_mat)
        # Sparsity from 3D points
        for s in range(3):
            sparse_mat[2 * i, Q_idxs * 3 + s] = 1
            sparse_mat[2 * i + 1,  Q_idxs * 3 + s] = 1

        return sparse_mat

    def rotate(self, Qs, rot_vecs):
        """
        Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.

        Parameters
        ----------
        Qs (ndarray): The 3D points
        rot_vecs (ndarray): The rotation vectors

        Returns
        -------
        Qs_rot (ndarray): The rotated 3D points
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(Qs * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * Qs + sin_theta * np.cross(v, Qs) + dot * (1 - cos_theta) * v

    def objective(self, params, cam_param, n_cams, n_Qs, cam_idxs, Q_idxs, qs):
        """
        The objective function for the bundle adjustment

        Parameters
        ----------
        params (ndarray): Camera parameters and 3-D coordinates.
        n_cams (int): Number of cameras
        n_Qs (int): Number of points
        cam_idxs (list): Indices of cameras for image points
        Q_idxs (list): Indices of 3D points for image points
        qs (ndarray): The image points

        Returns
        -------
        residuals (ndarray): The residuals
        """
        # Should return the residuals consisting of the difference between the observations qs and the reporjected points
        # Params is passed from bundle_adjustment() and contains the camera parameters and 3D points
        # project() expects an arrays of shape (len(qs), 3) indexed using Q_idxs and (len(qs), 9) indexed using cam_idxs
        # Copy the elements of the camera parameters and 3D points based on cam_idxs and Q_idxs

        # Get the camera parameters
        # cam_params = params[:n_cams * 9].reshape((n_cams, 9))
        K = np.array((718.856,0,0),(0,718.856,0),(0,0,1))

        cam_params = cam_param.reshape((n_cams,9))

        # Get the 3D points
        Qs = params.reshape((n_Qs, 3))
        # Qs = params[n_cams * 9:].reshape((n_Qs, 3))

        # Project the 3D points into the image planes
        qs_proj = self.project(Qs[Q_idxs], cam_params[cam_idxs])

        # Calculate the residuals
        residuals = (qs_proj - qs).ravel()

        #q = K* [R t] * Qs
        return residuals
        
    def project(self, Qs, cam_params):
        """
        Convert 3-D points to 2-D by projecting onto images.

        Parameters
        ----------
        Qs (ndarray): The 3D points
        cam_params (ndarray): Initial parameters for cameras

        Returns
        -------
        qs_proj (ndarray): The projectet 2D points
        """
        # Rotate the points
        qs_proj = self.rotate(Qs, cam_params[:, :3])
        # Translat the points
        qs_proj += cam_params[:, 3:6]
        # Un-homogenized the points
        qs_proj = -qs_proj[:, :2] / qs_proj[:, 2, np.newaxis]
        # Distortion
        f, k1, k2 = cam_params[:, 6:].T
        n = np.sum(qs_proj ** 2, axis=1)
        r = 1 + k1 * n + k2 * n ** 2
        qs_proj *= (r * f)[:, np.newaxis]
        # return qs_proj

        cam_params = cam_param.reshape((n_cams,9))

        # Get the 3D points
        Qs = params.reshape((n_Qs, 3))
        # Qs = params[n_cams * 9:].reshape((n_Qs, 3))

        # Project the 3D points into the image planes
        qs_proj = project(Qs[Q_idxs], cam_params[cam_idxs])

        # Calculate the residuals
        residuals = (qs_proj - qs).ravel()

        #q = K* [R t] * Qs
        return residuals

    def bundle_adjustment_with_sparsity(self, cam_params, Qs, cam_idxs, Q_idxs, qs, sparse_mat):
        """
        Preforms bundle adjustment with sparsity

        Parameters
        ----------
        cam_params (ndarray): Initial parameters for cameras
        Qs (ndarray): The 3D points
        cam_idxs (list): Indices of cameras for image points
        Q_idxs (list): Indices of 3D points for image points
        qs (ndarray): The image points
        sparse_mat (ndarray): The sparsity matrix

        Returns
        -------
        residual_init (ndarray): Initial residuals
        residuals_solu (ndarray): Residuals at the solution
        solu (ndarray): Solution
        """

        transformations = []
        for i in range(len(cam_params)):
            R ,_ = cv2.Rodrigues( cam_params[:3])
            t = cam_params[3:6]
            transformations.append(np.column_stack((R,t)))

        # Stack the camera parameters and the 3D points
        params = np.hstack((cam_params.ravel(), Qs.ravel()))
        params2 = Qs.ravel()

        # Save the initial residuals
        residual_init = self.objective(params2, cam_params.ravel() , cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)

        # Perform the least_squares optimization with sparsity
        res = least_squares(self.objective, params2, jac_sparsity=sparse_mat, verbose=2, x_scale='jac', ftol=1e-6, method='trf', max_nfev=50,
                            args=(cam_params.ravel(), cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))

        # Get the residuals at the solution and the solution
        residuals_solu = res.fun
        solu = res.x
        # normalized_cost = res.cost / res.x.size()
        # print ("\nAverage cost for each point (solution with sparsity): " +  str(normalized_cost))
        return residual_init, residuals_solu, solu

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
        residual_init = self.objective(params, cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs)

        # Perform the least_squares optimization
        res = least_squares(self.objective, params, verbose=2, x_scale='jac', ftol=1e-4, method='trf', max_nfev=50,
                            args=(cam_params.shape[0], Qs.shape[0], cam_idxs, Q_idxs, qs))

        # Get the residuals at the solution and the solution
        residuals_solu = res.fun
        solu = res.x
        # normalized_cost = res.cost / res.x.size()
        # print ("\nNormalized cost with reduced points: " +  str(normalized_cost))
        return residual_init, residuals_solu, solu

    def calc_3d(self, undistort_q1, undistort_q2, prev_pose, cur_pose):
        """
        Triangulate points from previous and current image 
        
        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image
        q2 (ndarray): Feature points in i'th image
        prev_pose: camera pose of i-1'th image frame
        cur_pose: camera pose of i'th image frame

        Returns
        -------
        Q (ndarray): 3D points seen from the i'th image
        """

        # Triangulate points from i-1'th image
        Q = cv2.triangulatePoints(prev_pose, cur_pose, undistort_q1.T, undistort_q2.T)
        # Un-homogenize
        Q = np.transpose(Q[:3] / Q[3])

        return Q
    
    def triangulate(self, undistort_img_pt0, undistort_img_pt1, world_T_cam0, world_T_cam1):
        A = np.zeros((3, 2))
        b = np.zeros((3, 1))

        p0 = np.array([undistort_img_pt0[0], undistort_img_pt0[1], 1.0])
        p1 = np.array([undistort_img_pt1[0], undistort_img_pt1[1], 1.0])

        ray0 = world_T_cam0[:3,:3] @ p0
        ray1 = world_T_cam1[:3,:3] @ p1

        A[:,0] = ray0
        A[:,1] = -ray1
        b = world_T_cam1[:3, 3] - world_T_cam0[:3, 3]

        At = np.transpose(A)
        x = np.linalg.inv(At @ A) @ At @ b

        X0 = world_T_cam0[:3, 3] + ray0*x[0]
        X1 = world_T_cam1[:3, 3] + ray1*x[1]

        return (X0 + X1)*0.5


def main():
    data_dir = '/home/gilbertogonzalez/Downloads/KITTI_data_gray/dataset/sequences/09/'
    vo = VisualOdometry(data_dir)

    gt_path = []
    estimated_path = []

    vid = None # cv2.VideoCapture('/home/gilberto/Downloads/test2.MP4')
    vid_frame = None
    prev_frame = None

    counter = 0
    while True:
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

            # Update current pose by multiplying inverse transformation
            cur_pose = cur_pose @ np.linalg.inv(transf)



            ######################################
            Q = []
            undistort_q1 = cv2.undistortPoints(q1, vo.K, vo.D)
            undistort_q2 = cv2.undistortPoints(q2, vo.K, vo.D)
            for u_q1, u_q2 in zip(undistort_q1, undistort_q2):
                Q.append(vo.triangulate(u_q1[0], u_q2[0], prev_pose, cur_pose))
            # print(undistort_q2[0])
            # print(Q[0])


            cam_params, Qs, cam_idxs, Q_idxs, qs = vo.shrink_problem

            rotation_vector, _ = cv2.Rodrigues(transf[:3, :3]).flatten()
            translation_vector = transf[:3, 3].flatten()
            
            cam_params = [rotation_vector[0], rotation_vector[1], rotation_vector[2], translation_vector[0], translation_vector[1], translation_vector[2], focal_distance, distortion_param_1, distortion_param_2]
            
            residual_init, residual_minimized, opt_params = vo.bundle_adjustment(cam_params, Q, (np.empty(len(Q), dtype=int)), Q_idxs, q2)


            #######################################


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
            vo.plot(estimated_path, gt_path)

            '''
            Scalar error value between each point, can be used for histogram or something similar

            gt_path_arr = np.array(gt_path)
            estimated_path_arr = np.array(estimated_path)
            diff = np.linalg.norm(gt_path_arr - estimated_path_arr, axis=1)
            '''

            # Extract keypoints coordinates
            q1x = [q1_point[0] for q1_point in q1]
            q1y = [q1_point[1] for q1_point in q1]

            q2x = [q2_point[0] for q2_point in q2]
            q2y = [q2_point[1] for q2_point in q2]

            # Show optical flow
            if vid is None:
                frame = cv2.cvtColor(cv2.imread(vo.image_paths[counter], cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
            else:
                frame = vid_frame
            for i in range(len(q2)):
                cv2.circle(frame, (int(q2x[i]), int(q2y[i])), 2, (0, 255, 0), -1)
                cv2.line(frame, (int(q1x[i]), int(q1y[i])), (int(q2x[i]), int(q2y[i])), (0, 0, 255), 1)
            
            cv2.imshow("VO", frame)

            # Update previous frame
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

        # print(f"time: {time.time() - start}")

    # Clean up
    if vid:
        vid.release()
    cv2.destroyWindow("VO")

if __name__ == "__main__":
    main()