import os
import numpy as np
import cv2
import json

class VisualOdometry():
    def __init__(self, data_dir, debug):
        # Params
        self.K, self.P   = self.load_calib(os.path.join(data_dir, 'calib.txt'))
        self.D           = np.zeros(5)
        self.gt_poses    = self.load_poses(os.path.join(data_dir, 'poses.txt'))
        self.image_paths = self.load_image_paths(data_dir)
        self.debug      = debug

        # # Import other calibration data
        # with open('/home/gilbertogonzalez/projects/visual_slam/libs/cal.json', 'r') as file:
        #     data = json.load(file)
        # # Extract K and D from the JSON data
        # self.K = np.array(data['camera_matrix'])
        # self.D = np.array(data['distortion_coefficients'])

        # ORB
        self.orb = cv2.ORB_create(3000)
        
        # SIFT
        self.sift = cv2.SIFT_create()

        # Matchers
        self.bf = cv2.BFMatcher()

        index_params       = dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1)
        search_params      = dict(checks = 50)
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
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath)) if file.endswith('.png') or file.endswith('.jpg')]
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
            if self.debug:
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
            if self.debug:
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
            
    def get_pose(self, q1, q2): 
        """
        Calculate transformation from keypoints
        """

        E, mask = cv2.findEssentialMat(q1, q2, self.K, cv2.RANSAC, 0.999, 1.0, None)

        _, R, tvec, _ = cv2.recoverPose(E, q1, q2, self.K, mask)

        return self.form_transf(R, tvec.squeeze())

    ## Version 1
    def triangulate(self, pts1, pts2, P1, P2):       
        """
        Triangulate 3d points from cooresponding 2d matching points
        """

        P1 = P1[:3, :]
        P2 = P2[:3, :]

        P1 = self.K @ P1
        P2 = self.K @ P2

        # Triangulate the 3D points
        points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
        points_3D = points_4D / points_4D[3]  # Convert from homogeneous to Cartesian coordinates
        points_3D = points_3D[:3, :].T

        return points_3D.flatten()
    
    # ## Version 2
    # def triangulate(self, pts1, pts2, pose1, pose2):
    #     # print(pose1, pose2, pts1, pts2)
        
    #     ret = np.zeros((pts1.shape[0], 4))
    #     pose1 = np.linalg.inv(pose1)
    #     pose2 = np.linalg.inv(pose2)
    #     for i, p in enumerate(zip(pts1, pts2)):
    #         A = np.zeros((4,4))
    #         A[0] = p[0][0] * pose1[2] - pose1[0]
    #         A[1] = p[0][1] * pose1[2] - pose1[1]
    #         A[2] = p[1][0] * pose2[2] - pose2[0]
    #         A[3] = p[1][1] * pose2[2] - pose2[1]
    #         _, _, vt = np.linalg.svd(A)
    #         ret[i] = vt[3]
    #     return ret

    
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
