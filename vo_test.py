import os
import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import plotting

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'image_0'))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        """
        Load calibration of camera
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('P0:'):
                    params = np.fromstring(line.strip().split(': ')[1], dtype=np.float64, sep=' ')
                    P = np.reshape(params, (3, 4))
                    K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Load GT poses
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Load images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Form transformation matrix from R and t
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        Detect and compute keypoints and descriptors from the i-1'th and i'th img
        """

        keypoints1, descriptors1 = self.orb.detectAndCompute(self.images[i - 1], None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(self.images[i], None)

        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)

        q1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ])
        q2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ])

        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculate transformation from keypoints
        """

        ess, mask = cv2.findEssentialMat(q1, q2, self.K)

        R, t = self.decomp_essential_mat(ess, q1, q2)

        return self._form_transf(R,t)

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix into R and t
        """

        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1,np.ndarray.flatten(t))
        T2 = self._form_transf(R2,np.ndarray.flatten(t))
        T3 = self._form_transf(R1,np.ndarray.flatten(-t))
        T4 = self._form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate(( self.K, np.zeros((3,1)) ), axis = 1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)
            
        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)
        if (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)

def main():
    data_dir = '/home/gilberto/Downloads/KITTI_data_gray/dataset/sequences/07/'
    vo = VisualOdometry(data_dir)

    gt_path = []
    estimated_path = []

    for i, gt_pose in enumerate(tqdm(vo.gt_poses)):
        if i == 0:
            cur_pose = gt_pose
        else:
            if len(gt_path) != 0 and len(estimated_path) != 0:
                x_est = [point[0] for point in estimated_path]
                y_est = [point[1] for point in estimated_path]

                x_gt = [point[0] for point in gt_path]
                y_gt = [point[1] for point in gt_path]

                plt.scatter(x_est, y_est, color='blue', marker='o', linewidths=1, label='Estimated points')
                plt.scatter(x_gt, y_gt, color='red', marker='o', linewidths=1, label='Ground Truth')

                plt.title('Visual Odometry')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True)
                plt.legend()
                plt.pause(0.1)
                plt.clf() 

            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))

            cur_x_est = cur_pose[0,3]
            cur_y_est = cur_pose[2,3]

            cur_x_gt = gt_pose[0,3]
            cur_y_gt = gt_pose[2,3]
        
            gt_path.append((cur_x_gt, cur_y_gt))
            estimated_path.append((cur_x_est, cur_y_est))

            q1x = [q1_point[0] for q1_point in q1]
            q1y = [q1_point[1] for q1_point in q1]

            q2x = [q2_point[0] for q2_point in q2]
            q2y = [q2_point[1] for q2_point in q2]

            # Show optical flow
            img = cv2.cvtColor(vo.images[i], cv2.COLOR_GRAY2BGR)
            for i in range(len(q2)):
                # Keypoints from the previous frame
                cv2.circle(img, (int(q1x[i]), int(q1y[i])), 2, (0, 255, 0), -1)
                # Motion vectors between matched keypoints
                cv2.line(img, (int(q1x[i]), int(q1y[i])), (int(q2x[i]), int(q2y[i])), (0, 0, 255), 1)
            
            cv2.imshow("VO", img)

            key = cv2.waitKey(1)
            if cv2.waitKey(1) == 27:  # ESC
                break

    cv2.destroyWindow("VO")

    # plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")

if __name__ == "__main__":
    main()
