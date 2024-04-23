#!/usr/bin/env python3

import os
import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

## TODO:
# - better user experience
# - keep digging for better estimation algorithms (loop closure?)

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
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

    def get_matches(self, i, detector, frame, prev_frame, ratio=0.5):
        """
        Detect and compute keypoints and descriptors from the i-1'th and i'th img
        """
        if detector == "ORB": # faster (less computational overhead), less accurate
            if frame is not None:
                kps1, features1 = self.orb.detectAndCompute(prev_frame, None)
                kps2, features2 = self.orb.detectAndCompute(frame, None) 
            else:
                print(i - 1, i)
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

        E, mask = cv2.findEssentialMat(q1, q2, self.K)

        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K, mask)

        return self._form_transf(R, t.squeeze())
        
    def plot(self, estimated_path, gt_path):
        x_est = [point[0] for point in estimated_path]
        y_est = [point[1] for point in estimated_path]
        plt.plot(x_est, y_est, color='blue')

        if gt_path:
            x_gt = [point[0] for point in gt_path]
            y_gt = [point[1] for point in gt_path]

            plt.plot(x_gt, y_gt, color='red')
            plt.plot([x_est, x_gt], [y_est, y_gt], linestyle='--', color='purple', linewidth=0.25) # plot error
        
        plt.legend(['Estimated points', 'Ground Truth', 'Error'])
        plt.title('Visual Odometry')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.grid(True)
        plt.pause(0.1)
        plt.clf() 

def main():
    data_dir = '/home/gilberto/Downloads/KITTI_data_gray/dataset/sequences/05/'
    vo = VisualOdometry(data_dir)

    gt_path = []
    estimated_path = []

    vid = None # cv2.VideoCapture('/home/gilberto/Downloads/test1.MOV')
    prev_frame = None
    vid_frame = None
    cur_pose = None
    
    counter = 0
    while True:
        if vid is not None:
            ret, vid_frame = vid.read()
            scale = 0.5
            vid_frame = cv2.resize(vid_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            if not ret:
                break      

        if cur_pose is None:
            if vo.gt_poses is not None:
                cur_pose = vo.gt_poses[counter]
            else:
                cur_pose = np.eye(4) 
            if vid is not None:
                cur_pose = np.eye(4) 
                prev_frame = vid_frame            
        else:
            q1, q2 = vo.get_matches(counter, "ORB", vid_frame, prev_frame)

            transf = np.nan_to_num(vo.get_pose(q1, q2), neginf=0, posinf=0)

            cur_pose = cur_pose @ np.linalg.inv(transf)
            cur_x_est = cur_pose[0,3]
            cur_y_est = cur_pose[2,3]
            # print(cur_x_est, cur_y_est)

            if vo.gt_poses and vid is None:
                cur_x_gt = vo.gt_poses[counter][0,3]
                cur_y_gt = vo.gt_poses[counter][2,3]
                gt_path.append((cur_x_gt, cur_y_gt))
            
            estimated_path.append((cur_x_est, cur_y_est))

            vo.plot(estimated_path, gt_path)

            '''
            Scalar error value between each point, can be used for histogram or something similar

            gt_path_arr = np.array(gt_path)
            estimated_path_arr = np.array(estimated_path)
            diff = np.linalg.norm(gt_path_arr - estimated_path_arr, axis=1)
            '''

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

            prev_frame = vid_frame

            # Break if no more images
            if vid is None:
                if counter >= len(vo.image_paths):
                    break

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
        
        counter += 1

    cv2.destroyWindow("VO")

if __name__ == "__main__":
    main()