#!/usr/bin/env python3

import numpy as np
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import json

def get_matches(frame, prev_frame, ratio=0.4):
    """
    Detect and compute keypoints and descriptors from the i-1'th and i'th img
    """
    # prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGBA2GRAY)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

    kps1, features1 = sift.detectAndCompute(prev_frame, None)
    kps2, features2 = sift.detectAndCompute(frame, None)

    # Brute Force matcher (slower, but lower chance of inaccuracies)
    matches = matcher.knnMatch(features1, features2, k=2)

    # Filter good matches
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)
    
    # Extract filtered keypoints
    q1 = np.float32([kps1[m.queryIdx].pt for m in good ])
    q2 = np.float32([kps2[m.trainIdx].pt for m in good ])

    return q1, q2
        
def get_pose(q1, q2, K): 
    """
    Calculate transformation from keypoints
    """

    E, mask = cv2.findEssentialMat(q1, q2, K, cv2.RANSAC, 0.999, 1.0, None)

    _, R, t, _ = cv2.recoverPose(E, q1, q2, K, mask)

    return form_transf(R, t.squeeze())

def form_transf(R, t):
    """
    Form transformation matrix from R and t
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def plot_3d(ax_3d, Q):
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

def triangulate(pts1, pts2, P1, P2):
    # # Convert the projection matrices to the camera coordinate system
    
    P1 = P1[:3, :]
    P2 = P2[:3, :]

    P1 = K @ P1
    P2 = K @ P2

    # Triangulate the 3D points
    points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3D = points_4D / points_4D[3]  # Convert from homogeneous to Cartesian coordinates
    points_3D = points_3D[:3, :].T

    return points_3D.flatten()

def main(vid_frame, prev_frame, cur_pose, prev_pose, ax_3d):
    # Detect viable matches between frames
    q1, q2 = get_matches(vid_frame, prev_frame)

    # Compute transformation between frames
    transf = np.nan_to_num(get_pose(q1, q2, K), neginf=0, posinf=0)

    # Update current pose by multiplying inverse transformation
    cur_pose = cur_pose @ np.linalg.inv(transf)

    undistort_q1 = cv2.undistortPoints(q1, K, D)
    undistort_q2 = cv2.undistortPoints(q2, K, D)
    for u_q1, u_q2 in zip(undistort_q1, undistort_q2):
        Q.append(triangulate(u_q1[0], u_q2[0], prev_pose, cur_pose))

    Q_arr = np.array(Q)
    Q_arr_downsampled = Q_arr[::3]
    with open("3d_pts.txt", 'w') as file:
        np.savetxt(file, Q_arr_downsampled, fmt='%f')

    # Plot
    plot_3d(ax_3d, Q_arr)
    plt.pause(0.1)

    # Extract keypoints coordinates
    q1x = [q1_point[0] for q1_point in q1]
    q1y = [q1_point[1] for q1_point in q1]

    q2x = [q2_point[0] for q2_point in q2]
    q2y = [q2_point[1] for q2_point in q2]

    for i in range(len(q2)):
        cv2.circle(vid_frame, (int(q2x[i]), int(q2y[i])), 2, (0, 255, 0), -1)
        cv2.line(vid_frame, (int(q1x[i]), int(q1y[i])), (int(q2x[i]), int(q2y[i])), (0, 0, 255), 1)
    
    # # Draw frame axis (for debugging)
    # rvec, _ = cv2.Rodrigues(cur_pose[:3, :3])
    # tvec = cur_pose[:3, 3] #np.zeros((3, 1))
    # cv2.drawFrameAxes(vid_frame, K, D, rvec, tvec, 0.5, 2)

    cv2.imshow("vid_frame", vid_frame)
    cv2.waitKey(1)

    return cur_pose


if __name__ == "__main__":
    # SIFT
    sift = cv2.SIFT_create()

    # Matchers
    matcher = cv2.BFMatcher()

    vid = cv2.VideoCapture('/home/gilbertogonzalez/Downloads/test.MOV')

    # Load intrinsics
    calibration_path = "calibration/iphone12.json"
    with open(calibration_path, 'r') as json_file:
        calibration_params = json.load(json_file)
    D = np.array(calibration_params["distortion_coefficients"])
    K = np.array(calibration_params["camera_matrix"])
    
    # Plot figure
    fig = plt.figure()
    ax_3d = fig.add_subplot(111, projection='3d')
    
    counter = 0
    save_every_n_frames = 1
    Q = []
    while True:
        ret, vid_frame = vid.read()
        scale = 0.8
        vid_frame = cv2.resize(vid_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)   
        
        if counter == 0:
            cur_pose = np.eye(4)
            prev_pose = np.eye(4)

            prev_frame = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY) 
        else:
            cur_pose = main(vid_frame, prev_frame, cur_pose, prev_pose, ax_3d)
            prev_pose = cur_pose

            if counter % save_every_n_frames == 0:
                prev_frame = vid_frame

        counter += 1
