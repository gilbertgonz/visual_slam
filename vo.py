#!/usr/bin/env python3
import numpy as np
import cv2
from multiprocessing import Process, Queue
import time
import os

from libs.plotter import Plotter
from libs.vo_utils import VisualOdometry

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

## TODO:
# - fix buggy image
# - keep digging for fixing 3d pointcloud SFM problem (remove obvious outliers?) (different triangulation algo?)


def main(q, debug):
    data_dir = '/home/gilbertogonzalez/Downloads/KITTI_data_gray/dataset/sequences/09/'
    '''
    Sequences for demo:
        - sequence 09: 0-257
        - sequence 02: 0-300
        - sequence 14: broken
        - sequence 20: broken
        - sequence 07: pretty good
    '''
    vo = VisualOdometry(data_dir, debug)

    gt_path = []
    estimated_path = []
    time_list = []
    Q = []
    optimized_Q = []
    Q_arr_downsampled = []
    poses = []

    vid = None #cv2.VideoCapture('/home/gilbertogonzalez/Downloads/test.MOV')
    vid_frame = None
    prev_frame = None

    counter = 0
    m = 2
    while True:
        start = time.time() # start timer for debug

        if debug:
            print(f"\nframe: {counter}")
            if ba:
                print("running BA") 
        
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
            frame = cv2.cvtColor(cv2.imread(vo.image_paths[counter], cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
            if vid:
                prev_frame = vid_frame            
        else:
            # Detect viable matches between frames            
            q1, q2 = vo.get_matches(counter, "ORB", vid_frame, prev_frame)

            # Compute transformation between frames            
            transf = np.nan_to_num(vo.get_pose(q1, q2), neginf=0, posinf=0)
            
            # Update current pose by multiplying inverse transformation
            cur_pose = cur_pose @ np.linalg.inv(transf)
            poses.append(cur_pose)
            
            # Update extrinsic vectors
            rvec_tf, _ = cv2.Rodrigues(transf[:3, :3])
            tvec_tf = transf[:3, 3]
            rvec_pose, _ = cv2.Rodrigues(cur_pose[:3, :3])
            tvec_pose = cur_pose[:3, 3]
            
            # Triangulate 3d points
            Q_local = []

            ## Version 1
            # for u_q1, u_q2 in zip(q1, q2):
            #     Q_local.append(vo.triangulate(u_q1, u_q2, prev_pose, cur_pose))
            #     Q.append(vo.triangulate(u_q1, u_q2, prev_pose, cur_pose))

            ## Version 2
            pts4d = vo.triangulate(q1, q2, prev_pose, cur_pose)
            pts4d /= pts4d[:, 3:]
            pts3d = pts4d[:, :3]

            for pt in pts3d:
                Q_local.append(pt)
                Q.append(pt)
    

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
                                       x_scale='jac', ftol=1e-1, method='trf', args=(vo.K, vo.D, q2))
                
                # Extract the optimized camera pose and 3D points from the result
                optimized_params = result.x
                optimized_cam_pose = optimized_params[:6]
                optimized_points3d = optimized_params[6:].reshape((-1, 3))

                tvec_pose = optimized_cam_pose[3:]
                optimized_Q.append(optimized_points3d)
                optimized_Q_arr = np.array(optimized_Q)

                reprj_err = vo.calc_reprojection_error(vo.K, vo.D, optimized_cam_pose[:3], optimized_cam_pose[3:], np.array(optimized_points3d), q2)

                if debug:
                    print(f"* post_ba_reprj_err: {np.mean(reprj_err)}")
                    print(f"- prev pose: {cur_pose[:3, 3]}")
                    print(f"- new pose : {optimized_cam_pose[3:]}")

            # Save all 3d points to txt file
            Q_arr = np.array(Q)
            Q_arr_downsampled = Q_arr[::2]
            if output_txt:
                with open("3d_pts.txt", 'w') as file:
                    np.savetxt(file, Q_arr, fmt='%f')
            
            # Update estimated path 
            estimated_path.append(tvec_pose)

            # Update ground truth path if exists in current data sequence
            if vo.gt_poses and vid is None:
                gt_path.append(vo.gt_poses[counter][:3, 3])
            
                # Scalar error value between each point
                diff = np.linalg.norm(np.array(gt_path) - np.array(estimated_path), axis=1)
                if debug:
                    print(f"gt error: {np.mean(diff)}")

            # Show optical flow
            if vid is None:
                frame = cv2.cvtColor(cv2.imread(vo.image_paths[counter], cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
                # cv2.drawFrameAxes(frame, vo.K, vo.D, rvec, tvec, 0.5, 2)
            else:
                frame = vid_frame

            for i in range(len(q2)):
                cv2.circle(frame, (int(q2[i][0]), int(q2[i][1])), 2, (0, 255, 0), -1)
                cv2.line(frame, (int(q1[i][0]), int(q1[i][1])), (int(q2[i][0]), int(q2[i][1])), (0, 0, 255), 1)
                        
            cv2.imshow("Image", frame)

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

        q.put((estimated_path, gt_path, Q_arr_downsampled, poses, frame))

        # Update counter
        counter += 1

        # Time info
        time_list.append(time.time() - start)
        time_arr = np.array([time_list])
        if debug:
            print(f"avg time: {np.mean(time_arr)}")

        # Check if plotter process is dead
        if not plot_process.is_alive():
            os._exit(-1)

    # Clean up
    if vid:
        vid.release()
    cv2.destroyWindow("VO")

def plotter_target(q):
    '''
    Plot target function
    '''
    plotter = Plotter(q)
    # plotter.plot_opencv()
    # plotter.plot()
    plotter.plot_pang()

if __name__ == "__main__":
    debug = False
    output_txt = False
    ba = False

    q = Queue()

    try:
        # Start plot process
        plot_process = Process(target=plotter_target, args=(q,))
        plot_process.start()

        main(q, debug)

    finally:
        if plot_process.is_alive():
            plot_process.terminate()
        plot_process.join()
