# Visual SLAM
Welcome to my VSLAM project! This project aims to track the camera's motion and estimate its pose using ORB feature detection (also tested using SIFT, but way too computationally expensive...), triangulate 3D points from consecutive frames, and optimize the estimated camera poses and 3D points through [bundle adjustment](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html) (BA). The resulting trajectory and point cloud are visualized using [Pangolin](https://github.com/gilbertgonz/pangolin). As can be seen in the gif below, I am utilizing short snips from the well-known [KITTI dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

![](assets/results.gif)


## To run:
1. Install [docker](https://docs.docker.com/engine/install/)

2. Clone repo

3. Build:
```
$ docker build -t visual_odom .
```

4. Run:
```
$ xhost +local:docker
$ docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix visual_odom
```

## Future work:
This project is still on-going, things I still plan to implement/test:
- Test performance on aerial robot (drone) image data
- Implement a dedicated backend thread for proper execution of BA
- Deploy on hardware platform and implement real-time operation
