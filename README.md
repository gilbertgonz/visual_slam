# Visual Odometry with KITTI dataset
Experimenting VO algorithms with KITTI dataset, **still in progress**

![](assets/results.gif)


## To run:
1. Install [docker](https://docs.docker.com/engine/install/)

2. Build:
```
$ docker build -t visual_odom .
```

3. Run:
```
$ xhost +local:docker
$ docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix visual_odom
```