# GraspSampler
GraspSampler is a framework that enables you to efficiently sample millions of parallel-jaw grasps around a target object object point cloud. One can quickly generate a dataset that contains numerous grasps and their quality scores.

This repository contains tutorials on how to sample and visualize grasps, render object point clouds, and a sample of the grasp dataset.


<!-- Grasp Definition -->

## Installation
GraspSampler was created using Ubuntu 18.04 and python 3.8. To install GraspSampler on your local machine follow [these intstructions.](https://github.com/patrickeala/Grasp-Sampler/blob/main/documentation/installation.md)

Once you have a working environment for GraspSampler, you can try the tutorials in the next section.

## Tutorials
The following are tutorials on how to use the different features of GraspSampler.

### Tutorial 1: Gripper Definitiion
Run the gripper tutorial:
```python -m tutorials.gripper```
![Gripper Visualization](documentation/pictures/gripper.png)

### Tutorial 2: Adding Target Object 
Run the target object tutorial:
```python -m tutorials.object```
![Object and Gripper](documentation/pictures/object.png)


### Tutorial 3: Visualizing Object Point Clouds
Run the point cloud tutorial:
```python -m tutorials.pc_manager```

### Tutorial 4: Sampling Grasps
Run the grasp sampler tutorial:
```python -m tutorials.graspsampler```
![Example of a sampled successful grasp](documentation/pictures/graspsampler.png)


## XXX
