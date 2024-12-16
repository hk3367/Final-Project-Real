A full implementation of an Unscented Kalman filter for the purpose of odometry of a Mecanum Drive holonomic mobile robot. 

The Mechanum drive robot in question is Tiago Omni-Base from PAL Robotics. They created a version of their robot built for ROS and Gazebo.
Here is the URL to their installation instructions as well as their instructions to launch an empty world with their robot spawned.

https://wiki.ros.org/Robots/TIAGo-OMNI-base

The following 2 nodes were part of project to simulate the effects of Unscented Kalman Filter based odometry node for the complex mechanum wheeled holonomic robot.
They are descibed below.

The GT Sensor Sim nodes simulates the effects of an IMU and wheel encoders as those sensors aren't actually a part of the robot description. 
It does this by taking the ground truth pose and twist of the robot from Gazebo and then republishing the ground truth body angular velocity with added sensor noise.
It also uses the robots current body twist as well as its knowledge of the inverse kinematics of the robot to solve for the wheel velocity then similarly add noise and republish it.

The other node is the main point of the package. It subscribes to the sensor topics and then uses an Unscented Kalman filter to produce odometry.
Combining the knowledge of the robot kinematics with proprioceptive sensor data. There is also an EKF implementation in the node for the purpose of comparison between the two
as they are two different ways to expand the Classic Kalman Filter to nonlinear problems.
