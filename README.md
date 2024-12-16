A full implementation of an Unscented Kalman filter for the purpose of odometry of a Mecanum Drive mobile robot. 

The GT Sensor Sim nodes simulatesthe effects of an IMU and wheel encoders as those sensors aren't actually a part of the robot description. 
It does this by taking the ground truth pose and twist of the robot from Gazebo and then republishing the ground truth body angular velocity with added sensor noise.
It also uses the robots current body twist as well as its knowledge of the inverse kinematics of the robot to solve for the wheel velocity then similarly add noise and republish it.

The other node is the main point of the package. It subscribes to the sensor topics and then uses an Unscented Kalman filter to produce odometry.
Combining the knowledge of the robot kinematics with proprioceptive sensor data. There is also an EKF implementation in the node for the purpose of comparison between the two
as they are two different ways to expand the Classic Kalman Filter to nonlinear problems.
