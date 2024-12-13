#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import Imu, JointState
import numpy as np
import random

#This is called UKF Odom, but its actually the ground truth extractor and sensor simulation node
class GT_Sensor_Sim:
    def __init__(self):
        # Initialize the node
        self.sub_gaz = rospy.Subscriber('/gazebo/link_states', LinkStates, self.ground_truth_callback,
                                        queue_size=10)
        self.pub_gt_twist = rospy.Publisher('/omni/ground_truth/twist', TwistStamped, queue_size = 10)
        self.pub_gt_pose = rospy.Publisher('/omni/ground_truth/pose', PoseStamped, queue_size = 10)
        self.imu_pub = rospy.Publisher('/imu_sim', Imu, queue_size = 10)
        self.wheel_encoder_pub = rospy.Publisher('/wheel_encoder_sim', JointState, queue_size = 10)    

    def ground_truth_callback(self, link_states):
        self.link_names = link_states.name
        self.link_pose = link_states.pose
        base_index = self.link_names.index("omni_base::base_footprint")
        front_left_index = self.link_names.index("omni_base::wheel_front_left_link")
        front_right_index = self.link_names.index("omni_base::wheel_front_right_link")
        rear_left_index = self.link_names.index("omni_base::wheel_rear_left_link")
        rear_right_link = self.link_names.index("omni_base::wheel_rear_right_link")

        now = rospy.get_rostime()
        gt_twist = TwistStamped()
        gt_pose = PoseStamped()

        gt_twist.header.stamp, gt_pose.header.stamp = now, now
        gt_twist.header.frame_id = "base_footprint"
        gt_pose.header.frame_id = "odom"

        gt_twist.twist = link_states.twist[base_index]
        gt_pose.pose = link_states.pose[base_index]        

        self.pub_gt_pose.publish(gt_pose)
        self.pub_gt_twist.publish(gt_twist)

        imu_data = Imu()

        imu_data.header.stamp = now
        imu_data.header.frame_id = "base_footprint"
        imu_data.angular_velocity.z = gt_twist.twist.angular.z + np.random.normal(0, 0.2)
        imu_data.angular_velocity_covariance = [0, 0, 0, 0, 0, 0, 0, 0, 0.2]
        self.imu_pub.publish(imu_data)

        #Use inverse kinematics with noise to simulate wheel encoders on the mechanum wheels
        #get robot dimensions from urdf
        wheel_radius = 0.0762
        wheel_pair_separation = 0.488
        wheel_separation = 0.44715
        wheel_width = 0.05

        b = 0.5*(wheel_separation + wheel_width)
        a = 0.5*(wheel_radius + wheel_pair_separation)

        velocity_vector = np.array([[gt_twist.twist.linear.x], [gt_twist.twist.linear.y], [gt_twist.twist.angular.z]])
        inverse_kinematic_model = (1/wheel_radius)*np.array([[1, -1, -(a+b)], [1, 1, a+b], [1, 1, -(a+b)], [1, -1, a+b]])

        #omega wheels is a 4x1 column vector of the angular velocities of the wheels, the order is fl, fr, rl, rr
        omega_wheels = inverse_kinematic_model@velocity_vector

        #wheel_encoder_sim now takes the angular velocity of the wheels from the IK and 
        wheel_encoder_sim = omega_wheels + np.random.normal(0, 1.5, size=(4,1))

        encoder_sim = JointState()
        encoder_sim.header.stamp = now
        
        encoder_sim.name = ["front_left", "front_right", "rear_left", "rear_right"]


        encoder_sim.velocity = [wheel_encoder_sim[0], wheel_encoder_sim[1], wheel_encoder_sim[2], wheel_encoder_sim[3]]

        self.wheel_encoder_pub.publish(encoder_sim)




def main():
    rospy.init_node('GT_Sensor_Sim')
    sensor_sim_node = GT_Sensor_Sim()
    rospy.loginfo('starting Sensor Sim and GT Publisher')
    rospy.spin()
    rospy.loginfo('done')

if __name__ == '__main__':
    main()