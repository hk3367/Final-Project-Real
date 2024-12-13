#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist, Point, Quaternion
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64
import math
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf import transformations

class UKF_Odometry:
    def __init__(self):
        rospy.loginfo("Initializing node...")

        self.ukf_odom_pub = rospy.Publisher('/ukf_odom', Odometry, queue_size=10)
        self.ekf_odom_pub = rospy.Publisher('/ekf_odom', Odometry, queue_size=10)
        self.ukf_error_pub = rospy.Publisher('/ukf_error', Float64, queue_size=10)
        self.ekf_error_pub = rospy.Publisher('/ekf_error', Float64, queue_size=10)

        # Subscribers for topics
        imu_sub = Subscriber('/imu_sim', Imu)
        wheel_encoder_sub = Subscriber('/wheel_encoder_sim', JointState)
        cmd_vel_sub = Subscriber('/mobile_base_controller/cmd_vel', Twist)
        gt_pose_sub = Subscriber('/omni/ground_truth/pose', PoseStamped)
        gt_twist_sub = Subscriber('/omni/ground_truth/twist', TwistStamped)

        # Synchronize the topics
        self.filter_sync = ApproximateTimeSynchronizer([imu_sub, wheel_encoder_sub, cmd_vel_sub], queue_size=10, slop=5, allow_headerless=True)
        self.filter_sync.registerCallback(self.filter_callback)

        self.gt_sync = ApproximateTimeSynchronizer([gt_pose_sub, gt_twist_sub], queue_size=10, slop=1.5, allow_headerless=True)
        self.gt_sync.registerCallback(self.error_callback)

        rospy.loginfo("Subscribers created and callback registered.")

        self.ukf_state = np.zeros((6,1))
        self.ukf_state_cov = 0.1*np.eye(6)

        self.ekf_state = np.zeros((6,1))
        self.ekf_state_cov = 0.1*np.eye(6)

        rospy.loginfo("init done")

        self.prev_time = None

    def error_callback(self, gt_pose, gt_twist):

        x_gt = gt_pose.pose.position.x
        y_gt = gt_pose.pose.position.y
        
        ukf_position_error = math.sqrt((self.ukf_state[0]-x_gt)**2 + (self.ukf_state[1] - y_gt)**2)
        ekf_position_error = math.sqrt((self.ekf_state[0]-x_gt)**2 + (self.ekf_state[1] - y_gt)**2)

        ukf_error = Float64()
        ekf_error = Float64()

        ukf_error.data = ukf_position_error
        ekf_error.data = ekf_position_error

        self.ukf_error_pub.publish(ukf_position_error)
        self.ekf_error_pub.publish(ekf_position_error)
        


    def filter_callback(self, imu_data, wheel_encoder_data, cmd_vel):
        #rospy.loginfo("entering callback")
        current_t = imu_data.header.stamp
        current_t_sec = current_t.to_sec()
        if self.prev_time is None:
            self.prev_time = current_t_sec
            return
        
        dt = current_t_sec - self.prev_time
        self.prev_time = current_t_sec

        measurement = np.zeros(5).T
        measurement[1:] = wheel_encoder_data.velocity
        measurement[0] = imu_data.angular_velocity.z
        z_cov = 3*np.eye(5)
        z_cov[0,0] = imu_data.angular_velocity_covariance[8]

        #find EKF prediction and Predicted covariance to compare with UKF
        EKF_pred, EKF_pred_cov = self.predict(self.ekf_state, cmd_vel, dt)

        #define dimension of state for use in UKF
        n = 6
        
        #define common parameter values for UKF
        alpha = 1e-3
        k = 3
        beta = 2
        lambda_value = 7 - n

        #need to define the sigma points that will be used to reconstuct predicted mean and covariance
        sigma = np.zeros((2*n+1, n))        
        cov_sqrt = np.linalg.cholesky(np.array(self.ukf_state_cov, dtype=np.float64))
        #rospy.loginfo(EKF_pred_cov)

        #first sigma point is just the current state
        sigma[0] = self.ukf_state.T
        for i in range(n):
            sigma[i+1] = self.ukf_state.T + (math.sqrt((n-k)+lambda_value)*cov_sqrt[:, i]).T
    
   
        for i in range(n):
            sigma[n+i+1] = self.ukf_state.T - (math.sqrt((n-k)+lambda_value)*cov_sqrt[:, i]).T

        mean_weights = 1/(2*(n+lambda_value))*np.ones(2*n+1)
        cov_weights = np.copy(mean_weights)
        
        mean_weights[0] = lambda_value/(lambda_value+n)
        cov_weights[0] = lambda_value/(lambda_value+n)
        weighted_pred = np.zeros((2*n+1, n))
        weighted_cov_pred = []

        mean_pred, _ = self.predict(self.ukf_state, cmd_vel, dt)

        for i in range(len(sigma)):
            pred, _ = self.predict(sigma[i], cmd_vel, dt)
            weighted_pred[i] = (mean_weights[i]*pred).T       
            distance = mean_pred-pred
            A = np.outer(distance, distance)
            weighted_cov_pred.append(cov_weights[i]*A)

        #rospy.loginfo(weighted_pred)
        pred_cov =np.zeros((6,6))
        for matrix in weighted_cov_pred:
            pred_cov += matrix
        
        pred_state = np.sum(weighted_pred, axis=0)
        UKF_posterior, UKF_cov_posterior = self.innovation(pred_state, pred_cov, measurement, z_cov)
        EKF_posterior, EKF_cov_posterior = self.innovation(EKF_pred.reshape(6), EKF_pred_cov, measurement, z_cov)

        self.ekf_state = EKF_posterior
        self.ekf_state_cov = EKF_cov_posterior

        self.ukf_state = UKF_posterior
        self.ukf_state_cov = UKF_cov_posterior

        ukf_message = self.odometry_message(current_t, UKF_posterior, UKF_cov_posterior)
        ekf_message = self.odometry_message(current_t, EKF_posterior, EKF_cov_posterior)

        self.ekf_odom_pub.publish(ekf_message)
        self.ukf_odom_pub.publish(ukf_message)
        

    def predict(self, state, cmd_vel, dt):
        vx = cmd_vel.linear.x
        vy = cmd_vel.linear.y
        omega = cmd_vel.angular.z


        #Jacobian to use EKF in order to compare
        Gx = np.array([[1, 0, -(vx*np.sin(state[2]) + vy*np.cos(state[2]))*dt, dt*np.cos(state[2]), -dt*np.sin(state[2]), 0],
                       [0, 1, (vx*np.cos(state[2]) - vy*np.sin(state[2]))*dt, dt*np.sin(state[2]), dt*np.cos(state[2]), 0],
                       [0, 0, 1, 0, 0, dt],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])

        #implement nonlinear motion model for prediction
        A = np.zeros((6,1))
        A[0] = state[0] + (vx*np.cos(state[2]) - vy*np.sin(state[2]))*dt
        A[1] = state[1] + (vy*np.cos(state[2]) + vx*np.sin(state[2]))*dt
        A[2] = state[2] + omega*dt
        A[3] = 0
        A[4] = 0
        A[5] = 0

        #input is linear
        B_u = np.zeros((6,1))
        B_u[5] = omega
        B_u[4] = vy
        B_u[3] = vx

        predicted_state = A + B_u
        #No input covariance due to the way 

        EKF_pred_cov = Gx@self.ekf_state_cov@Gx.T


        return predicted_state, EKF_pred_cov

    def innovation(self, pred_state, pred_cov, z, z_cov):
        #the measurement model is linear so will use the regular kalman filter equations for the measurement
        
        #from robot description
        wheel_radius = 0.0762
        wheel_pair_separation = 0.488
        wheel_separation = 0.44715
        wheel_width = 0.05

        b = 0.5*(wheel_separation + wheel_width)
        a = 0.5*(wheel_radius + wheel_pair_separation)
        roller_wheel_effect = (a+b)/wheel_radius

        #implement measurement model
        C = np.zeros((5,6))

        C[0, 5] = 1
        C[1:, 3] = 1/wheel_radius
        C[1:, 4] = 1/wheel_radius
        C[1:, 5] = roller_wheel_effect
        C[1, 4] = -1*C[1,4]
        C[4,4] = -1*C[4,4]
        C[1,5] = -1*C[1,5]
        C[3,5] = -1*C[3,5]

        epsilon = 1e-6

        innovation_cov = C@pred_cov@(C.T) + z_cov + epsilon*np.eye(5)
        innovation_cov = np.array(innovation_cov, dtype=np.float64)
        K = pred_cov@(C.T)@np.linalg.inv(innovation_cov)
        state = pred_state + K@(z - C@pred_state)
        state_cov = (np.eye(6) - K@C)@pred_cov + epsilon*np.eye(6)
        return state, state_cov
    
    def odometry_message(self, current_t, state, state_cov):
        message = Odometry()
        orientation_matrix = np.array([[np.cos(state[2]), -np.sin(state[2]), 0, 0],
                                      [np.sin(state[2]), np.cos(state[2]), 0, 0],
                                      [0, 0, 1, 0], [ 0, 0, 0, 1]])

        message.header.stamp = current_t
        message.header.frame_id = "odom"
        message.child_frame_id = "base_footprint"

        message.pose.pose.position = Point(state[0], state[1], 0)        
        message.pose.pose.orientation = Quaternion(*transformations.quaternion_from_matrix(orientation_matrix))

        pose_cov = np.zeros((6,6))
        pose_cov[0, 0] = state_cov[0,0]
        pose_cov[0,1], pose_cov[1, 0] = state_cov[0,1], state_cov[0,1]
        pose_cov[1,1] = state_cov[1, 1]
        pose_cov[0, -1], pose_cov[-1, 0] = state_cov[0, 2], state_cov[0, 2]
        pose_cov[1, -1], pose_cov[-1 ,1] = state_cov[1, 2], state_cov[1, 2]
        pose_cov[-1, -1] = state_cov[2, 2]

        message.pose.covariance = pose_cov.flatten().tolist()

        message.twist.twist.linear.x = state[3]
        message.twist.twist.linear.y = state[4]
        message.twist.twist.angular.z = state[5]

        twist_cov = np.zeros((6,6))
        twist_cov[0, 0] = state_cov[3, 3]
        twist_cov[1, 1] = state_cov[4, 4]
        twist_cov[-1, -1] = state_cov[5, 5]
        twist_cov[0, -1], twist_cov[-1, 0] = state_cov[-1, 3], state_cov[-1, 3]
        twist_cov[1, -1], twist_cov[-1, 1] = state_cov[4, -1], state_cov[-1, 4]

        message.twist.covariance = twist_cov.flatten().tolist()

        return message
        



def main():
    rospy.init_node('UKF_Odom')
    odom_node = UKF_Odometry()
    rospy.loginfo('starting ukf_odom')
    rospy.spin()
    rospy.loginfo('done')

if __name__=='__main__':
    main()
     