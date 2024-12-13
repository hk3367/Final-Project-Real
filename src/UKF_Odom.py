#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import Imu, JointState
import math
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber

class UKF_Odometry:
    def __init__(self):
        rospy.loginfo("Initializing node...")

        self.odom_pub = rospy.Publisher('/UKF_odom', Odometry, queue_size=10)

        # Subscribers for topics
        imu_sub = Subscriber('/imu_sim', Imu)
        wheel_encoder_sub = Subscriber('/wheel_encoder_sim', JointState)
        cmd_vel_sub = Subscriber('/mobile_base_controller/cmd_vel', Twist)

        # Synchronize the topics
        self.ats = ApproximateTimeSynchronizer([imu_sub, wheel_encoder_sub, cmd_vel_sub], queue_size=100, slop=5.0, allow_headerless=True)
        self.ats.registerCallback(self.callback)

        rospy.loginfo("Subscribers created and callback registered.")

        self.state = np.zeros((6,1))
        self.state_cov = 0.1*np.eye(6)

        rospy.loginfo("init done")

        self.prev_time = None

    def callback(self, imu_data, wheel_encoder_data, cmd_vel):
        rospy.loginfo("entering callback")
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

        #Compute EKF in order to compare and show the difference in complexity between EKF and UKF
        EKF_pred, EKF_pred_cov = self.predict(self.state, cmd_vel, dt)

        #define dimension of state for use in UKF
        n = 6
        
        #define common parameter values for UKF
        alpha = 1e-3
        k = 3
        beta = 2
        lambda_value = 7 - n

        #need to define the sigma points that will be used to reconstuct predicted mean and covariance
        sigma = np.zeros((2*n+1, n))        
        cov_sqrt = np.linalg.cholesky(self.state_cov)
        rospy.loginfo(EKF_pred_cov)

        #first sigma point is just the current state
        sigma[0] = self.state.T
        for i in range(n):
            sigma[i+1] = self.state.T + (math.sqrt((n-k)+lambda_value)*cov_sqrt[:, i]).T
    
   
        for i in range(n):
            sigma[n+i+1] = self.state.T - (math.sqrt((n-k)+lambda_value)*cov_sqrt[:, i]).T

        mean_weights = 1/(2*(n+lambda_value))*np.ones(2*n+1)
        cov_weights = np.copy(mean_weights)
        
        mean_weights[0] = lambda_value/(lambda_value+n)
        cov_weights[0] = lambda_value/(lambda_value+n)
        #rospy.loginfo(mean_weights)
        #rospy.loginfo(cov_weights)
        weighted_pred = np.zeros((2*n+1, n))
        weighted_cov_pred = []



        for i in range(len(sigma)):
            pred, _ = self.predict(sigma[i], cmd_vel, dt)
            #rospy.loginfo(pred)
            weighted_pred[i] = (mean_weights[i]*pred).T

        for i in range(len(sigma)):
            distance = sigma[i]-weighted_pred[i]
            #rospy.loginfo(np.shape(distanceT))
            #A = distanceT@distance
            #rospy.loginfo(distance)
            A = np.outer(distance, distance)
            weighted_cov_pred.append(cov_weights[i]*A)

        #rospy.loginfo(weighted_pred)
        pred_cov =np.zeros((6,6))
        for matrix in weighted_cov_pred:
            pred_cov += matrix
        
        pred_state = np.sum(weighted_pred, axis=0)
        #rospy.loginfo(pred_state)
        #self.state = pred_state
        #self.cov = pred_cov
        #rospy.loginfo(pred_cov)
        #rospy.loginfo(measurement)
        self.innovation(pred_state, pred_cov, measurement, z_cov)
        rospy.loginfo(self.state)
        #rospy.loginfo(self.state_cov)


        

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
        predicted_state = np.zeros((6,1))
        predicted_state[0] = state[0] + (vx*np.cos(state[2]) - vy*np.sin(state[2]))*dt
        predicted_state[1] = state[1] + (vy*np.cos(state[2]) + vx*np.sin(state[2]))*dt
        predicted_state[2] = state[2] + omega*dt
        predicted_state[3] = vx
        predicted_state[4] = vy
        predicted_state[5] = omega

        EKF_pred_cov = Gx@self.state_cov@Gx.T

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
        self.state = pred_state + K@(z - C@pred_state)
        self.state_cov = (np.eye(6) - K@C)@pred_cov + epsilon*np.eye(6)
        




def main():
    rospy.init_node('UKF_Odom')
    odom_node = UKF_Odometry()
    rospy.loginfo('starting ukf_odom')
    rospy.spin()
    rospy.loginfo('done')

if __name__=='__main__':
    main()
     