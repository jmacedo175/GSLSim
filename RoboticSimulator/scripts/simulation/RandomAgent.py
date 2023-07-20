#!/usr/bin/env python
import numpy as np
import math
import Utils
from numba import jit, njit, float32, int32
from threading import Lock
from std_msgs.msg import String, Float64, Float32MultiArray
from RoboticSimulator.msg import scan_msg
from Robotics import Robot
import rospy
import random
class Agent:
    def __init__(self, robot_name):

        self.name = robot_name
        self.current_time = None
        self.time_lock = Lock()
        self.x, self.y, self.heading = None, None, None
        self.odom_lock = Lock()
        ##gas
        self.has_detected_odour = False
        self.last_odor_sensed_time = None
        self.last_odor_sensed_location = np.zeros(2)
        self.sensing_odor = False
        self.odor_lock = Lock()

        ## wind
        self.farther_crosswind_dir=self.upwind_dir = self.downwind_dir = self.crosswind_dir = self.global_upwind_dir = self.global_downwind_dir = self.wind_speed = None
        self.wind_lock = Lock()
        
        ##laser stuff
        self.start_ang = self.FOV = self.ang_inc = self.n_readings = self.laser_range = None
        self.readings = None
        self.laser_angs = None
        self.obstacle_force = np.zeros(2)
        self.scan_lock = Lock()

        self.target = np.zeros(3)
        self.target_set = False

        self.cmd_pub = rospy.Publisher('/' + self.name + '/cmd_vel', String, queue_size=1)
        self.odom_sub = rospy.Subscriber('/' + self.name + '/odom', Float32MultiArray, self.odom_callback)
        self.gas_sub = rospy.Subscriber('/' + self.name + '/gas', Float64, self.gas_callback)
        self.wind_sub = rospy.Subscriber('/' + self.name + '/wind', Float32MultiArray, self.wind_callback)  # publish wind speed and direction
        self.scan_sub = rospy.Subscriber('/' + self.name + '/scan', scan_msg, self.scan_callback)
        self.time_sub = rospy.Subscriber('/simulator/simulation_time', Float64, self.time_callback)

    def time_callback(self, msg):
        self.time_lock.acquire()
        self.current_time = msg.data
        self.time_lock.release()

    def gas_callback(self, msg):
        self.odor_lock.acquire()
        if(msg.data > 0):
            self.sensing_odor = True
            self.has_detected_odour = True
            
            self.time_lock.acquire()
            self.last_odor_sensed_time = self.current_time
            self.time_lock.release()

            self.odom_lock.acquire()
            self.last_odor_sensed_location[0] = self.x
            self.last_odor_sensed_location[1] = self.y
            self.odom_lock.release()

        else:
            self.sensing_odor = False
            self.has_detected_odour = False

        self.odor_lock.release()

    def odom_callback(self, msg):
        
        self.odom_lock.acquire()
        self.x = msg.data[0]
        self.y = msg.data[1]
        self.heading = msg.data[2]
        self.odom_lock.release()



    def wind_callback(self, msg):
        self.wind_lock.acquire()

        self.upwind_dir = msg.data[0]
        self.downwind_dir = msg.data[1]
        self.crosswind_dir = msg.data[2]
        self.wind_speed = msg.data[-1]
        self.farther_crosswind_dir = Utils.fix_angle(self.crosswind_dir+math.pi)

        self.wind_lock.release()

    def scan_callback(self, msg):
        self.scan_lock.acquire()

        self.n_readings = msg.n_readings
        self.obstacle_force *= 0   
        self.start_ang = msg.start_angle
        self.FOV = msg.FOV
        self.ang_inc = msg.angle_increment
        self.laser_range = msg.max_range
        if(self.readings is None):
            self.readings = np.zeros(self.n_readings)
            self.laser_angs = np.zeros(self.n_readings)
        for i in range(self.n_readings):
            self.readings[i] = msg.ranges[i]
            self.laser_angs[i] = msg.start_angle + msg.angle_increment*i
                 
            d = (self.laser_range - self.readings[i])
            a = (self.laser_angs[i] + math.pi) % (2 * math.pi)  ##robot coordinates

            self.obstacle_force[0] += math.cos(a) * d
            self.obstacle_force[1] += math.sin(a) * d

        
        self.scan_lock.release()

    def turn_left(self):
        self.target[0] = self.x
        self.target[1] = self.y
        self.target[2] = Utils.fix_angle(self.heading-math.pi/2)

    def turn_right(self):
        self.target[0] = self.x
        self.target[1] = self.y
        self.target[2] = Utils.fix_angle(self.heading+math.pi/2)

    def move_front(self):
        ##move_front
        self.target[0] = self.x+math.cos(self.heading)
        self.target[1] = self.y+math.sin(self.heading)
        self.target[2] = self.heading

    def obstacle_ahead(self):
        a = math.atan2(self.obstacle_force[1], self.obstacle_force[0])
        d = math.sqrt(self.obstacle_force[0]**2 + self.obstacle_force[1]**2)

        return (d>0 and (a < -math.pi/2 or a > math.pi/2))

    def obstacle_left(self):
        a = math.atan2(self.obstacle_force[1], self.obstacle_force[0])
        d = math.sqrt(self.obstacle_force[0]**2 + self.obstacle_force[1]**2)

        return (d>0 and  a > math.pi/2)

    def obstacle_right(self):
        a = math.atan2(self.obstacle_force[1], self.obstacle_force[0])
        d = math.sqrt(self.obstacle_force[0]**2 + self.obstacle_force[1]**2)

        return (d>0 and  a < -math.pi/2)
    
    def act(self, goal_func):
        while(True):
            self.odom_lock.acquire()
            x = self.x
            self.odom_lock.release()
            print('waiting for odom initialization', self.sensing_odor, self.readings)
            if(x!=None):
                break
            rospy.sleep(1)

        print('Walking')
        while not rospy.is_shutdown():
            if(self.target_set):
                print('target set', self.target, self.x, self.y, self.heading)
                #first check if target has been reached
                a = self.target[2]-self.heading
                if(abs(a)>0.1):
                    ##must rotate
                    #print('rotating')
                    if(a>0):
                        self.cmd_pub.publish("<0,1>")
                    else:
                        self.cmd_pub.publish("<0,-1>")
                else:
                    d = np.sqrt((self.x-self.target[0])**2 + (self.y-self.target[1])**2)
                    if(d>0.1 and not self.obstacle_ahead()):
                        #print('moving',d)

                        self.cmd_pub.publish("<1,0>")
                    else:
                        #print('stopping', d)
                        self.target_set = False
            else:
                goal_func()
            rospy.sleep(0.1)#float in seconds

    def random_walk(self):
        print('setting a new target')
        ##must define a target
        r = random.random()
        if(r<0.1):
            self.turn_left()

        elif(r<0.9):
            self.move_front()

        else:
            self.turn_right()

        self.target_set=True

    def deterministic_agent(self):
        ##must define a target

        if(self.obstacle_ahead()):
            self.turn_left()
        else:
            self.move_front()

        self.target_set=True

    def move_and_turn_agent(self):
        ##must define a target

        if(self.obstacle_ahead()):
            if(self.obstacle_left):
                self.turn_right()
            elif(self.obstacle_right()):
                self.turn_left()
            else:
                if(random.random()<0.5):
                    self.turn_left()
                else:
                    self.turn_right()
        else:
            self.move_front()

        self.target_set=True

if __name__=='__main__':
    rospy.init_node('random_walker', anonymous=True)
    agent = Agent('robot_0')
    agent.act(agent.move_and_turn_agent)

