#!/usr/bin/env python
import numpy as np
import math
import Utils
from numba import jit, njit, float32, int32
from threading import Lock

class Robot:
    def __init__(self, i, x, y, heading, radius, max_linear_speed, max_angular_speed, FOV, n_beams, laser_range, clock, detection_threshold, saturation_threshold, odor_alpha, plume_lost_time):
        self.id = i
        self.name = 'robot_'+str(i)
        ##position, size and motion stuff
        self.x = x
        self.y = y
        self.heading = heading
        self.odom_lock = Lock()

        self.radius = radius
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.clock = clock

        ##odor stuff
        self.accumulated_odor = 0
        self.sensed_odor = 0
        self.has_detected_odour = False
        self.last_odor_sensed_time = None
        self.last_odor_sensed_concentration = 0
        self.last_odor_sensed_location = np.zeros(2)
        self.max_odor_sensed_location = np.zeros(2)
        self.max_odor_sensed = 0
        self.max_odor_sensed_time = -(plume_lost_time+1)
        self.detection_threshold = detection_threshold
        self.saturation_threshold = saturation_threshold
        self.odor_alpha = odor_alpha
        self.plume_lost_time = plume_lost_time
        self.sensing_odor = False
        self.odor_lock = Lock()

        self.global_downwind_dir = 0.0
        self.global_upwind_dir = 0.0
        self.upwind_dir = 0.0
        self.downwind_dir = 0.0
        self.crosswind_dir = 0.0
        self.farther_crosswind_dir=0.0
        self.wind_speed = 0.0
        self.wind_alpha = 1.0
        self.wind_lock = Lock()
        
        ##laser stuff
        self.start_ang = -1.0 * FOV / 2
        self.FOV = FOV
        self.ang_inc = 1.0 * FOV / n_beams
        self.n_readings = n_beams
        self.laser_range = laser_range

        self.readings = np.zeros(self.n_readings)
        self.laser_angs = np.zeros(self.n_readings)
        self.obstacle_force = np.zeros(2)
        self.max_f = np.zeros(2)
        self.scan_lock = Lock()

        self.target = np.zeros(2)
        self.target_set = False
        self.target_lock = Lock()

        ##pheromone stuff
        self.pheromone_lock = Lock()
        self.odor_pheromone_force = np.zeros(2)
        self.no_odor_pheromone_force = np.zeros(2)
        self.odor_pheromone_direction = 0
        self.odor_pheromone_inversedirection = 0
        self.no_odor_pheromone_direction = 0
        self.no_odor_pheromone_inversedirection = 0


    def restart(self, search_stage, start_pos):
        ##scan
        self.scan_lock.acquire()
        for i in range(self.n_readings):
            self.readings[i] = 0.0
        self.scan_lock.release()

        self.odor_lock.acquire()
        if search_stage == 'find':
            ##odor
            self.accumulated_odor = 0.0
            self.sensed_odor = 0.0
            self.sensing_odor = False

            self.last_odor_sensed_time = -(self.plume_lost_time+1)
            self.last_odor_sensed_concentration = 0
            self.max_odor_sensed = 0
            self.max_odor_sensed_time = -(self.plume_lost_time+1)
            self.has_detected_odour = False

        elif search_stage == 'track':
            #odor
            self.accumulated_odor = self.detection_threshold+(self.detection_threshold+self.saturation_threshold)*0.5
            self.sensed_odor = self.detection_threshold+(self.detection_threshold+self.saturation_threshold)*0.5
            self.last_odor_sensed_time = self.clock.get_time()
            self.last_odor_sensed_concentration = self.sensed_odor
            self.last_odor_sensed_location[0] = self.x
            self.last_odor_sensed_location[1] = self.y

            self.max_odor_sensed = self.sensed_odor
            self.max_odor_sensed_time = self.clock.get_time()
            self.max_odor_sensed_location[0] = self.x
            self.max_odor_sensed_location[1] = self.y
            self.has_detected_odour = True
            self.sensing_odor = True

        elif search_stage == 're-encounter':
            self.accumulated_odor = 0.0
            self.sensing_odor = False
            self.sensed_odor = 0.0
            self.last_odor_sensed_time = self.clock.get_time() - self.plume_lost_time
            self.last_odor_sensed_concentration = 0

            # self.last_odor_location = [self.x, self.y] ##where should this be??

            self.max_odor_sensed = self.detection_threshold+(self.detection_threshold+self.saturation_threshold)*0.5
            self.max_odor_sensed_time = self.clock.get_time() - self.plume_lost_time
            # self.max_odor_sensed_location = [self.x, self.y] ##where should this be??
            self.has_detected_odour = True
        self.odor_lock.release()

        #wind
        self.wind_lock.acquire()
        self.global_downwind_dir = -999
        self.upwind_dir = 0.0
        self.downwind_dir = 0.0
        self.crosswind_dir = 0.0
        self.farther_crosswind_dir=0.0
        self.wind_speed = -999
        self.wind_lock.release()

        #target
        self.target_lock.acquire()
        self.target_set = False
        self.target_lock.release()

        self.odom_lock.acquire()
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.heading = start_pos[2]
        self.odom_lock.release()

                



    def set_odor(self, sensed_odor):
        #self.odor_lock.acquire()
        #self.odom_lock.acquire()
        self.sensed_odor = sensed_odor

        if sensed_odor > 0:
            self.sensing_odor = True
            self.has_detected_odour = True
            self.last_odor_sensed_concentration = sensed_odor
            self.last_odor_sensed_time = self.clock.get_time()
            self.last_odor_sensed_location[0] = self.x
            self.last_odor_sensed_location[1] = self.y
            if sensed_odor > self.max_odor_sensed:
                self.max_odor_sensed = sensed_odor
                self.max_odor_sensed_time = self.clock.get_time()
                self.max_odor_sensed_location[0] = self.x
                self.max_odor_sensed_location[1] = self.y
        else:
            self.sensing_odor = False

        #self.odom_lock.release()
        #self.odor_lock.release()

    def set_target(self, x,y):
        self.target_lock.acquire()
        self.target[0] = x
        self.target[1] = y
        self.target_set = True
        self.target_lock.release()

    def reset_target(self):
        self.target_lock.acquire()
        self.target_set = False
        self.target_lock.release()

    def move(self, simulation_step):
        self.target_lock.acquire()
        xtarg, ytarg = self.target[0], self.target[1]
        targ_set = self.target_set
        self.target_lock.release()
        self.reset_target()
        self.scan_lock.acquire()
        xobs = self.obstacle_force[0]
        yobs = self.obstacle_force[1]
        self.scan_lock.release()

        if targ_set and (xtarg != 0 or ytarg != 0):
            obs_w = 0.8
            dt = np.hypot(xobs, yobs)


            if dt != 0:
                print('OBSTACLE AHEAD', xobs, yobs)
                #xtarg = obs_w * xobs + (1 - obs_w) * xtarg
                #ytarg = obs_w * yobs + (1 - obs_w) * ytarg

                ##potential field
                #h = math.atan2(ytarg, xtarg)
                #d = np.hypot(xtarg, ytarg)

                #just dont move
                h = ytarg
                d=0
            else:
                h = ytarg
                d = xtarg

            if h > 0:
                ang_z = min(h, self.max_angular_speed)
            elif h < 0:
                ang_z = max(h, -self.max_angular_speed)
            else:
                ang_z = 0

            if dt > 0:
                lin_x = 0
            else:
                lin_x = min(self.max_linear_speed,
                            max(0.0, math.cos(ang_z)))
            lin_x = min(d, lin_x)
            self.odom_lock.acquire()
            self.heading = Utils.fix_angle(self.heading + ang_z * simulation_step)

            self.x += lin_x * simulation_step * math.cos(self.heading)
            self.y += lin_x * simulation_step * math.sin(self.heading)
            self.odom_lock.release()


@njit()
def dist_circle_beam(self_x, self_y, self_ang_inc, circle, ang):
    m = math.tan(ang)
    c = self_y - m * self_x

    a, b, r = circle[0], circle[1], circle[2]
    first = 1.0 / (2.0 * (pow(m, 2) + 1))
    sq1 = pow(-2.0 * a * m - 2.0 * b * pow(m, 2) - 2.0 * c, 2)
    sq2 = -4.0 * (pow(m, 2) + 1)
    sq3 = pow(a, 2) * pow(m, 2) + 2.0 * a * c * m + pow(b, 2) * pow(m, 2) + pow(c, 2) - pow(m, 2) * pow(r, 2)
    squareroot = (sq1 + sq2 * sq3)
    if squareroot < 0:
        return -1
    squareroot = math.sqrt(squareroot)

    y1 = first * (squareroot + 2.0 * a * m + 2.0 * b * pow(m, 2) + 2.0 * c)
    y2 = first * (-1.0 * squareroot + 2.0 * a * m + 2.0 * b * pow(m, 2) + 2.0 * c)

    if m == 0:
        d = Utils.distance(self_x, self_y, circle[0], circle[1])
        x1 = self_x + d
        x2 = self_x + d + circle[2] * 2
    else:
        x1 = (y1 - c) * 1.0 / m
        x2 = (y2 - c) * 1.0 / m

    a1 = math.atan2(y1 - self_y, x1 - self_x) % (2 * math.pi)
    a2 = math.atan2(y2 - self_y, x2 - self_x) % (2 * math.pi)

    d = -1
    if abs(a1 - ang)<self_ang_inc:
        d = Utils.distance(x1, y1, self_x, self_y)
    if abs(a2 - ang) < self_ang_inc:
        d2 = Utils.distance(x2, y2, self_x, self_y)
        if d==-1 or d2<d:
            return d2
    return d

#@njit()
def get_laser_readings(n_readings, readings, laser_angs, ang_inc, start_ang, laser_range, self_x, self_y, self_heading, mapa_min_x, mapa_max_x, mapa_min_y, mapa_max_y, obstacles, robots, obstacle_force):

    for i in range(n_readings):
        laser_angs[i] = ((ang_inc * i + start_ang + ang_inc / 2) + self_heading) % (2 * math.pi)

        # walls
        a = laser_angs[i]
        x = self_x + math.cos(a) * laser_range
        y = self_y + math.sin(a) * laser_range
        l = laser_range
        if x >= mapa_max_x:
            l = min(l, (mapa_max_x - self_x) / math.cos(a))
        elif x <= mapa_min_x:
            l = min(l, (mapa_min_x - self_x) / math.cos(a))
        if y >= mapa_max_y:
            l = min(l, (mapa_max_y - self_y) / math.sin(a))
        if y <= mapa_min_y:
            l = min(l, (mapa_min_y - self_y) / math.sin(a))

        readings[i] = l

    for k in range(1,len(obstacles)):
        d = Utils.distance(self_x, self_y, obstacles[k][0], obstacles[k][1]) - obstacles[k][2]

        if d < laser_range:
            for i in range(n_readings):
                di = dist_circle_beam(self_x, self_y, ang_inc, obstacles[k], laser_angs[i])
                if readings[i] > di >= 0:
                    readings[i] = di

    if len(robots)>1:
        for k in range(len(robots)):
            if not (robots[k][0] == self_x and robots[k][1]==self_y):
                d = Utils.distance(self_x, self_y, robots[k][0], robots[k][1]) - robots[k][2]
                if d < laser_range:
                    for i in range(n_readings):
                        di = dist_circle_beam(self_x, self_y, ang_inc, robots[k], laser_angs[i])
                        #di = dist_circle_beam(o, laser_angs[i])
                        if readings[i] > di >= 0:
                            readings[i] = di

    obstacle_force[0], obstacle_force[1] = 0.0, 0.0

    for i in range(n_readings):
        d = (laser_range - readings[i])
        a = (laser_angs[i] - self_heading + math.pi) % (2 * math.pi)  ##robot coordinates

        obstacle_force[0] += math.cos(a) * d
        obstacle_force[1] += math.sin(a) * d

    return laser_angs, readings, obstacle_force


#@njit()
'''
def move(target_set, target, obstacle_force, max_angular_speed, max_linear_speed, self_x, self_y, self_heading, simulation_step):
    if target_set and (target[0] != 0 or target[1] != 0):
        obs_w = 0.8
        dt = math.sqrt(pow(obstacle_force[0], 2) + pow(obstacle_force[1], 2))

        targ = target
        if dt != 0:
            targ = [obs_w * obstacle_force[0] + (1 - obs_w) * targ[0],
                      obs_w * obstacle_force[1] + (1 - obs_w) * targ[1]]

            h = math.atan2(targ[1], targ[0])
            d = math.sqrt(pow(targ[0], 2) + pow(targ[1], 2))
        else:
            h = targ[1]
            d = targ[0]

        if h > 0:
            ang_z = min(h, max_angular_speed)
        elif h < 0:
            ang_z = max(h, -max_angular_speed)
        else:
            ang_z = 0

        #if math.pi - abs(math.atan2(obstacle_force[1], obstacle_force[0])) < 0.78 and dt > 0:
        if dt>0:
            lin_x = 0
        else:
            lin_x = min(max_linear_speed,
                        max(0.0, math.cos(ang_z)))
        lin_x = min(d, lin_x)

        self_heading = Utils.fix_angle(self_heading + ang_z * simulation_step)

        self_x += lin_x * simulation_step * math.cos(self_heading)
        self_y += lin_x * simulation_step * math.sin(self_heading)
    return self_x, self_y, self_heading
'''

@njit()
def robotToWorldHeading(a, self_heading):
    #converts a world angle to robot angle
    return Utils.fix_angle(a + self_heading)  # robot coordinates

@njit()
def worldToRobotHeading(a, self_heading):
    #converts a world angle to robot angle
    return Utils.fix_angle(a - self_heading)  # robot coordinates

@njit()
def get_wind_perceptions(mapa_min_x, mapa_max_x, mapa_min_y, mapa_max_y, wind_x_scale, wind_y_scale, wind_grid_length, wind_grid_width, wind_grid, self_x, self_y, self_heading, robot_wind_alpha, robot_wind_speed, robot_global_upwind_dir, robot_global_downwind_dir, robot_upwind_dir, robot_crosswind_dir, robot_farther_crosswind_dir, robot_downwind_dir):
    if (mapa_min_x < self_x < mapa_max_x and
            mapa_min_y < self_y < mapa_max_y):
        min_x = int(self_x / wind_x_scale) + 1  # self.params['scale'])
        max_x = min_x + 1  # self.params['scale'])
        if max_x >= wind_grid_length:
            max_x = int((mapa_max_x - 1) / wind_x_scale)
            min_x = int(max_x - 1)
        min_y = int(self_y / wind_y_scale) + 1
        max_y = min_y + 1
        if max_y >= wind_grid_width:
            max_y = int((mapa_max_y - 1) / wind_y_scale)
            min_y = int(max_y - 1)

        x = (wind_grid[min_x][min_y][0] + wind_grid[min_x][max_y][0] + wind_grid[max_x][min_y][0] +
             wind_grid[max_x][max_y][0]) / 4
        y = (wind_grid[min_x][min_y][1] + wind_grid[min_x][max_y][1] + wind_grid[max_x][min_y][1] +
             wind_grid[max_x][max_y][1]) / 4

        if(robot_wind_speed==-999):
            robot_wind_speed = math.sqrt(pow(x, 2) + pow(y, 2))
        else:
            robot_wind_speed = robot_wind_alpha * math.sqrt(pow(x, 2) + pow(y, 2)) +(1.0 - robot_wind_alpha) *robot_wind_speed

        wd = math.atan2(y, x)
        if(robot_global_downwind_dir==-999):
            robot_global_downwind_dir = wd   
        else:
            robot_global_downwind_dir = Utils.fix_angle(robot_global_downwind_dir * (
                1 - robot_wind_alpha) + wd * robot_wind_alpha)
        
        robot_global_upwind_dir = Utils.fix_angle(robot_global_downwind_dir + math.pi)
        robot_downwind_dir = Utils.fix_angle(worldToRobotHeading(robot_global_downwind_dir, self_heading))  # robot coordinates

        robot_upwind_dir = Utils.fix_angle(worldToRobotHeading(
            robot_global_downwind_dir + math.pi, self_heading))

        if robot_upwind_dir >= 0:
            robot_crosswind_dir = Utils.fix_angle(robot_upwind_dir - math.pi / 2)
        else:
            robot_crosswind_dir = Utils.fix_angle((math.pi / 2) + robot_upwind_dir)
        robot_farther_crosswind_dir = Utils.fix_angle(math.pi+robot_crosswind_dir)

    return robot_wind_speed, robot_global_upwind_dir, robot_global_downwind_dir, robot_upwind_dir, robot_crosswind_dir, robot_farther_crosswind_dir, robot_downwind_dir
