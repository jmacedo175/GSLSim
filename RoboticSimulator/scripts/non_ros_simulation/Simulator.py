#!/usr/bin/env python
import cv2
import numpy as np
import math
import random

import Odor
from Clock import Clock
from Robotics import *
from Utils import distance, fix_angle
import sys
from numba import njit
import rospy
import time

class Simulator:
    def __init__(self, n_robots, n_evaluations, ind_mapa, run, search_stage, safe_distance,
                 saturation_threshold, detection_threshold, odor_alpha, start_region_radius,
                 robot_radius, max_linear_speed, max_angular_speed, FOV, n_beams, laser_range, dist_threshold,
                 angular_threshold, com_duration, motion_length, plume_lost_time, visualize, scale,
                 map_folder='data_environment/', use_pheromones=False, debug=False, cell_size = 0.5):

        self.plume_lost_time = plume_lost_time
        self.use_pheromones = use_pheromones
        self.cell_size = cell_size
        self.target_search_stage = search_stage
        self.n_robots = n_robots
        self.n_evaluations = n_evaluations
        self.run = run
        self.ind_mapa = ind_mapa

        self.map_filename = map_folder + 'mapa_' + str(ind_mapa) + '_run' + str(run) + '_odor_wind_data.txt'
        self.odor_filename = map_folder + 'mapa_' + str(ind_mapa) + '_run' + str(run) + '_odor_wind_data_odor_grid.txt'
                
        self.safe_distance = safe_distance
        self.saturation_threshold = saturation_threshold
        self.detection_threshold = detection_threshold
        self.odor_alpha = odor_alpha

        self.dist_threshold = dist_threshold
        self.angular_threshold = angular_threshold
        self.com_duration = com_duration
        self.motion_length = motion_length
        self.debug = debug
        self.visualize = visualize
        self.scale = scale
        self.mapa = Odor.read_dataset(self.map_filename, self.odor_filename, 
                                      np.zeros((self.n_evaluations, 2)), start_region_radius, n_robots, n_evaluations)

        self.clock = Clock(self.mapa.simulation_step)
        self.mapa.wind_x_scale = (self.mapa.max_x - self.mapa.min_x) * 1.0 / self.mapa.wind_grid_length
        self.mapa.wind_y_scale = (self.mapa.max_y - self.mapa.min_y) * 1.0 / self.mapa.wind_grid_width

    
        self.robots = [
            Robot(i, 0, 0, 0, robot_radius, max_linear_speed, max_angular_speed, FOV, n_beams, laser_range,
                  self.clock, detection_threshold, saturation_threshold, odor_alpha, self.plume_lost_time) for i in range(n_robots)]

        self.robot_footprints = np.zeros((self.n_robots, 3))       

        self.odor_pheromones = np.zeros((int(round(self.mapa.world_length*1.0/self.cell_size)),int(round(self.mapa.world_width*1.0/self.cell_size)))) 
        self.no_odor_pheromones = np.zeros((int(round(self.mapa.world_length*1.0/self.cell_size)),int(round(self.mapa.world_width*1.0/self.cell_size))))

        self.start(False, 0)


        self.pheromone_x_coords = []#np.array([[i*self.cell_size+cell_size/2 for j in range(len(self.odor_pheromones[0]))] for i in range(len(self.odor_pheromones))])
        self.pheromone_y_coords = []#np.array([[j*self.cell_size+cell_size/2 for j in range(len(self.odor_pheromones[0]))] for i in range(len(self.odor_pheromones))])
        for i in range(len(self.odor_pheromones)):
            for j in range(len(self.odor_pheromones[0])):
                self.pheromone_x_coords.append(i*self.cell_size+cell_size/2)
                self.pheromone_y_coords.append(j*self.cell_size+cell_size/2)
        self.pheromone_x_coords = np.array(self.pheromone_x_coords)
        self.pheromone_y_coords = np.array(self.pheromone_y_coords)


        self.odor_pheromone_initial_energy = 100.0
        self.odor_pheromone_decay = 0.999
        self.no_odor_pheromone_initial_energy = 100.0
        self.no_odor_pheromone_decay = 0.99



    def collision(self, x, y, r, obstacles, safe_distance):
        for o in obstacles:
            if (distance(x, y, o[0], o[1]) <= (r + o[2] + safe_distance)):
                return True
        return False


    def start(self, restart, ind_eval):
        search_stage = self.target_search_stage
        if(search_stage=='all'):
            search_stage='find'


        if not restart:
            for i in range(self.n_evaluations):
                if i % 2 == 0:
                    self.mapa.start_region[i][0] = self.mapa.max_x - 4
                    self.mapa.start_region[i][1] = self.mapa.max_y - 4
                else:
                    self.mapa.start_region[i][0] = self.mapa.max_x - 4
                    self.mapa.start_region[i][1] = 4

                robs = []
                for j in range(self.n_robots):
                    ##start positions ###
                    self.mapa.start_positions[i][j][0] = random.gauss(self.mapa.start_region[i][0],
                                                                      self.mapa.start_region_radius)
                    self.mapa.start_positions[i][j][1] = random.gauss(self.mapa.start_region[i][1],
                                                                      self.mapa.start_region_radius)

                    while (self.collision(self.mapa.start_positions[i][j][0], self.mapa.start_positions[i][j][1],
                                          self.robots[j].radius, self.mapa.obstacles,
                                          self.safe_distance) or self.collision(self.mapa.start_positions[i][j][0],
                                                                                self.mapa.start_positions[i][j][1],
                                                                                self.robots[j].radius, robs,
                                                                                self.safe_distance)):
                        self.mapa.start_positions[i][j][0] = random.gauss(self.mapa.start_region[i][0],
                                                                          self.mapa.start_region_radius)
                        self.mapa.start_positions[i][j][1] = random.gauss(self.mapa.start_region[i][1],
                                                                          self.mapa.start_region_radius)
                    
                    robs.append(
                        (self.mapa.start_positions[i][j][0], self.mapa.start_positions[i][j][1], self.robots[j].radius))

                    if self.mapa.start_positions[i][j][1] < (self.mapa.min_y + self.mapa.world_width * 0.5):
                        self.mapa.start_positions[i][j][2] = random.gauss(math.pi * 0.5, math.pi * 0.25)
                    else:
                        self.mapa.start_positions[i][j][2] = random.gauss(-math.pi * 0.5, math.pi * 0.25)



                    self.mapa.start_positions[i][j][2] = fix_angle(self.mapa.start_positions[i][j][2])


        self.ind_eval = ind_eval
        self.clock.restart()
        
        if(self.use_pheromones):
            self.odor_pheromones *=0
            self.no_odor_pheromones *=0

        for i in range(self.n_robots):
            if search_stage == 're-encounter':
                self.robots[i].last_odor_sensed_location[0] = self.mapa.source_pos[0] + 1.0 * math.cos(
                    self.mapa.wind_dir)
                self.robots[i].last_odor_sensed_location[1] = self.mapa.source_pos[1] + 1.0 * math.sin(
                    self.mapa.wind_dir)

            self.robots[i].x = self.mapa.start_positions[ind_eval][i][0]
            self.robots[i].y = self.mapa.start_positions[ind_eval][i][1]
            self.robots[i].heading = self.mapa.start_positions[ind_eval][i][2]
            
            self.robot_footprints[i][0] = self.robots[i].x
            self.robot_footprints[i][1] = self.robots[i].y
            self.robot_footprints[i][2] = self.robots[i].radius


    def simulate_step(self):
        step = int(self.clock.get_step()%len(self.mapa.wind_data))
        

        if(self.visualize):
            print('Simulation time: '+str(self.clock.get_time()))

        wind_grid = self.mapa.wind_data[step]
        odor_grid = self.mapa.odor_data[step]

        if(self.use_pheromones):
            self.odor_pheromones *= self.odor_pheromone_decay
            self.no_odor_pheromones *= self.no_odor_pheromone_decay
            self.odor_pheromones[self.odor_pheromones<10] = 0
            self.no_odor_pheromones[self.no_odor_pheromones<10] = 0


        for i in range(self.n_robots):
            self.robots[i].odom_lock.acquire() ##prevent the robot from moving

            # scan
            self.robots[i].scan_lock.acquire()
            self.robots[i].laser_angs, self.robots[i].readings, self.robots[i].obstacle_force = get_laser_readings(self.robots[i].n_readings, self.robots[i].readings, self.robots[i].laser_angs, self.robots[i].ang_inc, self.robots[i].start_ang, self.robots[i].laser_range, self.robots[i].x, self.robots[i].y,
                                   self.robots[i].heading, self.mapa.min_x, self.mapa.max_x, self.mapa.min_y, self.mapa.max_y, self.mapa.obstacles, self.robot_footprints,
                                   self.robots[i].obstacle_force)
            self.robots[i].scan_lock.release()

            # odor
            self.robots[i].odor_lock.acquire()
            self.robots[i].accumulated_odor = max(
                Odor.get_odor_grid(odor_grid, self.mapa.odor_grid_spacing, self.robots[i].x,
                                   self.robots[i].y, self.robots[i].accumulated_odor,
                                   self.clock.simulation_step, self.odor_alpha), 0)
            
            self.robots[i].set_odor(
                Odor.odor_sensor(self.robots[i].accumulated_odor, self.saturation_threshold,
                                 self.detection_threshold))
            self.robots[i].odor_lock.release()

            # wind
            self.robots[i].wind_lock.acquire()
            self.robots[i].wind_speed, self.robots[i].global_upwind_dir, self.robots[i].global_downwind_dir, self.robots[i].upwind_dir, self.robots[i].crosswind_dir, self.robots[i].farther_crosswind_dir, self.robots[i].downwind_dir = get_wind_perceptions(self.mapa.min_x, self.mapa.max_x, self.mapa.min_y, self.mapa.max_y, self.mapa.wind_x_scale, self.mapa.wind_y_scale, self.mapa.wind_grid_length, self.mapa.wind_grid_width, wind_grid, self.robots[i].x, self.robots[i].y, self.robots[i].heading, self.robots[i].wind_alpha, self.robots[i].wind_speed, self.robots[i].global_upwind_dir, self.robots[i].global_downwind_dir, self.robots[i].upwind_dir, self.robots[i].crosswind_dir, self.robots[i].farther_crosswind_dir, self.robots[i].downwind_dir)
            self.robots[i].wind_lock.release()

            if(self.use_pheromones):
                indices = xy2ij((self.robots[i].x, self.robots[i].y), self.cell_size)
                indices[0] = max(0,min(indices[0], len(self.odor_pheromones)-1))
                indices[1] = max(0,min(indices[1], len(self.odor_pheromones[0])-1))
                if(self.robots[i].sensing_odor):
                    self.odor_pheromones[indices[0]][indices[1]] = self.odor_pheromone_initial_energy
                else:
                    self.no_odor_pheromones[indices[0]][indices[1]] = self.no_odor_pheromone_initial_energy
                
            self.robots[i].odom_lock.release()
 
        if(self.use_pheromones):
            ##maybe remove the pheromone forces and only publish the locations of the pheromones in the environment?? I.e., the matrices of pheromones
            #pass
            for i in range(self.n_robots):
                self.robots[i].pheromone_lock.acquire()
                compute_pheromone_force(self.robots[i].x,self.robots[i].y, self.cell_size, self.odor_pheromones, self.no_odor_pheromones, self.robots[i].odor_pheromone_force, self.robots[i].no_odor_pheromone_force)
            #    self.robots[i].odor_pheromone_direction = math.atan2(self.robots[i].odor_pheromone_force[1]-self.robots[i].y, self.robots[i].odor_pheromone_force[0]-self.robots[i].x)
            #    self.robots[i].odor_pheromone_inversedirection = (math.pi + self.robots[i].odor_pheromone_direction)%(2*math.pi)
            #    if(self.robots[i].odor_pheromone_inversedirection>math.pi):
            #        self.robots[i].odor_pheromone_inversedirection-=2*math.pi
            #    
            #    self.robots[i].no_odor_pheromone_direction = math.atan2(self.robots[i].no_odor_pheromone_force[1]-self.robots[i].y, self.robots[i].no_odor_pheromone_force[0]-self.robots[i].x)
            #    self.robots[i].no_odor_pheromone_inversedirection = (math.pi + self.robots[i].no_odor_pheromone_direction)%(2*math.pi)
            #    if(self.robots[i].no_odor_pheromone_inversedirection>math.pi):
            #        self.robots[i].no_odor_pheromone_inversedirection-=2*math.pi
                self.robots[i].pheromone_lock.release()

        if self.visualize:
            self.draw_world(wind_grid, odor_grid, self.odor_pheromones, self.no_odor_pheromones)

    

    def draw_world(self, wind_grid, odor_grid, odor_pheromones, no_odor_pheromones):
        width = int(round(self.mapa.world_width * self.scale))
        length = int(round(self.mapa.world_length * self.scale))

        img = np.zeros((width, length, 3), np.uint8)

        cv2.rectangle(img, (0, 0), (int(self.mapa.max_x * self.scale), int(self.mapa.max_y * self.scale)),
                      (255, 255, 255), -1)

        img = Odor.draw_world(img, length, width, wind_grid, self.mapa.wind_grid_spacing, odor_grid,
                              self.mapa.odor_grid_spacing, self.detection_threshold, self.saturation_threshold,
                              self.mapa.source_pos)

        for o in self.mapa.obstacles:
            if(len(o)==3 or o[3]==-1):
                cv2.circle(img, (int(o[0] * self.scale), int(o[1] * self.scale)), int(o[2] * self.scale), (155, 155, 155), -1)
            else:
                cv2.rectangle(img, (int((o[0]-o[2]/2)*self.scale),int((o[1]-o[3]/2)*self.scale)), (int((o[0]+o[2]/2)*self.scale),int((o[1]+o[3]/2)*self.scale)), (155,155,155),-1)


        cell_size = int(round(self.cell_size*self.scale))
        for i in range(len(odor_pheromones)):
            for j in range(len(odor_pheromones[i])):
                try:
                    x,y = ij2xy((i,j), self.cell_size)
                    xl = int(round(x*self.scale-self.cell_size*0.5))
                    yl = int(round(y*self.scale-self.cell_size*0.5))

                    c = min(1.0,no_odor_pheromones[i][j]/100)
                    if(c>0):
                    
                        cv2.circle(img, (int(round(x* self.scale)), int(round(y * self.scale))), int(self.cell_size *c* self.scale),
                           (0, 155, 155), -1)

                    c = min(1,odor_pheromones[i][j]/100)

                    if(c>0):
                    
                        cv2.circle(img, (int(round(x* self.scale)), int(round(y * self.scale))), int(self.cell_size*c * self.scale),
                           (c, 155, 0), -1)

                except:
                    continue

        best_rob = self.robots[0]
        for i in range(1, self.n_robots):
            if self.robots[i].sensed_odor > best_rob.sensed_odor:
                best_rob = self.robots[i]

        for rob in self.robots:

            rx = int(round(rob.x * self.scale))
            ry = int(round(rob.y * self.scale))

            cv2.circle(img, (rx, ry), int(round(rob.radius * self.scale)), (255, 0, 0), -1)

            for i in range(rob.n_readings):
                cv2.line(img, (rx, ry), (int(rx + math.cos(rob.laser_angs[i]) * rob.readings[i] * self.scale),
                                         int(ry + math.sin(rob.laser_angs[i]) * rob.readings[i] * self.scale)),
                         (0, 0, 255), 3)

            ##wind and crosswind representation
            #a = rob.global_upwind_dir#math.atan2(rob.wind_force[1],rob.wind_force[0])+heading
            #cv2.line(img, (rx,ry), (int(rx+2*rob.wind_speed*math.cos(a)*self.scale),int(ry+2*rob.wind_speed*math.sin(a)*self.scale)), (255,0,0), 5)

            ##obstacle force representation
            a = robotToWorldHeading(math.atan2(rob.obstacle_force[1], rob.obstacle_force[0]), rob.heading)
            d = math.sqrt(pow(rob.obstacle_force[0], 2) + pow(rob.obstacle_force[1], 2)) * 100
            cv2.line(img, (rx, ry),
                     (int(rx + d * math.cos(a) * self.scale), int(ry + d * math.sin(a) * self.scale)),
                     (0, 100, 200), 5)


            ##target representation
            if rob.target_set:
                a = math.atan2(rob.target[1], rob.target[0]) + rob.heading
                cv2.line(img, (rx, ry),
                         (int(rx + math.cos(a) * self.scale), int(ry + math.sin(a) * self.scale)),
                         (255, 0, 255), 5)


            ## pheromone force representation
            a = math.atan2(rob.odor_pheromone_force[1], rob.odor_pheromone_force[0])
            d = math.sqrt(math.pow(rob.odor_pheromone_force[0],2)+math.pow(rob.odor_pheromone_force[1],2))
            #print('odor pheromone force', rob.odor_pheromone_force)
            #if(d>0):
            #    cv2.line(img,(rx,ry), (int(rx+d*math.cos(a)*self.scale), int(ry+d*math.sin(a))*self.scale), (255,0,0),5)
            cv2.line(img,(rx,ry), (int((rob.x+rob.odor_pheromone_force[0])*self.scale), int((rob.y+rob.odor_pheromone_force[1])*self.scale)), (0,255,255),5)
            
            a = math.atan2(rob.no_odor_pheromone_force[1], rob.no_odor_pheromone_force[0])
            d = math.sqrt(math.pow(rob.no_odor_pheromone_force[0],2)+math.pow(rob.no_odor_pheromone_force[1],2))
            #if(d>0):
            #    cv2.line(img,(rx,ry), (int(rx+d*math.cos(a)*self.scale), int(ry+d*math.sin(a))*self.scale), (0,0,255),5)
            cv2.line(img,(rx,ry), (int((rob.x+rob.no_odor_pheromone_force[0])*self.scale), int((rob.y+rob.no_odor_pheromone_force[1])*self.scale)), (255,255,0),5)
            #print('no odor pheromone force', rob.no_odor_pheromone_force)
            if rob.sensing_odor:
                #print('--------------------SENSING ODOUR--------------------', rob.sensed_odor)
                if rob == best_rob:
                    cv2.circle(img, (rx, ry), int(round(rob.radius * self.scale)), (0, 255, 0), -1)
                else:
                    cv2.circle(img, (rx, ry), int(round(rob.radius * self.scale)), (0, 255, 255), -1)
        cv2.namedWindow('robot simulator', cv2.WINDOW_NORMAL)
        cv2.imshow('robot simulator', img)
        cv2.resizeWindow('erratic simulator', 1000, 700)

        cv2.waitKey(1)


    def continuous_simulation(self):
        print('Starting continuous simulation')
        step =0
        while(True):
            self.simulate_step()

            if(self.visualize):
                print('step: ',step)


            #for i in range(len(self.robots)):
            #    self.robots[i].move(self.mapa.simulation_step)
            #    self.robots[i].publish_all()

            step+=1
            self.clock.increment_time(self.mapa.simulation_step)



def read_config_file():
    f=open('config.txt')

    params={}
    line = f.readline()

    while(line!=''):
        line = line.split('\t')
        params[line[0]]=eval(line[1])

        line = f.readline()

    f.close()

    return params

@njit()
def compute_pheromone_force(x,y, cell_size, odor_pheromones, no_odor_pheromones, odor_pheromone_force, no_odor_pheromone_force):
    robot_inds = xy2ij((x,y), cell_size)
            
    odor_pheromone_force *= 0
    no_odor_pheromone_force *= 0

    xinds, yinds = np.where(odor_pheromones!=0)

    for i in range(len(xinds)):
        if(xinds[i]!=robot_inds[0] or yinds[i]!=robot_inds[1]):
            ang = math.atan2(yinds[i]-robot_inds[1],xinds[i]-robot_inds[0])

            odor_pheromone_force[0]+=odor_pheromones[xinds[i]][yinds[i]]*math.cos(ang)
            odor_pheromone_force[1]+=odor_pheromones[xinds[i]][yinds[i]]*math.sin(ang)

    xinds, yinds = np.where(no_odor_pheromones!=0)
    #print('pheromone inds',xinds, yinds)
    for i in range(len(xinds)):
        if(xinds[i]!=robot_inds[0] or yinds[i]!=robot_inds[1]):
            ang = math.atan2(robot_inds[1]-yinds[i],robot_inds[0]-xinds[i])

            no_odor_pheromone_force[0]+=no_odor_pheromones[xinds[i]][yinds[i]]*math.cos(ang)
            no_odor_pheromone_force[1]+=no_odor_pheromones[xinds[i]][yinds[i]]*math.sin(ang)
    #print('computed no odor force', no_odor_pheromone_force)


@njit()
def xy2ij(pos, cell_size):
    return np.array([int(pos[0] / cell_size), int(pos[1] / cell_size)])


@njit()
def ij2xy(indices, cell_size):
    xy = np.zeros(2)

    xy[0] = indices[0] * cell_size + cell_size / 2
    xy[1] = indices[1] * cell_size + cell_size / 2
    return xy

def parse_commandline(params):
    for j in range(1,len(sys.argv)):
        i = sys.argv[j]
        if (':' in i):
            key = i.split(':')[0]
            val = i.split(':')[1]
        elif('=' in i):
            key = i.split(':')[0]
            val = i.split('=')[1]

        if(key == 'visualize' or key == 'show'):
            params['visualize'] = eval(val)
            if(not isinstance(params['visualize'], bool)):
                params['visualize'] = (params['visualize']==1)
        elif(key=='debug'):
            debug = eval(val)
            if(not isinstance(debug, bool)):
                debug = (debug==1)
        else:
            params[key] = eval(val)


if __name__=='__main__':
    debug = False
    params = read_config_file()

    
    random.seed(params['seeds'][params['run']])

    sim = Simulator(params['n_robots'], params['ind_mapa'], params['ind_mapa'], params['run'], 'all', params['safe_distance'],
                 params['saturation_threshold'], params['detection_threshold'], params['odor_alpha'], params['start_region_radius'],
                 params['robot_radius'],params['max_linear_speed'], params['max_angular_speed'], params['FOV'], params['n_beams'], params['laser_range'], params['dist_threshold'],
                 params['angular_threshold'], params['com_duration'], params['motion_length'], params['plume_lost_time'], params['visualize'], params['scale'],
                 params['map_folder'], params['use_pheromones'], debug, params['cell_size'])
    print('Created simulator')



    sim.continuous_simulation()
	
