#!/usr/bin/env python
import random
import math
import sys
from Odor import *

import cv2
import numpy as np
import time
import make_odor_grid
from copy import deepcopy

def read_config_file(filename = '../config.txt'):
    f=open(filename)

    params={}
    line = f.readline()

    while(line!=''):
        line = line.split('\t')
        params[line[0]]=eval(line[1])

        line = f.readline()

    f.close()

    return params

class world_map:
    def __init__(self,min_x,max_x,min_y,max_y,grid,wd,ws,gr,Q):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.obstacles=[]
        self.sources=[]
        self.grid = grid
        self.wind_dir=wd
        self.wind_speed = ws
        self.length=(max_x-min_x)#*100
        self.width=(max_y-min_y)#*100
        self.g_rate=gr
        self.Q=Q

class RectObstacle:
    def __init__(self, x, y, length, width):
        self.x = x
        self.y = y
        self.length = length
        self.width = width

class RoundObstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

def make_mapa(min_x,max_x,min_y,max_y,grid,wind_dir,wind_speed,growth_rate,Q,source_pos,grid_spacing,obstacles,start_positions,end_positions,emission_rate, detection_threshold, saturation_threshold):
    mapa = world_map(min_x,max_x,min_y,max_y,deepcopy(grid),wind_dir,wind_speed,growth_rate,Q)
    for o in obstacles:
        mapa.obstacles.append(o)
    mapa.obstacles = np.array(mapa.obstacles)
    mapa.grid_spacing = grid_spacing
    mapa.sources.append(OdorSource(emission_rate,source_pos[0],source_pos[1],source_pos[2],0.2,100))
    return mapa

def roundf(n):
    return round(n*2)/2


def work(show, run, ind_mapa, path, eval_time, fil_model = 2, fil_init_radius = 0.0316, odor_grid_spacing=0.25, odor_grid_func=compute_odor_filament_farrell, growth_rate = 0.001, Q_line = 0.98625 * 10, detection_threshold = pow(10,-6), saturation_threshold = 500*pow(10,-6), obstacles=[]):

    seeds = [985, 570, 815, 356, 782, 148, 998, 188, 306, 783, 245, 521, 264, 121, 772, 635, 485, 754, 583, 740, 581, 926, 272, 855, 667, 742, 605, 148, 163, 353]

    parameters = {
        1:{'wind_speed':0.7, 'wind_angular_variance':0.075, 'emission_rate':0.1, 'length':40,'width':40,'Q_line':Q_line,'growth_rate':0.01, 'Kx':6, 'grid_spacing':0.15,'wind_dir':0, 'source_pos':[[0.2,0.35],[0.48,0.53]]},
        2:{'wind_speed':0.7, 'wind_angular_variance':0.075, 'emission_rate':0.1, 'length':40,'width':40,'Q_line':Q_line,'growth_rate':0.01, 'Kx':6, 'grid_spacing':0.15,'wind_dir':0, 'source_pos':[[0.2,0.35],[0.48,0.53]]}
        }


    while run <30:#len(seeds):

        random.seed(seeds[run])
        print('Run '+str(run))
        
        simulation_time=eval_time # seconds
        simulation_step=0.5 # seconds

        wind_speed = parameters[ind_mapa]['wind_speed']#m/s
        emission_rate=parameters[ind_mapa]['emission_rate']# #filaments/s
        growth_rate = parameters[ind_mapa]['growth_rate']
        Q_line = parameters[ind_mapa]['Q_line']##molecules/s or g/s
        Q = Q_line/emission_rate ##molecules/filament
        Kx=parameters[ind_mapa]['Kx']
        variance=parameters[ind_mapa]['wind_angular_variance']


        

        min_x=0
        max_x=min_x+parameters[ind_mapa]['length']
        min_y=0
        max_y=min_y+parameters[ind_mapa]['width']
        grid_spacing = (max_x-min_x)*parameters[ind_mapa]['grid_spacing']#meters

        length = parameters[ind_mapa]['length']#max_x-min_x
        width = parameters[ind_mapa]['width']#max_y-min_y

        #######making maps######
        wind_dir = parameters[ind_mapa]['wind_dir']
        grid = make_grid((max_x-min_x+grid_spacing*2),(max_y-min_y+grid_spacing*2),grid_spacing,wind_dir,wind_speed)
        


        #odour sources
        source_pos = parameters[ind_mapa]['source_pos']
        source_pos=[random.randint(int(length*source_pos[0][0]),int(length*source_pos[0][1]))+0.5,random.randint(int(width*source_pos[1][0]),int(width*source_pos[1][1]))+0.5,math.pi/4]
        while(True):
            collision=False
            for o in obstacles:
                if(distance(o[0],o[1],source_pos[0],source_pos[1])<o[2]*1.2):
                    collision=True
                    break
            if(not collision):
                break
            source_pos=[random.randint(5,20),random.randint(5,20),math.pi/4]
            
        mapa=make_mapa(min_x,max_x,min_y,max_y,grid,wind_dir,wind_speed,growth_rate,Q,source_pos,grid_spacing,obstacles,None,None,emission_rate, detection_threshold, saturation_threshold)
        mapa.Kx = Kx
        mapa.variance = variance
        scale = 100

        print('writing to '+path+'mapa_'+str(ind_mapa)+'_run'+str(run)+'_odor_wind_data.txt')
        grid_filename = path+'mapa_'+str(ind_mapa)+'_run'+str(run)+'_odor_wind_data_odor_grid.txt'
        f = open(path+'mapa_'+str(ind_mapa)+'_run'+str(run)+'_odor_wind_data.txt','w')
        t=0.0
        
        f.write('<min_x:'+str(min_x)+'>\n')
        f.write('<max_x:'+str(max_x)+'>\n')
        f.write('<min_y:'+str(min_y)+'>\n')
        f.write('<max_y:'+str(max_y)+'>\n')
        f.write('<simulation_time:'+str(simulation_time)+'>\n')
        f.write('<simulation_step:'+str(simulation_step)+'>\n')
        f.write('<wind_speed:'+str(wind_speed)+'>\n')
        f.write('<wind_dir:'+str(wind_dir)+'>\n')
        f.write('<grid_spacing:'+str(grid_spacing)+'>\n')
        f.write('<emission_rate:'+str(emission_rate)+'>\n')
        f.write('<growth_rate:'+str(growth_rate)+'>\n')
        f.write('<Q:'+str(Q)+'>\n')
        f.write('<source_pos:'+str(source_pos[0])+','+str(source_pos[1])+'>\n')
        f.write('<obstacles>\n')
        for o in obstacles:
            s = '<'
            for i in o:
                s+=str(i)+','
            s=s[:-1]+'>\n'
            f.write(s)
        f.write('</obstacles>\n')
    
        width = (mapa.max_y-mapa.min_y)*scale
        length = (mapa.max_x-mapa.min_x)*scale
        plume_extended=False

        step =0
        while(t<simulation_time):
            
            grid, p = update_odor(mapa,mapa.sources,mapa.wind_dir, mapa.wind_speed,simulation_step,mapa.length,mapa.width,grid,mapa.g_rate,grid_spacing,Kx,variance, fil_model, fil_init_radius)

            if(p):
                plume_extended=True
                
            if(plume_extended):
                s = '<time '+str(t)+'>\n'
                s+='<odor>\n'
                for src in mapa.sources:
                    s+='<source:'+str(src.emission_rate)+','+str(src.x)+','+str(src.y)+','+str(src.h)+','+str(src.radius)+','+str(src.concentration)+','+str(src.toEmit)+'>\n'
                    for fil in src.filaments:
                        s+='<'+str(fil.x)+','+str(fil.y)+','+str(fil.radius)+','+str(fil.concentration)+'>\n'
                    s+='</source>\n'
                s+='</odor>\n'
                s+='<wind:'+str(grid_spacing)+','+str(len(grid))+','+str(len(grid[0]))+','+str(wind_speed)+'>\n'
                i=0
                while(i<len(grid)):
                    j=0
                    while(j<len(grid[i])):
                        s+='<'+str(0.0+i*grid_spacing-1)+','+str(0.0+j*grid_spacing-1)+','+str(grid[i][j][0])+','+str(grid[i][j][1])+'>\n'
                        j+=1
                    i+=1
                s+='</wind>\n'
                s+='</time>\n'
                f.write(s)

                ##make odor grid
                ogrid, ogrid_spacing = make_odor_grid.make_grid_filament_based_step(mapa.sources,mapa.obstacles,Q, simulation_step, min_x, max_x, min_y, max_y, grid_filename, odor_grid_spacing, odor_grid_func, detection_threshold, emission_rate)
                log_grid(ogrid, ogrid_spacing, grid_filename, step)
                step+=1


            if(show):               
                img = np.ones((width,length,3), np.uint8)
                img*=255
                
                if(plume_extended):
                    img = draw_world(img, length, width, grid, grid_spacing, ogrid, ogrid_spacing, detection_threshold, saturation_threshold, source_pos)
                else:
                    img = draw_world(img, length, width, grid, grid_spacing, None, None, detection_threshold, saturation_threshold, source_pos, mapa.sources, draw_grid=False)

                cv2.line(img, (mapa.min_x*scale,mapa.min_y*scale), (mapa.max_x*scale,mapa.min_y*scale), (150,150,150), 3)
                cv2.line(img, (mapa.min_x*scale,mapa.min_y*scale), (mapa.min_x*scale,mapa.max_y*scale), (150,150,150), 3)
                cv2.line(img, (mapa.max_x*scale,mapa.max_y*scale), (mapa.max_x*scale,mapa.min_y*scale), (150,150,150), 3)
                cv2.line(img, (mapa.max_x*scale,mapa.max_y*scale), (mapa.min_x*scale,mapa.max_y*scale), (150,150,150), 3)

                for o in mapa.obstacles:
                    if(o[3]==-1):
                        ##round obstacle
                        cv2.circle(img, (int(round(o[0]*scale)),int(round(o[1]*scale))), int(round(o[2]*scale)), (155,155,155),-1)
                    else:
                        ##rectangular obstacle
                        cv2.rectangle(img, (int(round((o[0]-o[2]/2)*scale)),int(round((o[1]-o[3]/2)*scale))), (int(round((o[0]+o[2]/2)*scale)),int(round((o[1]+o[3]/2)*scale))), (155,155,155),-1)
                cv2.namedWindow('a',cv2.WINDOW_NORMAL)
                cv2.imshow('a',img)
                #cv2.resizeWindow('a', 1200,600)
                cv2.waitKey(1)

            if(plume_extended):
                t+=simulation_step

        f.close()        
        run+=1

if __name__ =='__main__':
    show = False
    run =0
    ind_mapa= 2
    eval_time = 500
    fil_model = 2 
    if(fil_model ==1):
        fil_init_radius = pow(0.0316,1.5)
    else:
        fil_init_radius = 0.0316

    odor_grid_spacing=0.25
    growth_rate= 0.01

    Q_line = 0.000986/2

    detection_threshold = pow(10,-9)
    saturation_threshold = 500*pow(10,-9)
    
    #odor_grid_func = compute_odor_filament_farrell
    odor_grid_func = compute_odor_filament_fontefarrell
    #odor_grid_func = compute_odor_filament_LM
    #odor_grid_func = compute_odor_filament_pompy


    ####create obstacles here#####
    ##Each obstacle is a circle, defined by a list with the form [x,y,width]
        
    if ind_mapa==1:
        obstacles=[
        [35,9,1,1],
        [35,10,1,1],
        [35,11,1,1],
        [35,12,1,1],
        [35,13,1,1],
        [35,14,1,1],
        [35,15,1,1],
        [35,16,1,1],
        [35,17,1,1],
        [35,18,1,1],
        [35,19,1,1],
        [35,20,1,1],
        [35,21,1,1],
        [35,22,1,1],
        [35,28,1,1],
        [35,29,1,1],
        [35,30,1,1],
        [35,31,1,1],
        [35,32,1,1],
        [35,33,1,1],
        [35,34,1,1],
        [35,35,1,1],
        [35,36,1,1],
                

        [31,5,1,1],
        [32,5,1,1],
        [33,5,1,1],
        [34,5,1,1],
        [35,5,1,1],
        [36,5,1,1],
        [37,5,1,1],
        [38,5,1,1],
        [39,5,1,1],
        [40,5,1,1],

        [34,32,1,1],
        [33,32,1,1],
        [32,32,1,1],
        [27,32,1,1],
        [26,32,1,1],
        [25,32,1,1],
        [24,32,1,1],
        [23,32,1,1],
        [22,32,1,1],
        [21,32,1,1],
        [20,32,1,1],
        [19,32,1,1],
        [19,40,1,1],
        [19,39,1,1],
        [19,38,1,1],
        [19,38,1,1],
        [19,37,1,1],
        [19,36,1,1],
        [18,36,1,1],


        [13,36,1,1],
        [12,36,1,1],
        [11,36,1,1],
        [10,36,1,1],
        [9,36,1,1],
        [8,36,1,1],
        [7,36,1,1],
        [6,36,1,1],
        [5,36,1,1],
        [5,35,1,1],
        [5,34,1,1],
        [5,33,1,1],        
        [5,32,1,1],
        [5,31,1,1],
        [5,30,1,1],
        [5,29,1,1],


        [25,7,1,1],
        [24.75,8,1,1],
        [24.5,9,1,1],
        [24.25,10,1,1],
        [24,11,1,1],
        [23.75,12,1,1],
        [23.5,13,1,1],
        [23.25,14,1,1],
        [23,15,1,1],
        [22.75,16,1,1],
        [22.5,17,1,1],
        
        [22.75,18,1,1],
        [23,19,1,1],
        [23.25,20,1,1],
        [23.5,21,1,1],
        [23.75,22,1,1],
        [24,23,1,1],
        [24.25,24,1,1],
        [24.5,25,1,1],
        [24.75,26,1,1],
        [25,27,1,1],

        [10,0,1,1],
        [10,1,1,1],
        [10,2,1,1],
        [10,3,1,1],
        [10,4,1,1],
        [10,5,1,1],
        [9,5,1,1],
        [8,5,1,1],
        [7,5,1,1],
        [6,5,1,1],
        [5,5,1,1],
        [5,6,1,1],
        [5,7,1,1],
        [5,8,1,1],
        [5,9,1,1],
        [5,10,1,1],
        
        [6,10,1,1],
        [7,10,1,1],
        [8,10,1,1],
        [9,10,1,1],
        [10,10,1,1],
        [11,10,1,1],
        [12,10,1,1],
        [13,10,1,1],
        [14,10,1,1],
        [15,10,1,1],
        [15,9,1,1],
        [15,8,1,1],
        [15,7,1,1],
        [15,6,1,1],
        [15,5,1,1],
        [16,5,1,1],
        [17,5,1,1],
        [18,5,1,1],
        [19,5,1,1],
        [20,5,1,1],
        #[21,5,1,1],
        [25.5,5,1,1],
        [25.25,6,1,1],
        [25,7,1,1],

        [5,15,1,1],
        [5,16,1,1],
        [5,17,1,1],
        [5,18,1,1],
        [5,19,1,1]
        ]
    else:
        obstacles=[]

    path = '../data_environment/'

    i=1
    while(i<len(sys.argv)):
        if(':' not in sys.argv[i] and '=' not in sys.argv[i]):
            print('Usage: python gen_odorwind_dataset.py <param:value> or <param=value>')
            exit(0)
        else:
            if(':' in sys.argv[i]):
                l = sys.argv[i].split(':')
            elif('=' in sys.argv[i]):
                l = sys.argv[i].split('=')

            if(l[0]=='show'):
                show=(l[1]=='true' or l[1]=='True')
            elif(l[0]=='run'):
                run=int(l[1])
            elif(l[0]=='ind_mapa'):
                ind_mapa=int(l[1])
        i+=1

    print('generating map '+str(ind_mapa))
    work(show, run, ind_mapa, path, eval_time, fil_model, fil_init_radius,  odor_grid_spacing, odor_grid_func, growth_rate, Q_line, detection_threshold, saturation_threshold, obstacles)
