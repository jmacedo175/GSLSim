#!/usr/bin/env python
import random
import math
import sys

import numpy as np
import time
import cv2
from numba import njit

class Filament:
    def __init__(self,x,y,r,concentration):
        self.x = x
        self.y=y
        self.radius = r
        self.concentration=concentration


class OdorSource:
    def __init__(self,emission_rate,x,y,h,r,concentration):
        self.emission_rate=emission_rate
        self.x=x
        self.y=y
        self.h = h
        self.radius=r
        self.concentration=concentration
        self.filaments=[]
        self.toEmit=0.0

def read_dataset(filename):
    odor_data = []
    wind_data=[]
    f = open(filename,'r')
    line = f.readline()
    map_params = {}
    while('<time' not in line):
        if('source_pos' in line):
            map_params['source_pos']=[]
            line = line[:-2].split(':')[1].split(',')
            for l in line:
                map_params['source_pos'].append(float(l))
        elif('<obstacles' in line):
            obs=[]
            line = f.readline()
            while(not '</obstacles' in line):
                line = line[1:-2].split(',')
                o=[]
                for l in line:
                    o.append(float(l))
                obs.append(o)
                line = f.readline()
            map_params['obstacles']=obs
        else:
            line = line[1:-2].split(':')
            map_params[line[0]]=float(line[1])
            
        line = f.readline()
    while(line!=''):
        if('<time 'in line):
            
            grid=[]
            sources = []
        
        elif('<source:' in line):
            
            line = line[1:-2].split(':')[1]
            line = line.split(',')
            emission_rate=float(line[0])
            x = float(line[1])
            y = float(line[2])
            h = float(line[3])
            radius = float(line[4])
            concentration=float(line[5])
            toEmit=float(line[6])
            source = OdorSource(emission_rate,x,y,h,radius,concentration)
            source.toEmit=toEmit

            line = f.readline()
            while('</source' not in line):
                line = line[1:-2].split(',')
                x = float(line[0])
                y=float(line[1])
                radius = float(line[2])
                concentration=float(line[3])
                fil = Filament(x,y,radius,concentration)
                source.filaments.append(fil)
                line = f.readline()
            sources.append(source)

        elif('</odor' in line):
            odor_data.append(sources)

        elif('<wind' in line):

            line = line[1:-2].split(':')[1]
            line = line.split(',')
            grid_spacing = float(line[0])
            n_lines = int(line[1])
            n_cols = int(line[2])
            wind_speed = (float(line[3]))

            line = f.readline()
            i=0
            j=0
            while(i<n_lines):

                j=0
                linha=[]
                while(j<n_cols):
                    line = line[1:-2].split(',')
                    linha.append([float(line[2]),float(line[3])])
                    line =f.readline()
                    j+=1
                i+=1
                grid.append(linha)
            wind_data.append(grid)

        line = f.readline()
    f.close()
    
    return [map_params,wind_data,odor_data]


def log_grid(grid, grid_spacing, filename, step):
    if(step==0):
        f = open(filename,'w')
        f.write('<grid_spacing:'+str(grid_spacing)+'>\n')
    else:
        f = open(filename,'a')

    f.write('<grid>\n')
    for x_line in grid: 
        s=''
        for y in x_line:
            s+= str(y)+'\t'
        f.write(s[:-1]+'\n')

    f.write('</grid>\n')
    f.close()



def read_odor_grid(filename, compact_grid=True):
    grid_list=[]
    

    f = open(filename)
    line = f.readline()
    grid_spacing = float(line[:-2].split(':')[1])

    line = f.readline()
    while(line!=''):
        line = f.readline()
        if(compact_grid):
            grid={}
        else:
            grid = []
        while('</grid>' not in line):
            line = line[:-1].split('\t')
            if(compact_grid):
                grid[(int(line[0]),int(line[1]))] = float(line[2])
            else:
                for i in range(len(line)):
                    line[i] = float(line[i])
                grid.append(line)
            line = f.readline()
        grid_list.append(grid)
        line = f.readline()

    return grid_list, grid_spacing


@njit()
def distance(x1,y1,x2,y2):
    return math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))

@njit()
def test_obstacles(x,y,radius,obstacles):
    ##obstacles as an array of [[x,y,width,length], ..., [x, y,width,length]]
    ##round obstacles have width=radius and length=-1
    for obs in obstacles:
        if(obs[3]==-1):
            if(distance(obs[0],obs[1],x,y)<=(obs[2]+radius)):
                a=math.atan2(y-obs[1],x-obs[0])+random.gauss(0,0.1)
                x = math.cos(a)*(obs[2]+radius)+obs[0]
                y = math.sin(a)*(obs[2]+radius)+obs[1]
                return [x,y]
        else:
            diffx, diffy = (x-obs[0]), (y-obs[1])
            theta = math.atan2(diffy, diffx)
            if(-math.pi/4 <= theta and theta <= math.pi/4):
                xd1 = obs[0]+obs[2]/2
                m = (diffy/diffx)
                b = obs[1]-m*obs[0]
                yd1 = m*xd1 + b

            elif(-3*math.pi/4 > theta or theta > 3*math.pi/4):
                xd1 = obs[0]-obs[2]/2
                m = (diffy/diffx)
                b = obs[1]-m*obs[0]
                yd1 = m*xd1 + b

            elif(theta>0):
                yd1 = obs[1]+obs[3]/2
                m = (diffy/diffx)
                b = obs[1]-m*obs[0]
                xd1 = (yd1-b)/m

            else:
                yd1 = obs[1]-obs[3]/2
                m = (diffy/diffx)
                b = obs[1]-m*obs[0]
                xd1 = (yd1-b)/m

            d2 = distance(x,y, xd1,yd1)
            if(d2<=(radius)):                
                a = random.gauss(theta, 0.1)
                d = distance(x,y, obs[0], obs[1])
                x += math.cos(a)*(radius)
                y += math.sin(a)*(radius)
                return [x,y]
    return [x,y]

@njit()
def grow_filament(r, g_rate, time_step, model):
    if(model==1):
        return r+((3/2) * g_rate * pow(r,1/3))*time_step
    else:
        return r+(g_rate/(2*r))*time_step

def update_odor(mapa,sources,wind_dir, wind_speed,time_step,length,width,grid,g_rate,grid_spacing,Kx,variance, fil_model=1, fil_init_radius=None):
    if(fil_init_radius==None):
        if(fil_model ==1):
            fil_init_radius = pow(0.0316,1.5)
        else:
            fil_init_radius = 0.0316


    #update wind
    plume_extended = False
    i = len(grid)-2
    while(i>0):
        line = grid[i]
        j=len(line)-2
        while(j>0):

            

            #solver like d_u_x = ((x+1)-(x-1))/(2h)
            #solver like d2_u_x2 = ((x+1)-2*x+(x-1))/(h^2)

            #u
            u = line[j][0]
            d_u_y = (grid[i][j+1][0]-grid[i][j-1][0])*1.0/(2*grid_spacing)
            d_u_x = (grid[i+1][j][0]-grid[i-1][j][0])*1.0/(2*grid_spacing)

            d2_u_y2 = (grid[i][j+1][0]-2*grid[i][j][0]+grid[i][j-1][0])*1.0/pow(grid_spacing,2)
            d2_u_x2 = (grid[i+1][j][0]-2*grid[i][j][0]+grid[i+1][j][0])*1.0/pow(grid_spacing,2)

            #v
            v=line[j][1]
            d_v_y = (grid[i][j+1][1]-grid[i][j-1][1])*1.0/(2*grid_spacing)
            d_v_x = (grid[i+1][j][1]-grid[i-1][j][1])*1.0/(2*grid_spacing)

            d2_v_y2 = (grid[i][j+1][1]-2*grid[i][j][1]+grid[i][j-1][1])*1.0/pow(grid_spacing,2)
            d2_v_x2 = (grid[i+1][j][1]-2*grid[i][j][1]+grid[i+1][j][1])*1.0/pow(grid_spacing,2)
                        

            
            u+=(-u*d_u_x-v*d_u_y+0.5*Kx*d2_u_x2+0.5*Kx*d2_u_y2)*time_step
            v+=(-u*d_v_x-v*d_v_y+0.5*Kx*d2_v_x2+0.5*Kx*d2_v_y2)*time_step
            
            ang = random.gauss(math.atan2(v,u),variance)
            d = random.gauss(0,0.1*wind_speed)+math.sqrt(pow(u,2)+pow(v,2))
            grid[i][j] = [math.cos(ang)*d,math.sin(ang)*d]
            
            j-=1
        i-=1




    #move the source
    #for source in sources:
    #   w = source.h+random.gauss(0,0.2)
    #   x = source.x+math.cos(w)*20*time_step
    #   y = source.y+math.sin(w)*20*time_step
    #   while(x<0 or y<0 or x>length or y>width):
    #       w += random.gauss(0,0.2)
    #       x = source.x+math.cos(w)*20*time_step
    #       y = source.y+math.sin(w)*20*time_step
    #   source.x=x
    #   source.y=y
    #   source.h = w

    #emit
    for source in sources:
        source.toEmit+=source.emission_rate*time_step
        
        if(source.toEmit>0):
            for i in range(int(source.toEmit)):
                source.filaments.append(Filament(float(source.x),float(source.y),fil_init_radius,source.concentration))
            source.toEmit-=int(source.toEmit)

    #move odor
    for source in sources:
        for fil in source.filaments:
            if(fil.x+fil.radius>mapa.min_x and fil.y+fil.radius>mapa.min_y and fil.x-fil.radius<mapa.max_x and fil.y-fil.radius<mapa.max_y):
                ##the filament is inside the map
                min_x = int(fil.x*1.0/grid_spacing)+1
                max_x = min_x+1
                
                if(max_x>=len(grid)):
                    max_x-=1
                    min_x-=1

                min_y = int(fil.y*1.0/grid_spacing)+1
                max_y = min_y+1

                if(max_y>=len(grid[0])):
                    max_y-=1
                    min_y-=1


                dist_corners = [distance(fil.x,fil.y,min_x,min_y), distance(fil.x,fil.y,min_x,max_y), distance(fil.x,fil.y,max_x,min_y), distance(fil.x,fil.y,max_x,max_y)]
                u = (grid[min_x][min_y][0]*dist_corners[0]+grid[min_x][max_y][0]*dist_corners[1]+grid[max_x][min_y][0]*dist_corners[2]+grid[max_x][max_y][0]*dist_corners[3])/(dist_corners[0]+dist_corners[1]+dist_corners[2]+dist_corners[3])
                v = (grid[min_x][min_y][1]*dist_corners[0]+grid[min_x][max_y][1]*dist_corners[1]+grid[max_x][min_y][1]*dist_corners[2]+grid[max_x][max_y][1]*dist_corners[3])/(dist_corners[0]+dist_corners[1]+dist_corners[2]+dist_corners[3])


                ang = math.atan2(v,u)+random.gauss(0,variance)
                d = math.sqrt(pow(u,2)+pow(v,2))

                a_dif= random.random()*2*math.pi
                s_dif = random.gauss(0,0.05*d)
                

                x=fil.x+time_step*(d*math.cos(ang)+s_dif*math.cos(a_dif))
                y=fil.y+time_step*(d*math.sin(ang)+s_dif*math.sin(a_dif))


                if(len(mapa.obstacles)>0):
                    [x1,y1] = test_obstacles(x,y,fil.radius,mapa.obstacles)
                    while(x1!=x and y1!=y):
                        x = x1
                        y=y1
                        [x1,y1] = test_obstacles(x,y,fil.radius,mapa.obstacles)

                    fil.x = x1
                    fil.y = y1
                else:
                    fil.x = x
                    fil.y = y
                fil.radius = grow_filament(fil.radius, g_rate, time_step, fil_model)
                #fil.radius += g_rate
            else:
                plume_extended=True
                source.filaments.remove(fil)

    return grid, plume_extended

@njit()
def compute_odor_filament(fil, x, y, Q):
    #instant odor from a single filament
    d=distance(x,y,fil[0],fil[1])*100
    r = fil[2]*100
    ci =Q/(math.sqrt(8.0*pow(math.pi,3)*pow(r,3)))
    ci*=math.exp(-1.0*pow(d,2)/pow(r,2))
    return ci

@njit()
def compute_odor_filament_farrell(fil, x, y, Q):

    d=distance(x,y,fil[0],fil[1])*100
    r = fil[2]*100
    ci =Q/(math.sqrt(8.0*pow(math.pi,3))*pow(r,3))
    ci*=math.exp(-1.0*pow(d,2)/pow(r,2))
    return ci


@njit()
def compute_odor_filament_fontefarrell(fil, x, y, Q):
    # Handbook on ATMOSPHERIC DIFFUSION 1982
    d=distance(x,y,fil[0],fil[1])*100
    r = fil[2]*100
    ci =Q/(math.sqrt(8.0*pow(math.pi,3))*pow(r,3))
    ci*=math.exp(-1.0*pow(d,2)/(2*pow(r,2)))
    return ci


@njit()
def compute_odor_filament_LM(fil, x, y, Q):
    r = fil[2]*100
    ci =Q/(2.0*math.pi*pow(r,2))
    ci*=math.exp(-pow((fil[0]-x)*100,2)/(2*pow(r,2)) - pow((fil[1]-y)*100,2)/(2*pow(r,2)))
    return ci

@njit()
def compute_odor_filament_pompy(fil, x, y, Q):
    d=distance(x,y,fil[0],fil[1])*100
    r = fil[2]*100
    ci =Q/(math.sqrt(8.0*pow(math.pi,3)*pow(r,3)))
    ci*=math.exp(-1.0*pow(d,2)/(r*2))
    return ci

def compute_odor(sources,x,y,Q):
    c=0.0
    for source in sources:
        for fil in source.filaments:
            c+=compute_odor_filament(fil, x, y, Q)          
    return c


def get_odor(sources,x,y,Q,accu_odor,simulation_step, alpha = (1.0/30)):
    c=0.0
    for source in sources:
        for fil in source.filaments:
            c+=compute_odor_filament(fil, x, y, Q)

    alpha = alpha*simulation_step
    c += (-alpha)*accu_odor+alpha*c
    return c

@njit()
def get_grid_coordinates(x,y,grid_spacing):
    xg = int((x/grid_spacing))
    yg = int((y/grid_spacing))

    return xg,yg

@njit()
def get_world_coordinates(xg,yg, grid_spacing):
    x = (xg*grid_spacing)+(grid_spacing*0.5)
    y = (yg*grid_spacing)+(grid_spacing*0.5)
    return x,y  

@njit()
def get_odor_grid(grid,grid_spacing,x,y,accu_odor,simulation_step, compact_grid=True):
    xg, yg = get_grid_coordinates(x,y,grid_spacing)

    c = grid[xg][yg]

    ##filter                
    alpha = 0.5*simulation_step
    c += (-alpha)*accu_odor+alpha*c
    return c    



@njit()
def make_grid(length,width,grid_spacing,wind_dir,wind_speed):
    grid=[]
    i=0
    while(i<=length):

        l=[]
        j=0
        while(j<=width):
            l.append([math.cos(wind_dir)*wind_speed,math.sin(wind_dir)*wind_speed])
            j+=grid_spacing
        grid.append(l)
        i+=grid_spacing
    return grid

def draw_world(img, length, width, grid, grid_spacing, ogrid, ogrid_spacing, detection_threshold, saturation_threshold, source_pos, sources=None, draw_grid=True):

    ##draw odour grid
    if(draw_grid):
        for i in range(len(ogrid)):
            for j in range(len(ogrid[i])):

                val = ogrid[i][j]
                if val > detection_threshold:
                    val = int(round((min(val, saturation_threshold) - detection_threshold) / (
                                saturation_threshold - detection_threshold) * 155))
                    cv2.rectangle(img, (int(round(i * ogrid_spacing * 100)), int(round(j * ogrid_spacing * 100))), (int(round((i + 1) * ogrid_spacing * 100)), int(round((j + 1) * ogrid_spacing * 100))), (100, 255 , 100), -1)

    #draw odour filaments
    if(sources!=None):
        for source in sources:
            cv2.circle(img, (int(source.x*100),int(source.y*100)), int(source.radius*100), (0,200,0),-1)
            for fil in source.filaments:
                
                fx = fil.x*100
                fy = fil.y*100
                if(fx>0 and fy>0 and fx<length and fy<width):
                    cv2.circle(img, (int(round(fx)),int(round(fy))), int(fil.radius*100), (0,150,0),5)

    ##draw wind
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            x = int(round((i - 1) * grid_spacing * 100))
            y = int(round((j - 1) * grid_spacing * 100))
            cv2.line(img, (x, y), (x + int(grid[i][j][0] * 100), y + int(grid[i][j][1] * 100)), (0, 0, 0), 4)

    cv2.circle(img, (int(source_pos[0] * 100), int(source_pos[1] * 100)), int(0.25 * 100), (0, 200, 0), -1)


    return img

