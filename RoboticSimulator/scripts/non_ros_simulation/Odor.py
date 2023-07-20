import numpy as np
from Utils import distance
import math
import cv2
from numba import njit

class WorldMap:
    def __init__(self, min_x, max_x, min_y, max_y, wind_data, odor_data, odor_grid_spacing, source_pos, obstacles, wd,
                 ws, gr, Q, emission_rate, wind_grid_length, wind_grid_width, wind_grid_spacing, simulation_time, simulation_step,
                 start_region, start_region_radius, n_robots, n_evaluations):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.source_pos = source_pos
        self.obstacles = obstacles

        self.world_length = (max_x - min_x)
        self.world_width = (max_y - min_y)
        self.max_dist = math.sqrt(pow(self.world_length, 2) + pow(self.world_width, 2))
        self.total_steps = len(wind_data)
        self.simulation_time = simulation_time
        self.simulation_step = simulation_step

        self.wind_data = wind_data
        self.wind_dir = wd
        self.wind_speed = ws
        self.wind_grid_length, self.wind_grid_width = wind_grid_length, wind_grid_width
        self.wind_grid_spacing = wind_grid_spacing

        self.odor_data = odor_data
        self.odor_grid_length, self.odor_grid_width = len(odor_data[0]), len(odor_data[0][0])
        self.odor_grid_spacing = odor_grid_spacing
        self.g_rate = gr
        self.Q = Q
        self.emission_rate = emission_rate
        self.max_dist_src = max([distance(source_pos[0], source_pos[1], v[0], v[1]) for v in
                                 ((min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y))])
        self.start_region = start_region
        self.start_region_radius = start_region_radius
        self.start_positions = np.zeros((n_evaluations, n_robots,3))
        self.n_robots = n_robots
        self.n_evaluations = n_evaluations
        self.max_collisions = 0

def draw_world(img, width, height, grid, grid_spacing, ogrid, ogrid_spacing, detection_threshold, saturation_threshold, source_pos):

    ##standard grid
    for i in range(len(ogrid)):
        for j in range(len(ogrid[i])):

            val = ogrid[i][j]
            #val = odor_sensor(ogrid[i][j], saturation_threshold, detection_threshold)
            #print('ogrid', ogrid[i][j],'val', val)
            if val > detection_threshold:
                val = int(round((min(val, saturation_threshold) - detection_threshold) / (
                            saturation_threshold - detection_threshold) * 155))
                #print(val)
                #cv2.rectangle(img, (int(round(i * ogrid_spacing * 100)), int(round(j * ogrid_spacing * 100))), (int(round((i + 1) * ogrid_spacing * 100)), int(round((j + 1) * ogrid_spacing * 100))), (100, 255 - val, 100), -1)
                cv2.rectangle(img, (int(round(i * ogrid_spacing * 100)), int(round(j * ogrid_spacing * 100))), (int(round((i + 1) * ogrid_spacing * 100)), int(round((j + 1) * ogrid_spacing * 100))), (100, 255 , 100), -1)
            #elif(ogrid[i][j]>0):
            #    print(ogrid[i][j], detection_threshold, saturation_threshold)
            #    cv2.rectangle(img, (int(round(i * ogrid_spacing * 100)), int(round(j * ogrid_spacing * 100))), (int(round((i + 1) * ogrid_spacing * 100)), int(round((j + 1) * ogrid_spacing * 100))), (255, 10, 255), -1)
        # else:
        #   cv2.rectangle(img, (int(round(i*ogrid_spacing*100)),int(round(j*ogrid_spacing*100))),(int(round((i+1)*ogrid_spacing*100)),int(round((j+1)*ogrid_spacing*100))), (0,0,0),5)

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            x = int(round((i - 1) * grid_spacing * 100))
            y = int(round((j - 1) * grid_spacing * 100))
            # print(x,y,grid_spacing)
            # if(x+int(grid[i][j][0]*50)>=0 and int(grid[i][j][1]*50)>=0 and x+int(grid[i][j][0]*50)<width and int(grid[i][j][1]*50)<height):
            cv2.line(img, (x, y), (x + int(grid[i][j][0] * 100), y + int(grid[i][j][1] * 100)), (0, 0, 0), 4)

    cv2.circle(img, (int(source_pos[0] * 100), int(source_pos[1] * 100)), int(0.25 * 100), (0, 200, 0), -1)


    return img


def read_odor_grid(filename):
    odor_data = []

    f = open(filename)
    line = f.readline()
    odor_grid_spacing = float(line[:-2].split(':')[1])

    line = f.readline()
    while line != '':
        line = f.readline()
        grid = []
        while '</grid>' not in line:
            line = line[:-1].split('\t')
            for i in range(len(line)):
                line[i] = float(line[i])
            grid.append(np.array(line))
            line = f.readline()

        odor_data.append(np.array(grid))
        line = f.readline()

    return odor_data, odor_grid_spacing


def read_dataset(map_filename, odor_filename, start_region, start_region_radius, n_robots, n_evaluations):
    min_x, min_y, max_x, max_y, simulation_time, simulation_step, wind_speed, wind_dir, grid_spacing, emission_rate, growth_rate, Q = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    wind_grid_lines, wind_grid_cols = 0, 0

    source_pos = np.zeros(2)
    obs = [[0,0,0]]
    f = open(map_filename, 'r')
    line = f.readline()

    while '<time' not in line:
        if 'source_pos' in line:
            line = line[:-2].split(':')[1].split(',')
            source_pos[0] = float(line[0])
            source_pos[1] = float(line[1])

        elif '<obstacles' in line:

            line = f.readline()
            while not '</obstacles' in line:
                line = line[1:-2].split(',')
                #print(line)
                o = np.array([float(line[i]) for i in range(len(line))])
                #print(o)
                #o[0] = float(line[0])
                #o[1] = float(line[1])
                #o[2] = float(line[2])
                obs.append(o)
                line = f.readline()

        elif "<min_x" in line:
            min_x = float(line[1:-2].split(':')[1])
        elif "<min_y" in line:
            min_y = float(line[1:-2].split(':')[1])
        elif "<max_x" in line:
            max_x = float(line[1:-2].split(':')[1])
        elif "<max_y" in line:
            max_y = float(line[1:-2].split(':')[1])
        elif "<simulation_time" in line:
            simulation_time = float(line[1:-2].split(':')[1])
        elif "<simulation_step" in line:
            simulation_step = float(line[1:-2].split(':')[1])
        elif "<wind_speed" in line:
            wind_speed = float(line[1:-2].split(':')[1])
        elif "<wind_dir" in line:
            wind_dir = float(line[1:-2].split(':')[1])
        elif "<grid_spacing" in line:
            grid_spacing = float(line[1:-2].split(':')[1])
        elif "<emission_rate" in line:
            emission_rate = float(line[1:-2].split(':')[1])
        elif "<growth_rate" in line:
            growth_rate = float(line[1:-2].split(':')[1])
        elif "<Q" in line:
            Q = float(line[1:-2].split(':')[1])

        line = f.readline()

    wind_data = []
    # step =0
    while line != '':

        if '<wind' in line:
            line = line[1:-2].split(':')[1]
            line = line.split(',')
            wind_grid_spacing = float(line[0])
            wind_grid_lines = int(line[1])
            wind_grid_cols = int(line[2])
            wind_speed = (float(line[3]))
            wind_grid = np.zeros((wind_grid_lines, wind_grid_cols, 2))
            line = f.readline()
            i = 0
            while i < wind_grid_lines:
                j = 0

                while j < wind_grid_cols:
                    line = line[1:-2].split(',')
                    wind_grid[i][j][0] = float(line[2])
                    wind_grid[i][j][1] = float(line[3])
                    line = f.readline()
                    j += 1
                i += 1

            wind_data.append(wind_grid)

        line = f.readline()

    f.close()

    odor_data, odor_grid_spacing = read_odor_grid(odor_filename)

    mapa = WorldMap(min_x, max_x, min_y, max_y, wind_data, odor_data, odor_grid_spacing, source_pos, np.array(obs), wind_dir,
                    wind_speed, growth_rate, Q, emission_rate, wind_grid_lines, wind_grid_cols, wind_grid_spacing, simulation_time,
                    simulation_step, start_region, start_region_radius, n_robots, n_evaluations)
    return mapa


@njit()
def get_grid_coordinates(x, y, grid_spacing):
    xg = int((x / grid_spacing))
    yg = int((y / grid_spacing))
    return xg, yg


@njit()
def get_world_coordinates(xg, yg, grid_spacing):
    x = (xg * grid_spacing) + (grid_spacing * 0.5)
    y = (yg * grid_spacing) + (grid_spacing * 0.5)
    return x, y


@njit()
def get_odor_grid(grid, grid_spacing, x, y, accu_odor, simulation_step, alpha = 0.5):
    # instant odor
    xg, yg = get_grid_coordinates(x, y, grid_spacing)

    xg = max(min(xg,len(grid)-1),0)
    yg = max(min(yg, len(grid[0]) - 1), 0)

    c = grid[xg][yg]
    alpha = alpha * simulation_step

    c +=  - alpha * accu_odor + alpha * c
    #c =  (1- alpha) * accu_odor + alpha * c

    return c


@njit()
def odor_sensor(c, saturation_threshold, detection_threshold):
    if c <= detection_threshold:
        c = 0
    return min(c, saturation_threshold)
