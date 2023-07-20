#!/usr/bin/env python
import os
import time
import random
import sys
import Odor
import math
import numpy as np
from numba import njit

def make_grid_filament_based_step(sources,obstacles,Q, simulation_step, min_x, max_x, min_y, max_y, filename, grid_spacing=0.5, odor_grid_func = Odor.compute_odor_filament_farrell, detection_threshold = 10*pow(10,-6), emission_rate = 1):
	##single step version of make_grid_filament_based
	## each cell contains the odor sensed from its center

	glength = int((max_x-min_x)/grid_spacing)
	gwidth = int((max_y-min_y)/grid_spacing)
	odor_thresh=detection_threshold/emission_rate

	grid = np.zeros((glength, gwidth))
	cells = np.zeros((glength+1, gwidth+1)) ##why +1?? Shouldn't this just be equal to grid? Must check!

	obstacle_grid = np.zeros((glength, gwidth))
	for o in obstacles:
		if (o[3]==-1):
			rx = ry = int(round(o[2]/grid_spacing))
		else:
			rx, ry = int(o[2]/grid_spacing), int(o[3]/grid_spacing)

		
		ox, oy = Odor.get_grid_coordinates(o[0], o[1], grid_spacing)
		#obstacle_grid[ox][oy] = 1
		oxi = o[0]-o[2]+grid_spacing
		while(oxi<=o[0]+o[2]-grid_spacing):
			oyi = o[1]-o[3]+grid_spacing
			while(oyi<=o[1]+o[3]-grid_spacing):
				ox,oy = Odor.get_grid_coordinates(oxi, oyi, grid_spacing)
				if(ox<len(obstacle_grid) and oy < len(obstacle_grid[1])):
					obstacle_grid[ox][oy] = 1
				oyi+=grid_spacing
			oxi+=grid_spacing

	for source in sources:
		for fil in source.filaments:
			xg, yg = Odor.get_grid_coordinates(fil.x,fil.y, grid_spacing)
			cells*=0
			fil = [fil.x,fil.y, fil.radius]
			if(xg>=0 and xg<glength and yg>=0 and yg<gwidth and obstacle_grid[xg][yg]==0):
				xw,yw = Odor.get_world_coordinates(xg,yg, grid_spacing)
				grid[xg][yg]+=odor_grid_func(fil, xw, yw, Q)
				cells[xg][yg] = 1
			i=1
			

			while(True):
				foundOdor = False
				##new approach
				'''
				for yi in range(yg-i,yg+i+1):
					xi = xg-i
					while(yi>=0 and yi<gwidth and xi<=xg+i and xi<glength):						
						if(obstacle_grid[xi][yi] ==1):
							break

						elif(xi>=0 and cells[xi][yi] == 0):
							
							c = odor_grid_func(fil, xw, yw, Q)
							if(c>odor_thresh):
								foundOdor=True

							grid[xi][yi]+=c
							cells[xi][yi]=1
						
						
						yi+=1



				'''
				for xi in range(xg-i,xg+i+1):
					yi = yg-i
					while(xi>=0 and xi<glength and yi<=yg+i and yi<gwidth):						
						if(yi>=0 and cells[xi][yi] == 0 and obstacle_grid[xi][yi]==0):
							xw,yw = Odor.get_world_coordinates(xi,yi, grid_spacing)
							c = odor_grid_func(fil, xw, yw, Q)
							if(c>odor_thresh):
								foundOdor=True

							grid[xi][yi]+=c
							cells[xi][yi]=1
						
						yi+=1
				
				if(not foundOdor):
					break
				
				i+=1

	return grid, grid_spacing


def make_grid_filament_based(odor_data,Q, simulation_step, min_x, max_x, min_y, max_y, filename, grid_spacing=0.5, odor_grid_func = Odor.compute_odor_filament_farrell):
	## each cell contains the odor sensed from its center

	glength = int((max_x-min_x)/grid_spacing)
	gwidth = int((max_y-min_y)/grid_spacing)
	odor_thresh=0.1

	step = 0
	grid = np.zeros((glength, gwidth))
	cells = np.zeros((glength+1, gwidth+1))
	n_steps = len(odor_data)

	for sim_step in odor_data:
		
		grid*=0
		
		miny=None
		maxy=None
		minc = None
		for source in sim_step:
			for fil in source:
				xg, yg = Odor.get_grid_coordinates(fil[0],fil[1], grid_spacing)
				cells*=0

				if(xg>=0 and xg<glength and yg>=0 and yg<gwidth):
					xw,yw = Odor.get_world_coordinates(xg,yg, grid_spacing)
					grid[xg][yg]+=odor_grid_func(fil, xw, yw, Q)	
					cells[xg][yg] = 1
				
				i=1
				

				while(True):
					foundOdor = False
					for xi in range(xg-i,xg+i+1):
						yi = yg-i
						while(xi>=0 and xi<glength and yi<=yg+i and yi<gwidth):						
							if(yi>=0 and cells[xi][yi] == 0):
								xw,yw = Odor.get_world_coordinates(xi,yi, grid_spacing)
								c = odor_grid_func(fil, xw, yw, Q)
								
								if(c>odor_thresh):
									foundOdor=True

								grid[xi][yi]+=c
								cells[xi][yi]=1

							yi+=1
					
					if(not foundOdor):
						break
					
					i+=1

				
		Odor.log_grid(grid, grid_spacing, filename, step)
		step+=1
	


def make_grid_filament_based2(odor_data,Q, simulation_step, min_x, max_x, min_y, max_y, filename, grid_spacing=0.5):
	## each cell contains the odor sensed from the center of all filaments that are inside it

	glength = int((max_x-min_x)/grid_spacing)
	gwidth = int((max_y-min_y)/grid_spacing)
	odor_thresh=0.0

	step = 0
	grid = np.zeros((glength, gwidth))
	cells = np.zeros((glength+1, gwidth+1))
	for sim_step in odor_data:
		grid*=0
		
		miny=None
		maxy=None
		minc = None
		for source in sim_step:
			for fil in source.filaments:
				xg, yg = Odor.get_grid_coordinates(fil.x,fil.y, grid_spacing)
				cells*=0

				if(xg>=0 and xg<glength and yg>=0 and yg<gwidth):
					grid[xg][yg]+=Odor.compute_odor_filament(fil, fil.x, fil.y, Q)
					cells[xg][yg] = 1
				
				i=1
		
				while(True):
					foundOdor = False
					##new approach
					for xi in range(xg-i,xg+i+1):
						yi = yg-i
						while(xi>=0 and xi<glength and yi<=yg+i+1 and yi<gwidth):						
							if(yi>=0 and cells[xi][yi] == 0):
								xw,yw = Odor.get_world_coordinates(xi,yi, grid_spacing)
								if(distance(xw, yw, fil.x, fil.y)<=fil.radius):
									c=Odor.compute_odor_filament(fil, fil.x, fil.y, Q)
								else:
									c=0
								if(c>odor_thresh):
									foundOdor=True

								grid[xi][yi]+=c
								cells[xi][yi]=1
							yi+=1
					
					if(not foundOdor):
						break
					i+=1

		Odor.log_grid(grid, grid_spacing, filename, step)
		step+=1

def distance(x1,y1,x2,y2):
	return math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))

def main(run, ind_mapa,path, grid_spacing, odor_grid_func):
	map_file='mapa_'+str(ind_mapa)+'_run'+str(run)+'_odor_wind_data.txt'
	[map_params,wind_data,odor_data]=Odor.read_dataset(path+map_file)
	

	odor_d=[]
	for sim_step in odor_data:
		srcs=[]
		for source in sim_step:
			fils=[]
			for fil in source.filaments:
				xg, yg = Odor.get_grid_coordinates(fil.x,fil.y, grid_spacing)
				fils.append([fil.x, fil.y, fil.radius])
			srcs.append(fils)
		odor_d.append(srcs)

	odor_data = odor_d
	make_grid_filament_based(odor_data,map_params['Q'], map_params['simulation_step'], map_params['min_x'], map_params['max_x'], map_params['min_y'], map_params['max_y'], path+map_file.split('.')[0]+'_odor_grid.txt', grid_spacing, odor_grid_func)
	

    
if __name__=='__main__':	
	
	path ='../data_environment/'
	ind_mapa=1
	run=1
	while(run<30):
		main(run,ind_mapa, path)
	
		run+=1
		

