import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import itertools
from scipy.special import hermite
import time
import operator
from collections import defaultdict

#Inputs

#M targets t_i (locations in 2d space)
M = 2000
box_min = -2000
box_max = 2000

d = 2 #Dimension of space

np.random.seed(1)
t = np.random.uniform(box_min,box_max,(M,d))
target_values = np.zeros(M)

#Scale t to unit box

#N sources s_j
N = 100000

puff_mol_amt = 1.

np.random.seed(1)
s = np.random.uniform(box_min,box_max,(N,d))

#radii
np.random.seed(1)
r_sq = np.random.uniform(15,20,(N,1))

#utility functions

class TwoDGrid(object):
    #Grid on given square with unit size largest s < R : (1/R)*s = box_width
    def __init__(self,box_min,box_max,R):
        self.bounds = np.array([[box_min,box_max],[box_min,box_max]])
        n_boxes = int(math.ceil((box_max-box_min)/R))
        self.unit_width = 1/n_boxes
        self.grid_min = 0
        self.grid_max = n_boxes
        self.R = R

    def assign_box(self,location): #1D--combine to 2D
        return (np.floor((location-self.bounds[:,0])/self.R)).astype(int)


    def obtain_box_neighbors(self,x,y): #2d grid neighbors of a given grid coord
        if ((self.grid_min <= x <= self.grid_max-1) and (
            self.grid_min <= y <= self.grid_max-1)):
            xs,ys = np.meshgrid(
                np.array([x-1,x,x+1]),np.array([y-1,y,y+1]))
            neighbors = np.array([xs,ys])
            #this is an array that is 2 x 3 x 3
            if self.grid_max-1==x:
                neighbors = neighbors[:,:,:-1]
            if self.grid_min==x:
                neighbors = neighbors[:,:,1:]
            if self.grid_min==y:
                neighbors = neighbors[:,1:,:]
            if self.grid_max-1==y:
                neighbors = neighbors[:,:-1,:]
            neighbors = np.squeeze(np.reshape(neighbors,(2,-1,1)))
            return list(zip(neighbors[0,:],neighbors[1,:]))
        else:
            print('Grid neighbor error: provided box not in big box')
            sys.exit()

def compute_Gaussian(px,py,r_sq,x,y):
    return (puff_mol_amt/(np.sqrt(8*np.pi**3)*(r_sq**1.5)))*np.exp(-1*(
        ((px-x)**2+(py-y)**2)/(2*r_sq)))

#Compute the box width
epsilon = 0.01
r_sq_max = 20
R = np.sqrt(-1*np.log(epsilon*(r_sq_max**1.5)*np.sqrt(8*np.pi**3)/(N))*2*r_sq_max)
print('R is '+str(R))

#Make the grid
grid = TwoDGrid(box_min,box_max,R)

print('The grid is '+str(grid.grid_max)+' by '+str(grid.grid_max))


last = time.time()
neighbors_dct = defaultdict(list)
for (i,j) in list(itertools.product(range(grid.grid_max),range(grid.grid_max))):
    #Find all the boxes 1 box away from the target's box: these sources considered
    neighbors_dct[(i,j)] = grid.obtain_box_neighbors(i,j)

#starting here this stuff all happens each time step
start_time = time.time()

source_boxes = grid.assign_box(s)

source_grid_dict = defaultdict(list)

for source_box,source,source_r_sq in zip(source_boxes,  s,r_sq):
    #find what box it's in
    #The value is a list: [source_x,source_y,r_sq]
    source_grid_dict[tuple(source_box)].append([source[0],source[1],source_r_sq])


#do the same for all the targets -- also collect the target index
target_boxes = grid.assign_box(t)

target_grid_dict = defaultdict(list)

for target_box,target_loc,target_index in list(zip(target_boxes,t,range(M))):
    #find what box it's in
    #The value is just the position
    target_grid_dict[tuple(target_box)].append([target_loc[0],target_loc[1],target_index])
# print(time.time()-last)
# ( ?) loop through the dict again and turn each value to a numpy array?



na = np.newaxis
#Loop through the grid boxes
for (i,j) in list(itertools.product(range(grid.grid_max),range(grid.grid_max))):
#Find all the boxes 1 box away from the target's box: these sources considered
    # t= time.time()
    # print(time.time()-t)
    relevant_targets = target_grid_dict[(i,j)]
    if len(relevant_targets)>0: #only proceed if there are targets in the box
        target_x,target_y,target_indices = np.array(relevant_targets).T
        target_x,target_y = target_x[:,na],target_y[:,na]
        relevant_sources = tuple(
            source_grid_dict[neighbor] for neighbor in neighbors_dct[(i,j)]
            if len(source_grid_dict[neighbor])>0)
        source_x,source_y,r_sq = np.concatenate(relevant_sources,0).T
        source_x,source_y,r_sq = source_x[na,:],source_y[na,:],r_sq[na,:]
    #For targets, only those in the box, not the neighbor boxes
        output_array =  compute_Gaussian(
            source_x,source_y,r_sq,target_x,target_y).sum(1)
        target_values[target_indices.astype(int)] = output_array
    else:
        pass
    # print(time.time()-t)
print(time.time()-start_time)
