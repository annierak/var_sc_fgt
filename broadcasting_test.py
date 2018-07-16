import numpy as np
import math

puff_mol_amt = 1

source_x = source_y = r_sq = np.random.rand(1,11)
target_x = target_y = np.random.rand(4,1)

def compute_Gaussian(px,py,r_sq,x,y):
    return (puff_mol_amt/(np.sqrt(8*np.pi**3)*(r_sq**1.5)))*np.exp(-1*(
        ((px-x)**2+(py-y)**2)/(2*r_sq)))

def compute_Gaussian_(px,py,r_sq,x,y):
    return -1*(
        ((px-x)**2+(py-y)**2)/(2*r_sq))

print(compute_Gaussian(source_x,source_y,r_sq,target_x,target_y))
