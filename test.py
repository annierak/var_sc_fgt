import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import itertools
from scipy.special import hermite
import time
import operator

#For timing purposes
counter = 0
testy_1 = np.random.randn(20)
testy_2 = np.random.randn(20)

#Inputs

#M targets t_i (locations in 2d space)
M = 100
box_min = -2000
box_max = 2000

d = 2 #Dimension of space

np.random.seed(1)
t = np.random.uniform(box_min,box_max,(M,d))

p = 24

#Scale t to unit box
t = (t-box_min)/(box_max-box_min)
# plt.scatter(t[:,0],t[:,1])
# plt.show()

#N sources s_j
N = 100

np.random.seed(1)
s = np.random.uniform(box_min,box_max,(N,d))
#Scale s to unit box
s = (s-box_min)/(box_max-box_min)

#Scales delta_j for each source s_j
np.random.seed(1)
delta = np.random.uniform(0.002,box_max-box_min,(N,1))

#Scale delta to unit box
delta = delta/(box_max-box_min)

#Source weights f_j
f = np.random.uniform(1,200,(N,1))

#Choose a and b -- left to the user, see pg. 1136 of paper
#initial puff rad = 0.01 ==> delta_min = 0.0002
a = (0.0001 +  1./N)#/(box_max-box_min)
b = 0.1
epsilon = 1e-6


#utility functions
def assign_box(location,bounds,width):
    #Given outer bounds on x, y, z (large box), box width, return
    #x, y, z coordinates of the box the location is in
    #location: [x,y,z]
    #bounds : array([[xmin,xmax],[ymin,ymax],[zmin,zmax]]) etc
    box_coords = np.zeros(len(location))
    for dim in range(len(location)):
        if (bounds[dim,0]<=location[dim]<=bounds[dim,1]):
            box_coords[dim] = int(np.floor((location[dim]-bounds[dim,0])/width))
        else:
            print('Box assignment error: provided location not in big box')
            sys.exit()
    return box_coords

def distance(p1,p2):
    #N-d distance between two points
    compwise_sq_diff = [(p1[i]-p2[i])**2 for i in range(len(p2))]
    return math.sqrt(sum(compwise_sq_diff))

def compute_Gaussian(f,t,s,delta):
    return f*math.exp(-1*(distance(t,s)**2)/delta)

def box_distance_check(i1,i2,max_diff):
    #Test whether two n-tuples (lists) of ints are within a certain max difference across
    #all coordinates
    for k in range(len(i1)):
        if abs(i1[k]-i2[k])>max_diff:
            return False
        else:
            continue
    return True

def nd_itertool(d,p):
    values = range(1,p+1)
    args = tuple([values for i in range(d)])
    return itertools.product(*args)

def nd_abs(alpha):
    return sum(alpha[i] for i in range(len(alpha)))
def nd_factorial(alpha):
    elements = [math.factorial(alpha[i]) for i in range(len(alpha))]
    return reduce(operator.mul,elements,1)
def nd_exp(base,alpha):
    elements = [base[i]**alpha[i] for i in range(len(alpha))]
    return reduce(operator.mul,elements,1)

# def evaluate_nd_hermite_function(alpha,t):
#     last = time.time()
#     element_hermite_polynomials = [hermite(alpha[i])(t[i]) for i in range(len(alpha)) ]
#     output = np.exp(-1*(nd_abs(t)**2))*np.prod(element_hermite_polynomials)
#     print(time.time()-last)
#     return output
def evaluate_nd_hermite_function(H_beta_is,powers,coeff):
    # global testy_1,testy_2
    # last = time.time()
    # element_hermite_polynomials = [H_beta_is[i](t[i]) for i in range(len(t)) ]
    # element_hermite_polynomials = [np.dot(testy_1,testy_2) for i in range(2)]
    prod = 1
    for i in range(len(powers)):
        last = time.time()
        powers_cut = powers[i,0:len(H_beta_is[i])]
        prod *= np.dot(powers_cut,H_beta_is[i])
        print(time.time()-last)
    output = coeff*prod
    # output = coeff*2.5*9.5
    # print(last-time.time())
    return output
# print box_distance_check([2,4,9,5],[1,3,4,5],1)# print assign_box([4.7,9.1,29.1],[0,27,0,27,0,27],3)

class TaylorExpansion(object):
    def __init__(self,t_0,delta,p,delta_j_set,s_set,d):
        if len(delta_j_set) != len(s_set):
            print('Error: deltas and sources length mismatch')
            sys.exit()
        self.d = d
        self.t_0=t_0
        self.delta=delta
        self.p=p
        self.delta_j_set=delta_j_set
        self.s_set=s_set
        self.indexer = [x for x in nd_itertool(self.d,self.p)]
        self.C_betas = np.zeros(tuple([p for i in range(d)]))
        #This piece is for obtaining hermite function from hermite polynomial
        #precomputed for efficiency
        #These are indexed by the sources
        self.hermite_coeffs = [np.exp(-1*(nd_abs(
        (s_j-self.t_0)/(math.sqrt(delta_j)))**2))
         for delta_j,s_j in list(zip(self.delta_j_set,self.s_set))]
        #these are the N x values that are entered into the hermite series
        self.xs = [((s_j-self.t_0)/(math.sqrt(delta_j
        ))) for delta_j,s_j in list(zip(self.delta_j_set,self.s_set))]
        #These are all the powers of the above x values (dim (x value) x exponent value)
        self.powers = np.array([[np.power(el,range(self.p+1)) for el in x] for x in self.xs])
        last = time.time()
        for beta in self.indexer:
            indices = tuple([el-1 for el in beta])
            self.C_betas[indices]=self.compute_C_beta(beta)
        print('time computing C_betas: '+str(time.time()-last))
    def compute_C_beta(self,beta):
        #First, create hermite objects for the 1D components H_beta_1(), H_beta_2(), etc
        H_beta_is = [hermite(beta_i).c for beta_i in beta]
        #Then, make a list of the summands:
        #the evaluted h_beta for each source (indexed by j),
        #times the delta fraction coefficient
        summands = [(self.delta/delta_j)**(
            nd_abs(beta)/2)*evaluate_nd_hermite_function(H_beta_is,
            powers,hermite_coeff) \
             for (delta_j,powers,hermite_coeff) in \
             zip(self.delta_j_set,self.powers,self.hermite_coeffs)]
        output = (1./(nd_factorial(beta)))*np.sum(summands)
        return output
    def evaluate(self,t):
        running_sum = 0
        sq_delta = math.sqrt(self.delta)
        for beta in self.indexer:
            # last = time.time()
            indices = tuple([el-1 for el in beta])
            running_sum += self.C_betas[indices]*nd_exp(((t-self.t_0)/(sq_delta)),beta)
            # print(time.time()-last)
        return running_sum

class TargetValueCollector(object):
    '''Object class for storing target values in Gauss transform'''
    def __init__(self,M,N,t,s,f,epsilon,delta,a,b,d,p):
        self.Sf = np.zeros(M) #Summed target values
        self.t = t #target positions
        self.s = s #source positions
        self.f = f
        self.epsilon = epsilon #Desired accuracy
        self.delta = delta #source deltas
        self.a = a
        self.d = d #dimension of space
        self.p = p
        self.group_1_indices = np.where(delta<=a)[0]
        self.group_2_indices = np.where((delta>a)&(delta<b))[0]
        self.group_3_indices = np.where(delta>b)[0]
    def update_group_1(self):
        last = time.time()
        #Process for group 1 of the sources: 0 \leq \delta \leq a
        #For all of the sources in this group, add their contribution to every target
        group_1_sources = self.s[self.group_1_indices,:]
        R = math.sqrt(-1*self.a*math.log(self.epsilon))
        n_boxes = math.floor(1/R)
        box_width = 1/n_boxes
        #We create a subdivision of the box [0,1]^3 into boxes of length R
        #Obtain a list of box locations for every target
        bounds = np.array([[0,1] for i in range(d)])
        target_boxes = [assign_box(target,bounds,box_width) for target in self.t]
        #And for every source
        source_boxes = [assign_box(source,bounds,box_width) for source in group_1_sources]
        #Loop through the sources
        for source_index in range(len(group_1_sources)):
        #Find all the boxes 1 box away from the source's box (by looping through targets)
            for target_index in range(len(self.t)):
                if box_distance_check(target_boxes[target_index],\
                        source_boxes[source_index],1):
        #Add the directly computed value to the Sf collector entry corresponding
        #to that target
                    self.Sf[target_index]+= compute_Gaussian(
                    self.f[self.group_1_indices[source_index]],
                    self.t[target_index],
                    self.s[self.group_1_indices[source_index]],
                    self.delta[self.group_1_indices[source_index]])
        print('Time updating group 1:'+str(time.time()-last))
    # def update_group_2(self):
    #     #Process for group 2 of the sources: a < \delta < b
    #     #For all of the sources in this group, add their contribution to every target
    def update_group_3(self):
        last = time.time()
        #Process for group 3 of the sources: b \leq \delta
        #Create a Taylor polynomial encompassing all sources in this group
        #add the value of this polynomial at the target to every target
        t_0 = tuple([.5 for i in range(d)])
        delta_j_set = self.delta[self.group_3_indices]
        s_set = self.s[self.group_3_indices]
        delta = min(delta_j_set) #in the category
        taylorExpansion = TaylorExpansion(t_0,delta,self.p,delta_j_set,s_set,self.d)
        group_3_sources = self.s[self.group_3_indices,:]
        last1 = time.time()
        for target_index in range(len(self.t)):
            self.Sf[target_index]+=taylorExpansion.evaluate(self.t[target_index])
        print('time evaluating taylor series: '+str(time.time()-last1))
        print('Time updating group 3:'+str(time.time()-last))

targetCollector = TargetValueCollector(M,N,t,s,f,epsilon,delta,a,b,d,p)
targetCollector.update_group_1()
targetCollector.update_group_3()
# targetCollector.update_group_2()

last = time.time()
for (source,f_j,delta_j) in zip(s,f,delta):
    for target in t:
        Sf = compute_Gaussian(f_j,target,source,delta_j)
print('naive cross time',time.time()-last,counter)
