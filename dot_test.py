import numpy as np
import time

iterations = 100
terms = 24

p = np.random.randn(24)

t1 = time.time()
for i in range(iterations):
    prod = np.dot(p,p)

t2 = time.time()

dt = t2-t1
print(dt/iterations)    
