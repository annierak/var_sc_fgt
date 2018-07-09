from itertools import product
import numpy as np
import time
d = 3
p = 4
values = range(1,p+1)
args = tuple([values for i in range(d)])
betas = np.array([x for x in product(*args)])
C_betas = np.zeros(tuple([p for i in range(d)]))
print(np.shape(C_betas))

for beta in betas:
    indices = tuple([el-1 for el in beta])
    C_betas[indices] = np.prod(beta)

print C_betas
