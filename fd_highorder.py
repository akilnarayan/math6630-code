# Demo of using higher-order finite difference schemes

import time

import numpy as np
import scipy as sp
from scipy import sparse as sp
from scipy.sparse import diags
from scipy.sparse import linalg as splinalg

import matplotlib.pyplot as plt

# Solving u_{xx} - eps*u = f on spatial domain x \in [0, 1] with periodic boundary conditions

eps = 1e-1
ue = lambda x: np.exp(np.sin(2*np.pi*x))
ued = lambda x: ue(x) * 2*np.pi*np.cos(2*np.pi*x)
f = lambda x: ued(x) * 2*np.pi*np.cos(2*np.pi*x) - ue(x) * 4*(np.pi)**2*np.sin(2*np.pi*x) - eps*ue(x)

# Stencils for FD approximations
ss = [[1, -2, 1],  \
      [-1/12., 16/12., -30/12., 16/12., -1/12.], \
      [1/90., -3/20., 3/2.,-49/18., 3/2., -3/20., 1/90], \
      [-1/560., 8/315., -1/5., 8/5., -205/72., 8/5., -1/5., 8/315., -1/560.],\
      [1/3150, -5/1008, 5/126, -5/21, 5/3, -5269/1800, 5/3, -5/21, 5/126, -5/1008, 1/3150]
     ]

centerinds = [1, 2, 3, 4, 5]
colors = ['m', 'b', 'r', 'k', 'g']
markers = ['o', 'v', '<', 's', 'D']

def makeA(stencil, M, h, centerind=0):
    """
    Creates a finite difference matrix, assumed to discretize the
    second derivative given the input stencil. The value
    stencil[centerind] is assumed to correspond to the point at
    which the second derivative is being taken.
    """

    v_diags = []
    stencil_offsets = [ind-centerind for ind in range(len(stencil))]

    for ind, val in enumerate(stencil):
        row = val*np.ones(M-int(np.abs(centerind-ind)))
        v_diags.append(row)

    A = diags(v_diags, stencil_offsets)/h**2

    # Now handle boundaries
    A = A.tolil()
    # First few rows
    for rowind in range(abs(stencil_offsets[0])):
        for colind in range(stencil_offsets[0]+rowind, 0):
            A[rowind,colind] = stencil[centerind + colind - rowind]/h**2

    # Last few rows
    for rowind in range(-stencil_offsets[-1],0):
        for colind in range(stencil_offsets[-1]+rowind+1):
            A[rowind,colind] = stencil[centerind + colind - rowind]/h**2

    return A.tocsr()

Ms = np.logspace(1, 2.5, 40, dtype=int)
Nruns = 10

errs = np.zeros([len(Ms), len(ss)])
times = errs.copy()

for Mind, M in enumerate(Ms):

    xj = np.linspace(0, 1, M+1)[:-1]
    h = xj[1] - xj[0]
    fx = f(xj)

    for sind, s in enumerate(ss):
        start = time.time()
        for run in range(Nruns):
            A = makeA(s, M, h, centerind=centerinds[sind]) - eps*diags([np.ones(M)], [0])
            ux = splinalg.spsolve(A, fx)
        end = time.time()

        errs[Mind, sind] = np.sqrt(h)*np.linalg.norm(ux - ue(xj))
        times[Mind, sind] = (end-start)/Nruns

plt.figure()
for sind in range(len(ss)):
    tmp = plt.scatter(times[:,sind], errs[:,sind], times[:,sind]*10000, \
                      c=colors[sind], marker=markers[sind], \
                      label='{0:d}-point stencil'.format(len(ss[sind])))

plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.xlabel('CPU time')
plt.ylabel('$\ell^2$ norm solution error')
plt.legend(frameon=False)
plt.show()
