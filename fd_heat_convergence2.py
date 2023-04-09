"""
Demo for temporal and spatial convergence for the heat equation.

Produces "more expected" convergence plots
"""

import time

import numpy as np
from scipy.sparse import diags

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time_stepping import forward_euler, rk4

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# Solving u_t = a u_{x x} up to time T = 1 on spatial domain x \in [0, 2\pi]
t0 = 0.
a = 1
T = 0.5
u0f = lambda x: np.sin(2*x)
uef = lambda x: np.exp(-4*T)*np.sin(2*x)

## Temporal convergence 
Nruns = 5
Ns = np.array(np.logspace(3, 5, Nruns), dtype=int)

M0 = 100 # Number of spatial points
fe_err_time = np.zeros(Nruns)

for ind, N in enumerate(Ns):
    # Determine spatial resolution for this N
    k = float(T-t0)/N

    h = np.sqrt(2*k)
    M0 = int(np.floor(2*np.pi/h))

    xj = np.linspace(0, 2*np.pi, M0+1)[:-1]

    h = xj[1] - xj[0]

    # Form a sparse matrix corresponding to the second derivative operator
    main = -2*np.ones(M0)
    off = 1*np.ones(M0-1)
    A = a*diags([main, off, off], [0, 1, -1])/h**2
    A = A.tolil()
    A[0,-1] = a/h**2
    A[-1,0] = a/h**2

    def f(t,u):
        """
        Semi-discrete form for the heat equation based on a centered 3-point
        discretization of the second derivative.
        """
        return A @ u

    u0 = u0f(xj)
    ue = uef(xj)

    k = float(T)/N

    u_fe = forward_euler(u0, f, k, N)

    fe_err_time[ind] = np.linalg.norm(u_fe[:,-1] - ue)/np.sqrt(M0)

## Spatial
Nruns = 5
Ms = np.array(np.logspace(1.5, 3, Nruns), dtype=int)

# Choose time resolution as smallest value for stability
h0 = 2*np.pi/float(Ms[-1])
k = h0**2/6.
N0 = int(np.ceil(T/k))

fe_err_space = np.zeros(Nruns)
rk_err_space = np.zeros(Nruns)

for ind, M in enumerate(Ms):

    xj = np.linspace(0, 2*np.pi, M+1)[:-1]

    h = xj[1] - xj[0]

    # Form a sparse matrix corresponding to the second derivative operator
    main = -2*np.ones(M)
    off = 1*np.ones(M-1)
    A = a*diags([main, off, off], [0, 1, -1])/h**2
    A = A.tolil()
    A[0,-1] = a/h**2
    A[-1,0] = a/h**2

    def f(t,u):
        """
        Semi-discrete form for the heat equation based on a centered 3-point
        discretization of the second derivative.
        """
        return A @ u

    u0 = u0f(xj)
    ue = uef(xj)

    # Run the computation several times for timing
    u_fe = forward_euler(u0, f, k, N0)
    u_rk = rk4(u0, f, k, N0)

    fe_err_space[ind] = np.linalg.norm(u_fe[:,-1] - ue)/np.sqrt(M)
    rk_err_space[ind] = np.linalg.norm(u_rk[:,-1] - ue)/np.sqrt(M)

fig = plt.figure()
plt.subplot(2,2,1)
plt.loglog(Ns, fe_err_time, 'r.-')
scale = fe_err_time[0]*(Ns[0]**1)
h = plt.loglog(Ns, scale/(Ns)**1, 'k:')
plt.legend(h, ['First-order convergence',], frameon=False)
plt.xlabel('Time steps $N = T/\Delta t$')
plt.ylabel('Normalized $\ell^2$ error at time $T$')
plt.title('Forward Euler'.format(M0))

plt.subplot(2,2,3)
plt.loglog(Ms, fe_err_space, 'r.-')
scale = fe_err_space[0]*(Ms[0]**2)
h = plt.loglog(Ms, scale/(Ms)**2, 'k:')
plt.legend(h, ['Second-order convergence',], frameon=False)
plt.xlabel('Spatial points $M \sim 1/\Delta x$')
plt.ylabel('Normalized $\ell^2$ error at time $T$')
plt.title('Forward Euler, $N={0:d}$ temporal points'.format(N0))

plt.subplot(2,2,4)
plt.loglog(Ms, rk_err_space, 'r.-')
scale = rk_err_space[0]*(Ms[0]**2)
h = plt.loglog(Ms, scale/(Ms)**2, 'k:')
plt.legend(h, ['Second-order convergence',], frameon=False)
plt.xlabel('Spatial points $M \sim 1/\Delta x$')
plt.ylabel('Normalized $\ell^2$ error at time $T$')
plt.title('Runge-Kutta 4, $N={0:d}$ temporal points'.format(N0))

fig.tight_layout()

plt.show()
