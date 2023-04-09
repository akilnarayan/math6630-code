"""
Some demo animations for solving the heat equation.
"""

import time

import numpy as np
from scipy.sparse import diags

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time_stepping import forward_euler, rk4, crank_nicholson

# Solving u_t = a u_{x x} up to time T = 1 on spatial domain x \in [0, 2\pi]
t0 = 0.
a = 5
T = 0.5
u0f = lambda x: np.exp(np.sin(4*x))

M = 100 # Number of spatial points
N = 10000 # Number of temporal steps

xj = np.linspace(0, 2*np.pi, M+1)[:-1]

h = xj[1] - xj[0]
k = float(T)/N

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

# Run the computation several times for timing
Nruns = 5
fe_time = time.time()
for j in range(Nruns):
    u_fe = forward_euler(u0, f, k, N)
fe_time = (time.time() - fe_time)/Nruns

rk_time = time.time()
for j in range(Nruns):
    u_rk = rk4(u0, f, k, N)
rk_time = (time.time() - rk_time)/Nruns

cn_time = time.time()
for j in range(Nruns):
    u_cn = crank_nicholson(u0, A, k, N)
cn_time = (time.time() - cn_time)/Nruns

## Animate results
fig = plt.figure()

# Forward Euler
fe_ax = plt.subplot(1,3,1)
fe_line = fe_ax.plot([], [], 'r')[0]
time_template = 'time = %.3fs'
time_text = fe_ax.text(0.05, 0.90, '', transform=fe_ax.transAxes)
fe_ax.set_title('Forward Euler, CPU time$\sim$ {0:1.2e}s'.format(fe_time))

# RK4 
rk_ax = plt.subplot(1,3,2)
rk_line = rk_ax.plot([], [], 'r')[0]
time_template = 'time = %.3fs'
rk_ax.set_title('Runge-Kutta 4, CPU time$\sim$ {0:1.2e}s'.format(rk_time))

# Crank-Nicolson
cn_ax = plt.subplot(1,3,3)
cn_line = cn_ax.plot([], [], 'r')[0]
time_template = 'time = %.3fs'
cn_ax.set_title('Crank-Nicolson, CPU time$\sim$ {0:1.2e}s'.format(cn_time))

def init():
    fe_ax.set_xlim(0, 2*np.pi)
    fe_ax.set_ylim(0, 3)
    rk_ax.set_xlim(0, 2*np.pi)
    rk_ax.set_ylim(0, 3)
    cn_ax.set_xlim(0, 2*np.pi)
    cn_ax.set_ylim(0, 3)
    time_text.set_text('')
    return fe_line, time_text

def anim_update(step):
    fe_line.set_data(xj, u_fe[:,step])
    rk_line.set_data(xj, u_rk[:,step])
    cn_line.set_data(xj, u_cn[:,step])
    time_text.set_text(time_template % (step*k))
    return fe_line, rk_line, cn_line, time_text

ani = animation.FuncAnimation(fig, anim_update, range(N+1), \
                  interval=50, blit=True, init_func=init, repeat_delay=300)

plt.show()
