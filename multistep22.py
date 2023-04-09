"""
Demo of an s=2 multistep methods.
"""

import numpy as np
from matplotlib import pyplot as plt

from ode_examples import lorenz
from time_stepping import forward_euler, rk4, multistep_22

f, u0 = lorenz()

t0 = 0.
T = 10
N = 1000000

k = (T-t0)/N
t = np.linspace(t0, t0 + k*N, N+1)

u_fe = forward_euler(u0, f, k, N, t0)
u_rk4 = rk4(u0, f, k, N, t0)
u_ms = multistep_22(u0, f, k, N, t0)

plt.subplot(131)
plt.plot(t, u_fe.T, 'r')
plt.title('Forward Euler')
plt.ylabel('Components of $u$')
plt.xlabel('Time $t$')

plt.subplot(132)
plt.plot(t, u_rk4.T, 'r')
plt.title('Runge-Kutta 4')
plt.ylabel('Components of $u$')
plt.xlabel('Time $t$')

plt.subplot(133)
u_ms[~np.isfinite(u_ms)] = np.finfo(u_ms.dtype).max
plt.plot(t, np.abs(u_ms).T, 'r')
plt.ylim(ymin=1e-1, ymax=1e200)
plt.yscale('log')
plt.title('Multistep (2,2)')
plt.ylabel('Components of $u$')
plt.xlabel('Time $t$')

plt.show()
