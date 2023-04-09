# Quick demonstration of computing Gauss quadrature

import numpy as np
from matplotlib import pyplot as plt

from opoly_utils import legendre_recurrence, hermite_recurrence
from opoly_utils import poly_eval, gauss_quadrature

N = 50

abL = legendre_recurrence(N)

## Legendre quadrature
XL = np.zeros([N, N])
WL = np.zeros([N, N])
for n in range(N):
    XL[:n+1,n], WL[:n+1,n] = gauss_quadrature(abL, n+1)

## Legendre DFT analogue
kappaL = np.zeros(N)
for n in range(N):
    V = poly_eval(XL[:n+1,n], np.arange(n+1), abL)
    Vtilde = np.diag(np.sqrt(WL[:n+1,n])) @ V
    kappaL[n] = np.linalg.cond(Vtilde)


## Hermite quadrature
abH = hermite_recurrence(N)
XH = np.zeros([N, N])
WH = np.zeros([N, N])

for n in range(N):
    XH[:n+1,n], WH[:n+1,n] = gauss_quadrature(abH, n+1)

## Hermite DFT analogue
kappaH = np.zeros(N)
for n in range(N):
    V = poly_eval(XH[:n+1,n], np.arange(n+1), abH)
    Vtilde = np.diag(np.sqrt(WH[:n+1,n])) @ V
    kappaH[n] = np.linalg.cond(Vtilde)

# Quadrature figure
plt.figure()
plt.subplot(1,2,1)
for n in range(N):
    plt.plot(XL[:n+1,n], (n+1)*np.ones(n+1), 'r.')

plt.ylabel('Number of nodes $N$')
plt.xlabel('Quadrature nodes, $x_n$, $n=1, \ldots, N$')
plt.title('Legendre-Gauss nodes')

plt.subplot(1,2,2)
for n in range(N):
    plt.plot(XH[:n+1,n], (n+1)*np.ones(n+1), 'r.')

plt.ylabel('Number of nodes $N$')
plt.xlabel('Quadrature nodes, $x_n$, $n=1, \ldots, N$')
plt.title('Hermite-Gauss nodes')

# Conditioning figure
plt.figure()
plt.subplot(1,2,1)
plt.plot(np.arange(1, N+1), kappaL, 'b.-')

plt.ylabel('Condition number of size-$N$ DFT analogue')
plt.xlabel('$N$')
plt.title('Legendre DFT analogue')
plt.gca().set_ylim(bottom=0., top=2.)

plt.subplot(1,2,2)
plt.plot(np.arange(1, N+1), kappaH, 'b.-')

plt.ylabel('Condition number of size-$N$ DFT analogue')
plt.xlabel('$N$')
plt.title('Hermite DFT analogue')
plt.gca().set_ylim(bottom=0., top=2.)

# Weights figure
fig = plt.figure()
ax = fig.add_subplot(1,2,1, projection='3d')

for n in range(N):
    plt.plot(XL[:n+1,n], (n+1)*np.ones(n+1), (n+1)*WL[:n+1,n], 'r.')

ax.set_ylabel('Number of nodes $N$')
ax.set_xlabel('Quadrature nodes, $x_n$, $n=1, \ldots, N$')
ax.set_zlabel('Normalized quadrature weights, $n w_n$, $n=1, \ldots, N$')
plt.title('Legendre-Gauss nodes and weights')

ax = fig.add_subplot(1,2,2, projection='3d')

for n in range(N):
    plt.plot(XH[:n+1,n], (n+1)*np.ones(n+1), (n+1)*WH[:n+1,n], 'r.')

ax.set_ylabel('Number of nodes $N$')
ax.set_xlabel('Quadrature nodes, $x_n$, $n=1, \ldots, N$')
ax.set_zlabel('Normalized quadrature weights, $n w_n$, $n=1, \ldots, N$')
plt.title('Hermite-Gauss nodes and weights')

plt.show()
