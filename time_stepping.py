import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

def crank_nicholson(u0, A, k, N, t0=0., Asparse=True):
    """
    Performs N steps of Crank-Nicholson time-stepping for linear problems with
    an initial condition of u0, a right-hand side function defined by A*u, and
    a timestep of k.

    Returns the entire solution history, including the initial data.
    """

    assert k > 0 
    assert N >= 0

    M = u0.size
    u = np.zeros([M, N+1])
    u[:,0] = u0
    t = t0

    An = eye(M) - k/2*A
    Arhs = eye(M) + k/2*A

    for n in range(N):
        u[:,n+1] = spsolve(An, Arhs @ u[:,n])
        t += k

    return u

def forward_euler(u0, f, k, N, t0=0.):
    """
    Performs N steps of forward Euler time-stepping with an initial
    condition of u0, a right-hand side function f, and a timestep of k.

    Returns the entire solution history, including the initial data.
    """

    assert k > 0 
    assert N >= 0

    M = u0.size
    u = np.zeros([M, N+1])
    u[:,0] = u0
    t = t0

    for n in range(N):
        u[:,n+1] = u[:,n] + k * f(t, u[:,n])
        t += k

    return u

def rk4(u0, f, k, N, t0=0.):
    """
    Performs N steps of the classic RK-4 time-stepping method with an initial
    condition of u0, a right-hand side function f, and a timestep of k.

    Returns the entire solution history, including the initial data.
    """

    assert k > 0 
    assert N >= 0

    M = u0.size
    u = np.zeros([M, N+1])
    u[:,0] = u0
    t = t0

    for n in range(N):
        fU1 = f(t, u[:,n])
        fU2 = f(t + k/2, u[:,n] + k/2 * fU1)
        fU3 = f(t + k/2, u[:,n] + k/2 * fU2)
        fU4 = f(t + k,   u[:,n] + k*fU3)

        u[:,n+1] = u[:,n] + k/6. * (fU1 + 2*fU2 + 2*fU3 + fU4)
        t += k

    return u

def multistep_22(u0, f, k, N, t0=0.):
    """
    Performs N steps of a non-0-stable multistep method (s=2). Uses RK4 as the
    initial stepping method.
    """

    assert k > 0
    assert N >= 0

    M = u0.size
    u = np.zeros([M, N+1])
    fu = np.zeros([M, N+1])
    u[:,0] = u0
    fu[:,0] = f(t0, u0)
    t = t0

    if N > 0:
        u[:,:2] = rk4(u0, f, k, 1, t0=t0)
        fu[:,1] = f(t+k, u[:,1])
        t += k 

    for n in range(1, N):
        u[:,n+1] = k*(4*fu[:,n] + 2*fu[:,n-1]) - 4*u[:,n] + 5*u[:,n-1]
        fu[:,n+1] = f(t+k, u[:,n+1])
        t += k

    return u
