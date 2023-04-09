import numpy as np

def lorenz():
    """
    Returns the ODE system corresponding to the Lorenz system.
    """

    sigma = 10.
    beta = 8/3.
    rho = 28.

    def florenz(t,u):

        fu = np.zeros(3)
        fu[0] = sigma*(u[1]-u[0])
        fu[1] = u[0] * (rho - u[2]) - u[1]
        fu[2] = u[0]*u[1] - beta*u[2]

        return fu

    u0 = np.zeros(3)
    u0 = np.ones(3)

    return florenz, u0
