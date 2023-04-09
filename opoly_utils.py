import numpy as np
from numpy.linalg import eigh
from scipy.special import gamma, gammaln

def poly_eval(x, n, ab, d=0):
    # Evaluates univariate orthonormal polynomials given their
    # three-term recurrence coefficients ab (a, b).
    #
    # Evaluates the d'th derivative. (Default = 0)
    #
    # Returns a numel(x) x numel(n) x numel(d) array.

    nmax = np.max(n)

    p = np.zeros(x.shape + (nmax+1,))
    xf = x.flatten()

    p[:, 0] = 1/ab[0, 1]

    if nmax > 0:
        p[:, 1] = 1/ab[1, 1] * ((xf - ab[1, 0])*p[:, 0])

    for j in range(2, nmax+1):
        p[:, j] = 1/ab[j, 1] * ((xf - ab[j, 0])*p[:, j-1] - ab[j-1, 1]*p[:, j-2])

    if type(d) == int:
        d = [d]

    preturn = np.zeros([p.shape[0], n.size, len(d)])

    def assign_p_d(dval, parray):
        """
        Assigns dimension 2 of the nonlocal array preturn according to values
        in the derivative list d.
        """
        nonlocal preturn

        indlocations = [i for i, val in enumerate(d) if val == dval]
        for i in indlocations:
            preturn[:, :, i] = parray[:, n.flatten()]

    assign_p_d(0, p)

    for qd in range(1, max(d)+1):

        pd = np.zeros(p.shape)

        for qn in range(qd, nmax+1):
            if qn == qd:
                # The following is an over/underflow-resistant way to
                # compute ( qd! * kappa_{qd} ), where qd is the
                # derivative order and kappa_{qd} is the leading-order
                # coefficient of the degree-qd orthogonal polynomial.
                # The explicit formula for the lading coefficient of the
                # degree-qd orthonormal polynomial is prod(1/b[j]) for
                # j=0...qd.
                pd[:, qn] = np.exp(gammaln(qd+1) - np.sum(np.log(ab[:(qd+1), 1])))
            else:
                pd[:, qn] = 1/ab[qn, 1] * ((xf - ab[qn, 0]) * pd[:, qn-1] - ab[qn-1, 1] * pd[:, qn-2] + qd*p[:, qn-1])

        assign_p_d(qd, pd)

        p = pd

    if len(d) == 1:
        return preturn.squeeze(axis=2)
    else:
        return preturn


def jacobi_matrix_driver(ab, N):
    """
    Returns the N x N jacobi matrix associated to the input recurrence
    coefficients ab. (Requires ab.shape[0] >= N+1.)
    """

    return np.diag(ab[1:N, 1], k=1) + np.diag(ab[1:(N+1), 0], k=0) + np.diag(ab[1:N, 1], k=-1)

def gauss_quadrature(ab, N):
    """
    Computes the N-point Gauss quadrature rule associated to the
    recurrence coefficients ab. (Requires ab.shape[0] >= N+1.)
    """

    from numpy.linalg import eigh

    if N > 0:
        lamb, v = eigh(jacobi_matrix_driver(ab, N))
        return lamb, ab[0, 1]**2 * v[0, :]**2
    else:
        return np.zeros(0), np.zeros(0)

def jacobi_recurrence(N, alpha, beta):
    """
    Returns the first N+1 recurrence coefficient pairs for the (alpha, beta)
    Jacobi family
    """
    if N < 1:
        ab = np.ones((1, 2))
        ab[0, 0] = 0
        ab[0, 1] = np.exp((alpha + beta + 1.) * np.log(2.) +
                          gammaln(alpha + 1.) + gammaln(beta + 1.) -
                          gammaln(alpha + beta + 2.))
        ab = np.sqrt(ab)
        return ab

    ab = np.ones((N+1, 2)) * np.array([beta**2. - alpha**2., 1.])

    # Special cases
    ab[0, 0] = 0.
    ab[1, 0] = (beta - alpha) / (alpha + beta + 2.)
    ab[0, 1] = np.exp((alpha + beta + 1.) * np.log(2.) +
                      gammaln(alpha + 1.) + gammaln(beta + 1.) -
                      gammaln(alpha + beta + 2.))

    ab[1, 1] = 4. * (alpha + 1.) * (beta + 1.) / (
                   (alpha + beta + 2.)**2 * (alpha + beta + 3.))

    if N > 1:
        ab[1, 1] = 4. * (alpha + 1.) * (beta + 1.) / (
                   (alpha + beta + 2.)**2 * (alpha + beta + 3.))

        ab[2, 0] /= (2. + alpha + beta) * (4. + alpha + beta)
        inds = 2
        ab[2, 1] = 4 * inds * (inds + alpha) * (inds + beta) * (inds + alpha + beta)
        ab[2, 1] /= (2. * inds + alpha + beta)**2 * (2. * inds + alpha + beta + 1.) * (2. * inds + alpha + beta - 1)

    if N > 2:
        inds = np.arange(2., N+1)
        ab[3:, 0] /= (2. * inds[:-1] + alpha + beta) * (2 * inds[:-1] + alpha + beta + 2.)
        ab[2:, 1] = 4 * inds * (inds + alpha) * (inds + beta) * (inds + alpha + beta)
        ab[2:, 1] /= (2. * inds + alpha + beta)**2 * (2. * inds + alpha + beta + 1.) * (2. * inds + alpha + beta - 1)

    ab[:, 1] = np.sqrt(ab[:, 1])

    return ab

def legendre_recurrence(N):
    """
    Returns the first N+1 recurrence coefficient pairs for the (alpha, beta)
    Legendre family.
    """

    return jacobi_recurrence(N, 0., 0.)

def hermite_recurrence(N, mu=0.):
    ab = np.zeros((N+1, 2))
    ab[0, 1] = gamma(mu + 1/2)
    ab[1:, 1] = 1/2 * np.arange(1, N+1)
    ab[np.arange(N+1) % 2 == 1, 1] += mu
    ab[:, 1] = np.sqrt(ab[:, 1])

    return ab
