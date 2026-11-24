"""Stationary radial-coupling quadratures for Fig. 10."""
import numpy as np


def choose_rmax(beta):
    """Upper radial integration limit used for the saved theory maps."""
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    a = 1.0 + 6.0 / max(np.sqrt(beta), 1e-9)
    b = 6.0 * (3.0 / max(beta, 1e-12)) ** (1.0 / 3.0)
    return float(np.clip(max(3.0, a, b), 3.0, 12.0))


def _radial_grid(beta, points):
    if not isinstance(points, (int, np.integer)) or points < 3:
        raise ValueError("The radial grid needs at least three points")
    r = np.linspace(0.0, choose_rmax(beta), points)
    exponent = beta * (0.5 * r ** 2 - (1.0 / 3.0) * r ** 3)
    # A constant shift cancels in normalization; use it only to avoid overflow.
    if np.max(exponent) > 600:
        exponent -= np.max(exponent)
    p0 = np.exp(exponent)
    p0 /= np.trapezoid(p0 * r, r)
    return r, p0


def radial_moments(beta, ks, points=600):
    """Mean and variance of r1, integrating the stationary radial marginal."""
    r, p0 = _radial_grid(beta, points)
    delta2 = (r[:, None] - r[None, :]) ** 2
    means, variances = [], []
    for k in ks:
        kernel = np.exp(-0.5 * beta * k * delta2)
        marginal = p0 * r * np.trapezoid(kernel * (p0 * r)[None, :], r, axis=1)
        marginal /= np.trapezoid(marginal, r)
        mean = np.trapezoid(r * marginal, r)
        means.append(mean)
        variances.append(np.trapezoid((r - mean) ** 2 * marginal, r))
    return np.asarray(means), np.asarray(variances)


def radial_covariance(beta, ks, points=700):
    """Covariance of ((r1+r2)/2)^2 and (r1-r2)^2 under the joint density."""
    r, p0 = _radial_grid(beta, points)
    r1, r2 = r[:, None], r[None, :]
    bar2, delta2 = 0.25 * (r1 + r2) ** 2, (r1 - r2) ** 2
    free_measure = (p0 * r)[:, None] * (p0 * r)[None, :]

    def integrate(values):
        return np.trapezoid(np.trapezoid(values, r, axis=1), r)

    covariances = []
    for k in ks:
        density = free_measure * np.exp(-0.5 * beta * k * delta2)
        density /= integrate(density)
        covariances.append(integrate(bar2 * delta2 * density)
                           - integrate(bar2 * density) * integrate(delta2 * density))
    return np.asarray(covariances)


def compute(betas, ks, gradient_points=600, covariance_points=700):
    """Generate the Fig. 10 gradient and covariance maps on specified grids."""
    betas, ks = np.asarray(betas, dtype=float), np.asarray(ks, dtype=float)
    if betas.ndim != 1 or betas.size == 0 or not np.isfinite(betas).all() or np.any(betas <= 0):
        raise ValueError("betas must be a nonempty vector of finite positive values")
    if (ks.ndim != 1 or ks.size < 2 or not np.isfinite(ks).all()
            or ks[0] != 0 or np.any(np.diff(ks) <= 0)):
        raise ValueError("ks must increase strictly from zero and contain at least two values")
    means, variances, covariances = [], [], []
    for beta in betas:
        mean, variance = radial_moments(beta, ks, gradient_points)
        means.append(mean / mean[0])
        variances.append(variance / variance[0])
        covariances.append(radial_covariance(beta, ks, covariance_points))
    means, variances = np.asarray(means), np.asarray(variances)
    result = {
        "betas": betas.copy(), "ks": ks.copy(),
        "grads": np.gradient(means, ks, axis=1, edge_order=1),
        "grads_v": np.gradient(variances, ks, axis=1, edge_order=1),
        "covs": np.asarray(covariances),
    }
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Nonfinite quadrature result")
    return result


def uncoupled_covariance(beta, upper):
    """Factorized k=0 covariance from independent one-dimensional moments."""
    from scipy.integrate import quad

    def integral(order):
        return quad(lambda r: r**(order + 1) * np.exp(beta * (r*r/2 - r**3/3)),
                    0, upper, epsabs=1e-11, epsrel=1e-11)[0]
    m0 = integral(0)
    m1, m2, m4 = (integral(order)/m0 for order in (1, 2, 4))
    return .5*m4 - 1.5*m2**2 + m1**4
