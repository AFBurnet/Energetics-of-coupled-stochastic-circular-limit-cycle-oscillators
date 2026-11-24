"""Stationary radial quadrature for the Cartesian deformation maps."""
import numpy as np
from scipy.special import ive


def radius_limit(beta):
    """Radial integration domain used for the saved maps."""
    a = 1.0 + 6.0 / max(np.sqrt(beta), 1e-9)
    b = 6.0 * (3.0 / max(beta, 1e-12)) ** (1.0 / 3.0)
    return float(np.clip(max(3.0, a, b), 3.0, 12.0))


def _grid(beta, radial_points):
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive.')
    if int(radial_points) != radial_points or radial_points < 3:
        raise ValueError('radial_points must be an integer of at least three.')
    r = np.linspace(0.0, radius_limit(beta), int(radial_points))
    p0 = np.exp(beta * (0.5 * r**2 - r**3 / 3.0))
    p0 /= np.trapezoid(p0 * r, r)
    r1, r2 = np.meshgrid(r, r, indexing='ij')
    return r, p0 * r, r1, r2


def _kernel(beta, k, r1, r2):
    if not np.isfinite(k) or k < 0:
        raise ValueError('k must be finite and nonnegative.')
    return np.exp(-0.5 * beta * k * (r1 - r2)**2) * ive(0, beta * k * r1 * r2)


def radial_moments(beta, k, radial_points=600):
    """Return the mean and variance of either oscillator's radius."""
    r, free, r1, r2 = _grid(beta, radial_points)
    kernel = _kernel(beta, k, r1, r2)
    density = free * np.trapezoid(kernel * free[None, :], r, axis=1)
    density /= np.trapezoid(density, r)
    mean = np.trapezoid(r * density, r)
    variance = np.trapezoid((r - mean)**2 * density, r)
    return float(mean), float(variance)


def grid_values(values, name, *, positive=False, minimum=1):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or len(values) < minimum or
            not np.isfinite(values).all() or np.any(np.diff(values) <= 0) or
            np.any(values <= 0 if positive else values < 0)):
        raise ValueError(f'{name} must be a finite, increasing one-dimensional grid.')
    return values


def compute(betas, small_ks, large_ks, radial_points=600):
    """Calculate the three derivative maps, normalized by uncoupled moments."""
    betas = grid_values(betas, 'betas', positive=True)
    small_ks = grid_values(small_ks, 'small_ks', minimum=2)
    large_ks = grid_values(large_ks, 'large_ks', minimum=2)
    if small_ks[0] != 0.0:
        raise ValueError('The small-coupling grid must begin at zero.')
    output = dict(betas=betas, small_ks=small_ks, large_ks=large_ks)
    small, large = [], []
    for beta in betas:
        moments = np.array([radial_moments(beta, k, radial_points) for k in small_ks])
        baseline = moments[0]
        small.append(np.gradient(moments/baseline, small_ks, axis=0))
        means = np.array([radial_moments(beta, k, radial_points)[0] for k in large_ks])
        large.append(np.gradient(means/baseline[0], large_ks))
    small = np.asarray(small)
    output.update(small_mean_gradient=small[:, :, 0],
                  small_variance_gradient=small[:, :, 1],
                  large_mean_gradient=np.asarray(large))
    return output


def uncoupled_moments(beta, upper):
    """Independent adaptive mean and variance for the uncoupled radial density."""
    from scipy.integrate import quad

    def integrate(function):
        return quad(lambda r: function(r)*r*np.exp(beta*(r*r/2-r**3/3)),
                    0, upper, epsabs=1e-11, epsrel=1e-11)[0]
    z = integrate(lambda r: 1.)
    mean = integrate(lambda r: r)/z
    variance = integrate(lambda r: (r-mean)**2)/z
    return mean, variance
