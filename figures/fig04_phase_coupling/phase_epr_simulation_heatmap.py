"""Optimized phase-pair trajectories for the Figure 4 minimum-EPR heatmap.

Use figure4.py heatmap for settings, independent rows and replica summaries;
figure4.py plot renders the saved paper results. The scalar Heun kernel preserves
its full-drift accepted-state spatial-midpoint estimator and seed ordering.
"""
import numpy as np
from numba import njit, prange


@njit(inline="always")
def drift(x1, y1, x2, y2, k, w1, w2):
    """Full Cartesian drift; the phase sine is zero at a zero radius."""
    r1, r2 = np.hypot(x1, y1), np.hypot(x2, y2)
    denominator = r1*r2
    sine = (x1*y2-y1*x2)/denominator if denominator > 0 else 0.0
    v1, v2 = -w1+k*sine, -w2-k*sine
    return (-v1*y1+(1-r1)*x1, v1*x1+(1-r1)*y1,
            -v2*y2+(1-r2)*x2, v2*x2+(1-r2)*y2)


@njit
def trajectory_epr(k, T, w1, w2, dt, n_burn, n_measure, seed):
    """Heun path with the full drift at accepted-state spatial midpoints."""
    np.random.seed(seed)
    th1, th2 = 2*np.pi*np.random.random(), 2*np.pi*np.random.random()
    x1, y1, x2, y2 = np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2)
    # Match the frequency convention of the shared phase-coupling core.
    omega_bar, delta_omega = (w1+w2)/2, w2-w1
    w1, w2 = omega_bar-delta_omega/2, omega_bar+delta_omega/2
    noise_scale = np.sqrt(2*T*dt)
    heat = 0.0
    for step in range(n_burn+n_measure):
        f1, f2, f3, f4 = drift(x1, y1, x2, y2, k, w1, w2)
        z1 = noise_scale*np.random.normal()
        z2 = noise_scale*np.random.normal()
        z3 = noise_scale*np.random.normal()
        z4 = noise_scale*np.random.normal()
        p1, p2, p3, p4 = drift(x1+dt*f1+z1, y1+dt*f2+z2,
                               x2+dt*f3+z3, y2+dt*f4+z4, k, w1, w2)
        # Correct all four coordinates with the same predictor noise.
        a1, b1 = x1+0.5*dt*(f1+p1)+z1, y1+0.5*dt*(f2+p2)+z2
        a2, b2 = x2+0.5*dt*(f3+p3)+z3, y2+0.5*dt*(f4+p4)+z4
        if step >= n_burn:
            m1, m2, m3, m4 = drift((x1+a1)/2, (y1+b1)/2,
                                   (x2+a2)/2, (y2+b2)/2, k, w1, w2)
            heat += m1*(a1-x1)+m2*(b1-y1)+m3*(a2-x2)+m4*(b2-y2)
        x1, y1, x2, y2 = a1, b1, a2, b2
    return heat/(T*n_measure*dt)


@njit(parallel=True)
def simulate_row(beta, deltas, ks, wbar, dt, n_burn, n_measure, n_rep, seed):
    """Independent runs at one beta; each parallel job has its own seed."""
    samples = np.empty((len(deltas), len(ks), n_rep))
    for job in prange(len(deltas)*len(ks)*n_rep):
        rep = job % n_rep
        ik = (job // n_rep) % len(ks)
        jd = job // (n_rep*len(ks))
        w1, w2 = wbar-deltas[jd]/2, wbar+deltas[jd]/2
        samples[jd, ik, rep] = trajectory_epr(
            ks[ik], 1/beta, w1, w2, dt, n_burn, n_measure, seed+job)
    return samples
