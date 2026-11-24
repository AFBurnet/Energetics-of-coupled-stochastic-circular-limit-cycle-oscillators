"""Effective frequency estimates for single-coordinate Cartesian coupling.

The additive Cartesian noise has variance 2 dt / beta in each coordinate.
Heun uses the same increment in predictor and corrector. Clockwise phase winding is
accumulated at every integration step. A
statistical replica is one independent N-oscillator system; its N oscillators
are averaged together, never counted as independent samples.
"""

import math

import numpy as np
from numba import njit, prange

INITIALIZATIONS = ('random', 'synchronized', 'death')


def death_branch(k, omega):
    """Exact small-a N=2 branch, with both Jacobian sectors checked.

    a=1-r=omega**2/(k+sqrt(k**2-omega**2)); existence and strict
    deterministic stability are separate. Marginal eigenvalues are not stable.
    """
    k, omega = float(k), float(omega)
    if not math.isfinite(k) or k < 0 or not math.isfinite(omega) or omega <= 0:
        raise ValueError('Require finite k>=0 and omega>0')
    result = dict(exists=False, stable=False, number=2, k=k, omega=omega)
    if k < omega:
        return result
    a = omega**2/(k+math.sqrt(max(0., k*k-omega*omega)))
    radius = 1-a
    if radius <= 0:
        return result
    direction = np.array([a, omega])/math.hypot(a, omega)
    point = radius*direction
    symmetric = np.array([[a, omega], [-omega, a]])-radius*np.outer(direction, direction)
    antisymmetric = symmetric-np.diag([2*k, 0.])
    eig = np.concatenate((np.linalg.eigvals(symmetric), np.linalg.eigvals(antisymmetric)))
    maximum = float(eig.real.max())
    tolerance = 64*np.finfo(float).eps*max(1., k, omega)
    result.update(exists=True, stable=bool(maximum < -tolerance), a=a,
                  radius=radius, points=np.array([point, -point]),
                  maximum_real_eigenvalue=maximum)
    return result


def stability_threshold(omega=1.):
    """N=2 small-a Hopf threshold, for omega>=1/sqrt(8).

    The restriction avoids extending the simple trace criterion through
    additional low-frequency stationary bifurcations. The Figure 7 zoom uses
    omega=1, giving k=5/3 and r=2/3.
    """
    omega = float(omega)
    if not math.isfinite(omega) or omega < 1/math.sqrt(8):
        raise ValueError('Simple threshold formula requires omega>=1/sqrt(8)')
    return (1+9*omega*omega)/6


@njit(cache=True)
def _heun_step(x, y, k, omega, dt, noise_x, noise_y, px, py, xn, yn):
    number, sx = len(x), 0.
    for i in range(number):
        sx += x[i]
    spx = 0.
    for i in range(number):
        radius = math.hypot(x[i], y[i])
        fx = (1-radius)*x[i]+omega*y[i]+k*(sx-number*x[i])
        fy = (1-radius)*y[i]-omega*x[i]
        px[i], py[i] = x[i]+dt*fx+noise_x[i], y[i]+dt*fy+noise_y[i]
        spx += px[i]
        if not (math.isfinite(px[i]) and math.isfinite(py[i])):
            return 1
    for i in range(number):
        radius = math.hypot(x[i], y[i])
        rp = math.hypot(px[i], py[i])
        fx = (1-radius)*x[i]+omega*y[i]+k*(sx-number*x[i])
        fy = (1-radius)*y[i]-omega*x[i]
        gx = (1-rp)*px[i]+omega*py[i]+k*(spx-number*px[i])
        gy = (1-rp)*py[i]-omega*px[i]
        xn[i] = x[i]+dt*(fx+gx)/2+noise_x[i]
        yn[i] = y[i]+dt*(fy+gy)/2+noise_y[i]
        if not (math.isfinite(xn[i]) and math.isfinite(yn[i])):
            return 2
    return 0


@njit(cache=True, parallel=True)
def _simulate(number, beta, k, omega, dt, burn_steps, block_steps, seeds,
              initial, fixed_x, fixed_y):
    nrep, nblocks = len(seeds), len(block_steps)
    individual = np.zeros((nrep, nblocks, number))
    diagnostic = np.zeros((nrep, 2))
    flags = np.zeros(nrep, dtype=np.int64)
    noise = math.sqrt(2*dt/beta)
    total_steps = burn_steps+block_steps.sum()
    for rep in prange(nrep):
        np.random.seed(seeds[rep])
        x, y = np.empty(number), np.empty(number)
        nx, ny = np.empty(number), np.empty(number)
        px, py = np.empty(number), np.empty(number)
        xn, yn = np.empty(number), np.empty(number)
        common = 2*math.pi*np.random.random()
        inversion = 1. if np.random.random() < .5 else -1.
        for i in range(number):
            angle = 2*math.pi*np.random.random() if initial == 0 else common
            x[i], y[i] = math.cos(angle), math.sin(angle)
            if initial == 2:
                sign = inversion if i == 0 else -inversion
                x[i], y[i] = sign*fixed_x, sign*fixed_y
        large_count = origin_count = 0.
        block = block_count = 0
        failed = False
        for step in range(total_steps):
            for i in range(number):
                nx[i], ny[i] = noise*np.random.normal(), noise*np.random.normal()
            flag = _heun_step(x, y, k, omega, dt, nx, ny, px, py, xn, yn)
            if flag:
                flags[rep] = flag
                failed = True
                break
            if step >= burn_steps:
                for i in range(number):
                    # Keep principal increments and count exact-origin endpoints separately.
                    angle = -math.atan2(x[i]*yn[i]-y[i]*xn[i], x[i]*xn[i]+y[i]*yn[i])
                    individual[rep, block, i] += angle
                    radius = math.hypot(xn[i], yn[i])
                    old_radius = math.hypot(x[i], y[i])
                    if abs(angle) > math.pi/2:
                        large_count += 1
                    if radius == 0 or old_radius == 0:
                        origin_count += 1
                block_count += 1
                if block_count == block_steps[block]:
                    scale = block_steps[block]*dt
                    for i in range(number):
                        individual[rep, block, i] /= scale*omega
                    block_count, block = 0, block+1
            for i in range(number):
                x[i], y[i] = xn[i], yn[i]
        if not failed:
            diagnostic[rep] = np.array([large_count, origin_count])
    return individual, diagnostic, flags


def _integer(value, name, minimum):
    if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != value or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def run_frequency(config):
    """Run a reproducible independent-system ensemble and return small arrays.

    Required configuration: number,beta,k,omega,dt,burn,measure,replicas,blocks,
    seed,initial. Durations are physical times; measurement has >=blocks steps.
    SeedSequence spawns deterministic seeds, invariant under Numba thread count.
    Block time edges record the actual rounded durations for weighted estimates.
    Diagnostic columns count increments above pi/2 and exact-origin endpoints.
    """
    cfg = dict(config)
    number = _integer(cfg['number'], 'number', 2)
    replicas = _integer(cfg['replicas'], 'replicas', 2)
    blocks = _integer(cfg['blocks'], 'blocks', 1)
    seed = _integer(cfg.get('seed', 0), 'seed', 0)
    if seed >= 2**32:
        raise ValueError('seed must be <2**32')
    beta, k, omega, dt, burn, measure = (float(cfg[name]) for name in
                                      ('beta', 'k', 'omega', 'dt', 'burn', 'measure'))
    if not all(math.isfinite(v) for v in (beta, k, omega, dt, burn, measure)):
        raise ValueError('All physical parameters and durations must be finite')
    if beta <= 0 or k < 0 or omega <= 0 or dt <= 0 or burn < 0 or measure <= 0:
        raise ValueError('Require beta,omega,dt,measure>0 and k,burn>=0')
    if dt*(1+number*k+omega) >= 1:
        raise ValueError('Timestep too coarse for the fastest deterministic scale')
    initial = cfg.get('initial', 'random')
    if initial not in INITIALIZATIONS:
        raise ValueError(f'initial must be one of {INITIALIZATIONS}')
    burn_steps, measure_steps = round(burn/dt), round(measure/dt)
    if measure_steps < blocks:
        raise ValueError('At least one integration step per measurement block is required')
    if burn > 0 and burn_steps == 0:
        raise ValueError('Positive burn is smaller than half a timestep')
    branch = death_branch(k, omega)
    if initial == 'death' and (number != 2 or not branch['stable']):
        raise ValueError('death initialization requires N=2 and a strictly stable fixed point')
    fixed_x, fixed_y = branch.get('points', np.zeros((2, 2)))[0]
    block_edges = np.linspace(0, measure_steps, blocks+1, dtype=np.int64)
    block_steps = np.diff(block_edges)
    sequences = np.random.SeedSequence(seed).spawn(replicas)
    seeds = np.array([child.generate_state(1)[0] for child in sequences], dtype=np.uint32)
    if len(np.unique(seeds)) != replicas:
        raise ValueError('Replica seeds must be unique')
    arrays = _simulate(number, beta, k, omega, dt, burn_steps, block_steps, seeds,
                       INITIALIZATIONS.index(initial), fixed_x, fixed_y)
    individual, diagnostics, flags = arrays
    if np.any(flags):
        raise FloatingPointError(f'Nonfinite trajectory in {np.count_nonzero(flags)} replicas; refine timestep')
    if not all(np.isfinite(array).all() for array in arrays):
        raise FloatingPointError('Nonfinite numerical result')
    return dict(seeds=seeds, block_time_edges=block_edges*dt,
                block_frequency=individual.mean(axis=2), diagnostics=diagnostics)
