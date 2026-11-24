"""Cartesian Heun trajectories for coupling through x alone (C=r_c=1).

The interaction is k/2 sum_{i<j}(x_i-x_j)^2, with per-edge k, and the
clockwise drive is q_i=(omega*y_i,-omega*x_i). The interaction is not
rotation invariant: beta*q.F contains a coupling term, so the radial-moment
EPR identity and the Gibbs distribution for full Cartesian coupling do not
apply here. No stationary-distribution reference is assumed in this module.
"""

import hashlib
import json
import math
from functools import lru_cache

import mpmath as mp
import numpy as np
from numba import njit, prange

FIELDS = ('work', 'observable', 'second')
INITIALIZATIONS = ('random', 'synchronized', 'antiphase')


@njit(cache=True)
def _observables(x, y, beta, k, omega):
    """Return the instantaneous beta f_rot · A and the mean squared radius."""
    number = len(x)
    sx, observable, second = 0.0, 0.0, 0.0
    for i in range(number):
        sx += x[i]
    for i in range(number):
        r = math.hypot(x[i], y[i])
        coupling = k*(sx-number*x[i])
        second += r*r
        observable += beta*(omega*omega*r*r+omega*y[i]*coupling)
    return observable, second/number


def _log_I(beta, order):
    """Positive 2F2 expression for log int r^(order+1) exp[-beta U] dr."""
    b, n = mp.mpf(beta), mp.mpf(order)
    third, z = mp.mpf(1)/3, b/6
    terms = (
        8*mp.gamma((n+2)/3)*mp.hyper([(n+2)/6, (n+5)/6],
                                    [third, 2*third], z),
        4*mp.power(3, 2*third)*mp.power(b, third)*mp.gamma((n+4)/3)
        *mp.hyper([(n+4)/6, (n+7)/6], [2*third, 4*third], z),
        3*mp.power(3, third)*mp.power(b, 2*third)*mp.gamma((n+6)/3)
        *mp.hyper([(n+6)/6, (n+9)/6], [4*third, 5*third], z),
    )
    return (-mp.log(8)+(n-1)*mp.log(3)/3-(n+2)*mp.log(b)/3
            +mp.log(mp.fsum(terms)))


@lru_cache(maxsize=1024)
def free_second(beta, dps=35):
    """Uncoupled M2(beta), with Cartesian radial measure r dr."""
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')
    if isinstance(dps, bool) or int(dps) != dps or dps < 20:
        raise ValueError('dps must be an integer >=20')
    with mp.workdps(int(dps)):
        return float(mp.exp(_log_I(beta, 2)-_log_I(beta, 0)))


@njit(cache=True)
def _heun_step(x, y, k, omega, dt, nx, ny, px, py, fx, fy, xn, yn):
    """One step with externally supplied additive-noise increments."""
    number, sx = len(x), 0.0
    for i in range(number):
        sx += x[i]
    spx = 0.0
    for i in range(number):
        r = math.hypot(x[i], y[i])
        fx[i] = (1-r)*x[i]+omega*y[i]+k*(sx-number*x[i])
        fy[i] = (1-r)*y[i]-omega*x[i]
        px[i], py[i] = x[i]+dt*fx[i]+nx[i], y[i]+dt*fy[i]+ny[i]
        spx += px[i]
        if not (math.isfinite(px[i]) and math.isfinite(py[i])):
            return 1
    for i in range(number):
        r = math.hypot(px[i], py[i])
        gx = (1-r)*px[i]+omega*py[i]+k*(spx-number*px[i])
        gy = (1-r)*py[i]-omega*px[i]
        xn[i], yn[i] = x[i]+dt*(fx[i]+gx)/2+nx[i], y[i]+dt*(fy[i]+gy)/2+ny[i]
        if not (math.isfinite(xn[i]) and math.isfinite(yn[i])):
            return 2
    return 0


@njit(cache=True, parallel=True)
def _simulate(number, beta, k, omega, dt, burn_steps, measure_steps, seeds,
              initial_code, block_edges):
    nrep, nblocks = len(seeds), len(block_edges)-1
    raw = np.full((nrep, len(FIELDS)), np.nan)
    flags = np.zeros(nrep, dtype=np.int64)
    blocks = np.zeros((nrep, nblocks, 2))
    noise = math.sqrt(2*dt/beta)
    for rep in prange(nrep):
        np.random.seed(seeds[rep])
        x, y = np.empty(number), np.empty(number)
        nx, ny, px, py = np.empty(number), np.empty(number), np.empty(number), np.empty(number)
        fx, fy, xn, yn = np.empty(number), np.empty(number), np.empty(number), np.empty(number)
        common_angle = 2*math.pi*np.random.random()
        inversion = 1.0 if np.random.random() < .5 else -1.0
        # This near-y antiphase preparation is an initialization, not a claim
        # that a noisy antiphase state is an oscillation-death solution.
        x_to_y = omega/(k*number) if k*number > abs(omega) else 0.0
        for i in range(number):
            if initial_code == 0:
                angle = 2*math.pi*np.random.random()
                x[i], y[i] = math.cos(angle), math.sin(angle)
            elif initial_code == 1:
                x[i], y[i] = math.cos(common_angle), math.sin(common_angle)
            else:
                sign = inversion if i % 2 == 0 else -inversion
                y[i] = sign/math.sqrt(1+x_to_y*x_to_y)
                x[i] = x_to_y*y[i]
        work = observable = second = 0.0
        old_observable = old_second = 0.0
        block_index = 0
        for step in range(burn_steps+measure_steps):
            if step == burn_steps:
                old_observable, old_second = _observables(x, y, beta, k, omega)
            for i in range(number):
                nx[i], ny[i] = noise*np.random.normal(), noise*np.random.normal()
            flags[rep] = _heun_step(x, y, k, omega, dt, nx, ny,
                                     px, py, fx, fy, xn, yn)
            if flags[rep]:
                break
            if step >= burn_steps:
                measured = step-burn_steps+1
                while measured > block_edges[block_index+1]:
                    block_index += 1
                smx, step_work = 0.0, 0.0
                for i in range(number):
                    smx += (x[i]+xn[i])/2
                for i in range(number):
                    xm, ym = (x[i]+xn[i])/2, (y[i]+yn[i])/2
                    dx, dy = xn[i]-x[i], yn[i]-y[i]
                    r = math.hypot(xm, ym)
                    rotation = omega*(ym*dx-xm*dy)
                    conservative = (1-r)*(xm*dx+ym*dy)+k*(smx-number*xm)*dx
                    step_work += beta*(rotation+conservative)
                new_observable, new_second = _observables(xn, yn, beta, k, omega)
                step_observable = (old_observable+new_observable)/2
                work += step_work
                observable += step_observable
                second += (old_second+new_second)/2
                blocks[rep, block_index, 0] += step_work
                blocks[rep, block_index, 1] += step_observable
                old_observable, old_second = new_observable, new_second
            for i in range(number):
                x[i], y[i] = xn[i], yn[i]
        if flags[rep] == 0:
            duration = measure_steps*dt
            raw[rep, 0], raw[rep, 1] = work/duration, observable/measure_steps
            raw[rep, 2] = second/measure_steps
            for block in range(nblocks):
                count = block_edges[block+1]-block_edges[block]
                blocks[rep, block, 0] /= count*dt
                blocks[rep, block, 1] /= count
    return raw, flags, blocks


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f'{name} must be an integer >={minimum}')
    return int(value)


def _ceil_steps(duration, dt, name, zero=False):
    if not math.isfinite(duration) or duration < 0 or (duration == 0 and not zero):
        raise ValueError(f'{name} must be finite and positive'+(' or zero' if zero else ''))
    ratio = duration/dt
    nearest = round(ratio)
    return int(nearest if math.isclose(ratio, nearest, rel_tol=1e-13, abs_tol=1e-13)
               else math.ceil(ratio))


def simulate_replicas(number, beta, k, omega, dt, burn, measure, replicas, seed,
                      initial='random', blocks=10):
    """Return the two EPR estimates and the mean squared radius for each system.

    Block columns are the same two EPR estimators. Burn and measurement
    durations round up to complete steps. Seeds depend on the physical and
    integration settings and initialization.
    Extending an ensemble preserves all existing replica streams.
    """
    number = _positive_integer(number, 'number', 2)
    replicas, seed = _positive_integer(replicas, 'replicas'), _positive_integer(seed, 'seed', 0)
    blocks = _positive_integer(blocks, 'blocks')
    beta, k, omega, dt, burn, measure = map(
        float, (beta, k, omega, dt, burn, measure))
    if (not all(map(math.isfinite, (beta, k, omega, dt)))
            or beta <= 0 or k < 0 or dt <= 0):
        raise ValueError('Require finite beta>0, k>=0, dt>0, omega')
    if initial not in INITIALIZATIONS:
        raise ValueError(f'initial must be one of {INITIALIZATIONS}')
    burn_steps = _ceil_steps(burn, dt, 'burn', True)
    measure_steps = _ceil_steps(measure, dt, 'measure')
    if blocks > measure_steps:
        raise ValueError('blocks cannot exceed the number of measurement steps')
    block_edges = np.arange(blocks+1, dtype=np.int64)*measure_steps//blocks
    settings = dict(number=number, beta=beta, k=k, omega=omega, dt=dt,
                    burn=burn, measure=measure, initial=initial)
    digest = hashlib.blake2b(json.dumps(settings, sort_keys=True).encode(), digest_size=16).digest()
    entropy = [seed, *np.frombuffer(digest, dtype='<u4').tolist()]
    seeds = np.array([sequence.generate_state(1, dtype=np.uint32)[0]
                      for sequence in np.random.SeedSequence(entropy).spawn(replicas)], dtype=np.uint32)
    if len(np.unique(seeds)) != replicas:
        raise ValueError('Duplicate replica seeds; choose another master seed')
    raw, flags, block_values = _simulate(
        number, beta, k, omega, dt, burn_steps, measure_steps, seeds,
        INITIALIZATIONS.index(initial), block_edges)
    if (flags.any() or not np.isfinite(raw).all() or not np.isfinite(block_values).all()):
        bad = np.flatnonzero((flags != 0) | ~np.isfinite(raw).all(axis=1)
                             | ~np.isfinite(block_values).all(axis=(1, 2)))
        raise FloatingPointError(f'Invalid single-coordinate trajectories {bad.tolist()}; '
                                 f'flags={flags[bad].tolist()} (1 predictor, 2 corrector); '
                                 'refine timestep; no trajectories were discarded')
    result = dict(zip(FIELDS, raw.T))
    result.update(replica_seeds=seeds, block_values=block_values, block_time_edges=block_edges*dt)
    return result
