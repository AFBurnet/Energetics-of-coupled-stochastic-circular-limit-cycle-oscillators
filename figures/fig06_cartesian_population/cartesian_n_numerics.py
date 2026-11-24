"""N identical Cartesian-coupled oscillators, with C=r_c=1.

The conservative potential is sum_i(r_i**3/3-r_i**2/2) plus
k/2 sum_{i<j}|z_i-z_j|**2. Each oscillator rotates clockwise at omega.
There is no division of the per-edge coupling k by N.
"""

import hashlib
import json
import math
from functools import lru_cache

import mpmath as mp
import numpy as np
from numba import njit, prange
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, quad_vec
from scipy.optimize import minimize_scalar
from scipy.special import i0e

def _log_I(beta, order):
    """Log integral r**(order+1) exp[beta*(r**2/2-r**3/3)] dr.

    This positive 2F2 expression uses the caller's mp.workdps precision.
    """
    b, n = mp.mpf(beta), mp.mpf(order)
    if b <= 0 or n <= -2:
        raise ValueError('Require beta>0 and moment order>-2')
    third = mp.mpf(1)/3
    z = b/6
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


@lru_cache(maxsize=4096)
def _log_free_second(beta, dps):
    with mp.workdps(dps):
        return _log_I(beta, 2)-_log_I(beta, 0)


def _precision(dps):
    if isinstance(dps, bool) or int(dps) != dps or dps < 20:
        raise ValueError('dps must be an integer >=20')
    return int(dps)


def free_second(beta, dps=35):
    """Analytical free second radial moment, with Cartesian measure r dr."""
    beta, dps = float(beta), _precision(dps)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('beta must be finite and positive')
    with mp.workdps(dps):
        return float(mp.exp(_log_free_second(beta, dps)))


def strong_ratio(beta, number, dps=35):
    """sigma_inf/(N*sigma0)=M2(N*beta)/M2(beta), also for real N>=1.

    Cartesian locking leaves a single common vector with measure r dr;
    the radial-coupling r**N measure does not apply to this interaction.
    beta=0 returns the analytical high-temperature limiting ratio.
    """
    beta, number, dps = float(beta), float(number), _precision(dps)
    if (not math.isfinite(beta) or beta < 0 or not math.isfinite(number)
            or number < 1):
        raise ValueError('Require finite beta>=0 and N>=1')
    if beta == 0:
        return number**(-2/3)
    with mp.workdps(dps):
        value = float(mp.exp(_log_free_second(number*beta, dps)
                             -_log_free_second(beta, dps)))
    if not math.isfinite(value) or value <= 0:
        raise ArithmeticError('Invalid strong-coupling ratio')
    return value


def strong_contour(betas, numbers, dps=35):
    """Analytical normalized strong limit, indexed [beta,N]."""
    return np.array([[strong_ratio(beta, number, dps) for number in numbers]
                     for beta in betas])


@lru_cache(maxsize=16)
def _legendre(order):
    if isinstance(order, bool) or int(order) != order or order < 16:
        raise ValueError('Quadrature orders must be integers >=16')
    return leggauss(int(order))


def _stationary_settings(beta, k, number, rmax, width):
    beta, k, width = float(beta), float(k), float(width)
    if isinstance(number, bool) or int(number) != number or number < 2:
        raise ValueError('number must be an integer >=2')
    if (not all(map(math.isfinite, (beta, k, width)))
            or beta <= 0 or k < 0 or width <= 0):
        raise ValueError('Require finite beta>0, k>=0 and width>0')
    rmax = 1+(144/beta)**(1/3) if rmax is None else float(rmax)
    if not math.isfinite(rmax) or rmax <= 0:
        raise ValueError('rmax must be finite and positive')
    return beta, k, int(number), rmax, width


def stationary_ratio(beta, k, number, order=128, inner_order=96, rmax=None,
                     width=12.0):
    """Positive finite-N Cartesian Hubbard-Stratonovich quadrature.

    With a=beta*k*N, define
      Bp(s)=int_0^rmax r**(p+1) exp[beta*(h(r)-1/6)-a*(r-s)**2/2]
                                *i0e(a*s*r) dr,
    where h(r)=r**2/2-r**3/3. Then
      <r_i**2>=int s*B0(s)**(N-1)*B2(s) ds / int s*B0(s)**N ds.
    The identity follows by integrating a two-dimensional Gaussian auxiliary
    field; no coupling or Fourier/Bessel series is truncated. The inner
    Gaussian ridge is retained to width/sqrt(a), with independently adjustable
    radial cutoff and both Gauss-Legendre orders. All powers use log scaling.
    """
    beta, k, number, rmax, width = _stationary_settings(beta, k, number, rmax, width)
    nodes, weights = _legendre(order)
    inner_nodes, inner_weights = _legendre(inner_order)
    if k == 0:
        return 1.0
    a = beta*k*number
    half_width = width/math.sqrt(a)
    smax = rmax+half_width
    s, sw = (nodes+1)*smax/2, weights*smax/2
    lo, hi = np.maximum(0, s-half_width), np.minimum(rmax, s+half_width)
    radii = (lo+hi)[:, None]/2+(hi-lo)[:, None]*inner_nodes[None, :]/2
    rw = (hi-lo)[:, None]*inner_weights[None, :]/2
    h = radii*radii/2-radii**3/3-1/6
    base = (radii*np.exp(beta*h-a*(radii-s[:, None])**2/2)
            *i0e(a*s[:, None]*radii)*rw)
    b0, b2 = np.sum(base, axis=1), np.sum(base*radii*radii, axis=1)
    valid = b0 > 0
    if not np.any(valid):
        raise ArithmeticError('Empty stationary quadrature support; refine orders or domain')
    logs = number*np.log(b0[valid])
    joint = s[valid]*sw[valid]*np.exp(logs-np.max(logs))
    second = np.sum(joint*b2[valid]/b0[valid])/np.sum(joint)
    ratio = second/free_second(beta)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ArithmeticError('Invalid stationary quadrature moment')
    return float(ratio)


def _reference_free_second(beta, eps=1e-12, rmax=None):
    """Independent adaptive radial reference, without the 2F2 expression."""
    rmax = 1+(144/beta)**(1/3) if rmax is None else float(rmax)
    def term(radius, power):
        return radius**power*math.exp(beta*(radius*radius/2-radius**3/3-1/6))
    numerator = quad(term, 0, rmax, args=(3,), epsabs=eps, epsrel=eps)[0]
    denominator = quad(term, 0, rmax, args=(1,), epsabs=eps, epsrel=eps)[0]
    return numerator/denominator


def reference_ratio(beta, k, number, eps=1e-10, rmax=None, width=14.0):
    """Independent adaptive nested quadrature of the positive field identity.

    Both inner radial integrals cover [0,rmax], without a Gaussian ridge
    cutoff. The outer field extends width Gaussian deviations beyond rmax.
    Its normalization uses independently integrated uncoupled radial moments.
    """
    beta, k, number, rmax, width = _stationary_settings(beta, k, number, rmax, width)
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    if k == 0:
        return 1.0
    a = beta*k*number
    smax = rmax+width/math.sqrt(a)

    @lru_cache(maxsize=16384)
    def conditional(s):
        def inner(radius):
            base = (radius*math.exp(beta*(radius*radius/2-radius**3/3-1/6)
                                   -a*(radius-s)**2/2)*i0e(a*s*radius))
            return base*np.array([1.0, radius*radius])
        points = sorted(set([0.0, rmax]+[min(rmax, max(0, s+j/math.sqrt(a)))
                                         for j in (-6, -2, 0, 2, 6)]))
        value, _ = quad_vec(inner, 0, rmax, points=points[1:-1],
                            epsabs=eps*1e-3, epsrel=eps*.1, limit=200)
        if value[0] <= 0:
            return -math.inf, 0.0
        return math.log(value[0]), float(value[1]/value[0])
    optimum = minimize_scalar(lambda s: -conditional(float(s))[0],
                               bounds=(0, rmax), method='bounded')
    shift = max(conditional(0.0)[0], conditional(float(optimum.x))[0])
    def outer(s):
        log_norm, second = conditional(float(s))
        weight = s*math.exp(number*(log_norm-shift))
        return weight*np.array([1.0, second])
    value, _ = quad_vec(outer, 0, smax, points=[float(optimum.x), rmax],
                        epsabs=eps, epsrel=eps, limit=200)
    ratio = value[1]/value[0]/_reference_free_second(beta, eps*.01, rmax)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ArithmeticError('Invalid adaptive stationary quadrature moment')
    return float(ratio)


@njit(cache=True, parallel=True)
def _simulate(number, beta, k, omega, dt, burn_steps, measure_steps, seeds,
              trace_stride=0):
    """Additive-noise Heun with one independent complete N-system per row."""
    nrep = len(seeds)
    raw = np.full((nrep, 2), np.nan)
    flags = np.zeros(nrep, dtype=np.int64)
    trace = np.empty((nrep, measure_steps//trace_stride+1 if trace_stride else 0))
    noise = math.sqrt(2*dt/beta)
    for rep in prange(nrep):
        np.random.seed(seeds[rep])
        x, y = np.empty(number), np.empty(number)
        px, py = np.empty(number), np.empty(number)
        nx, ny = np.empty(number), np.empty(number)
        fx, fy = np.empty(number), np.empty(number)
        xn, yn = np.empty(number), np.empty(number)
        for i in range(number):
            angle = 2*math.pi*np.random.random()
            x[i], y[i] = math.cos(angle), math.sin(angle)
        work = second = 0.0
        for step in range(burn_steps+measure_steps):
            sx, sy = 0.0, 0.0
            for i in range(number):
                sx += x[i]
                sy += y[i]
            if trace_stride and step == burn_steps:
                trace[rep, 0] = x[0]
            spx, spy = 0.0, 0.0
            for i in range(number):
                radius = math.hypot(x[i], y[i])
                fx[i] = (1-radius)*x[i]+k*(sx-number*x[i])+omega*y[i]
                fy[i] = (1-radius)*y[i]+k*(sy-number*y[i])-omega*x[i]
                nx[i], ny[i] = noise*np.random.normal(), noise*np.random.normal()
                px[i], py[i] = x[i]+dt*fx[i]+nx[i], y[i]+dt*fy[i]+ny[i]
                spx += px[i]
                spy += py[i]
                if not (math.isfinite(px[i]) and math.isfinite(py[i])):
                    flags[rep] = 1
            if flags[rep]:
                break
            smx, smy = 0.0, 0.0
            for i in range(number):
                radius = math.hypot(px[i], py[i])
                gx = (1-radius)*px[i]+k*(spx-number*px[i])+omega*py[i]
                gy = (1-radius)*py[i]+k*(spy-number*py[i])-omega*px[i]
                xn[i] = x[i]+dt*(fx[i]+gx)/2+nx[i]
                yn[i] = y[i]+dt*(fy[i]+gy)/2+ny[i]
                smx += (x[i]+xn[i])/2
                smy += (y[i]+yn[i])/2
                if not (math.isfinite(xn[i]) and math.isfinite(yn[i])):
                    flags[rep] = 2
            if flags[rep]:
                break
            if step >= burn_steps:
                for i in range(number):
                    xm, ym = (x[i]+xn[i])/2, (y[i]+yn[i])/2
                    dx, dy = xn[i]-x[i], yn[i]-y[i]
                    rm = math.hypot(xm, ym)
                    rotation = omega*(ym*dx-xm*dy)
                    conservative = ((1-rm)*(xm*dx+ym*dy)
                                    +k*((smx-number*xm)*dx+(smy-number*ym)*dy))
                    work += rotation+conservative
                    radius, rn = math.hypot(x[i], y[i]), math.hypot(xn[i], yn[i])
                    second += (radius*radius+rn*rn)/2
                measured = step-burn_steps+1
                if trace_stride and measured % trace_stride == 0:
                    trace[rep, measured//trace_stride] = xn[0]
            for i in range(number):
                x[i], y[i] = xn[i], yn[i]
        if flags[rep] == 0:
            duration = measure_steps*dt
            raw[rep, 0] = beta*work/duration
            raw[rep, 1] = beta*omega**2*second/measure_steps
    return raw, flags, trace


def _steps(duration, dt, name, zero=False):
    if not math.isfinite(duration) or duration < 0 or (duration == 0 and not zero):
        raise ValueError(f'{name} must be finite and positive'+(' or zero' if zero else ''))
    count = int(round(duration/dt))
    if (count < (0 if zero else 1)
            or not math.isclose(count*dt, duration, rel_tol=1e-11, abs_tol=1e-12)):
        raise ValueError(f'{name} must be an integer number of timesteps')
    return count


def simulate_replicas(number, beta, k, omega, dt, burn, measure, replicas, seed,
                      trace_interval=None):
    """Independent-system estimates using Cartesian additive-noise Heun.

    Work uses the spatial-midpoint full-force Stratonovich integral. The
    moment estimate uses the sum of radial second moments across oscillators,
    with trapezoidal time weights over every measured step. Optional traces
    record the first oscillator's x coordinate for each replica.

    RNG seeds depend on the physical and integration settings. Changing trace
    cadence or extending replica count preserves existing trajectories.
    """
    if isinstance(number, bool) or int(number) != number or number < 2:
        raise ValueError('number must be an integer >=2')
    if isinstance(replicas, bool) or int(replicas) != replicas or replicas < 1:
        raise ValueError('replicas must be a positive integer')
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    number, replicas, seed = int(number), int(replicas), int(seed)
    beta, k, omega, dt, burn, measure = map(float, (beta, k, omega, dt, burn, measure))
    if not all(map(math.isfinite, (beta, k, omega, dt))) or beta <= 0 or k < 0 or dt <= 0:
        raise ValueError('Require finite beta>0, k>=0, dt>0 and omega')
    burn_steps = _steps(burn, dt, 'burn', True)
    measure_steps = _steps(measure, dt, 'measure')
    stride = 0 if trace_interval is None else _steps(float(trace_interval), dt, 'trace_interval')
    if stride and (stride > measure_steps or measure_steps % stride):
        raise ValueError('Trace interval must divide measurement duration')
    settings = dict(number=number, beta=beta, k=k, omega=omega, dt=dt,
                    burn=burn, measure=measure)
    digest = hashlib.blake2b(json.dumps(settings, sort_keys=True).encode(), digest_size=16).digest()
    entropy = [seed, *np.frombuffer(digest, dtype='<u4').tolist()]
    seeds = np.array([sequence.generate_state(1, dtype=np.uint32)[0]
                      for sequence in np.random.SeedSequence(entropy).spawn(replicas)],
                     dtype=np.uint32)
    if len(np.unique(seeds)) != replicas:
        raise ValueError('Duplicate replica seeds; choose another master seed')
    raw, flags, trace = _simulate(number, beta, k, omega, dt, burn_steps,
                                  measure_steps, seeds, stride)
    if flags.any() or not np.isfinite(raw).all() or (stride and not np.isfinite(trace).all()):
        bad = np.flatnonzero(flags)
        raise FloatingPointError(f'Invalid Cartesian trajectories {bad.tolist()}; '
                                 f'flags={flags[bad].tolist()} (1: predictor, 2: corrector); '
                                 'refine timestep; no trajectories were discarded')
    result = dict(work=raw[:, 0], moment=raw[:, 1], replica_seeds=seeds)
    if stride:
        result.update(trace_x=trace, trace_time=np.arange(trace.shape[1])*stride*dt)
    return result
