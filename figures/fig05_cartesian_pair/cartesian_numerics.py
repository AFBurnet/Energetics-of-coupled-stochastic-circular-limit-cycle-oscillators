"""Full Cartesian coupling of two circular oscillators (C=r_c=1).

Independent-pair Heun trajectories, exact stationary quadrature and diagnostics.
All radial probability densities use the Cartesian Jacobian explicitly.
"""
import hashlib
import json
import math
from functools import lru_cache
import numpy as np
from numba import njit, prange
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, quad_vec
from scipy.optimize import brentq
from scipy.special import i0e, i1e
from scipy.stats import t as student_t

def _potential(r):
    return r*r/2 - r*r*r/3


def radial_domain(beta, tail_log=48.0):
    """Conservative finite radius, with exponent below its peak by >tail_log."""
    if beta <= 0:
        raise ValueError("Finite-temperature calculations require beta > 0")
    return 1.0 + (3.0*tail_log/beta)**(1.0/3.0)


@lru_cache(maxsize=8192)
def _free_integral(beta, power):
    """Integral of r**power exp[beta*(h(r)-h(1))], scaled at small beta."""
    beta = float(beta)
    if beta <= 0 or power < 0:
        raise ValueError("Require beta > 0 and power >= 0")
    scale = max(1.0, beta**(-1.0/3.0))
    def integrand(u):
        r = scale*u
        return u**power * math.exp(beta*(_potential(r)-1.0/6.0))
    at_peak = 1.0/scale
    value = sum(quad(integrand, a, b, epsabs=2e-12, epsrel=2e-12)[0]
                for a, b in ((0.0, at_peak), (at_peak, np.inf)))
    return scale**(power+1)*value


@lru_cache(maxsize=8192)
def free_moment(beta, power):
    """E[r**power] for the uncoupled radial density r exp[beta*h(r)]."""
    return _free_integral(beta, power+1)/_free_integral(beta, 1)


def sigma0(beta, omega=0.2):
    """One oscillator's free EPR; the pair's normalization is 2*sigma0."""
    return float(beta)*omega**2*free_moment(float(beta), 2)


@lru_cache(maxsize=16)
def _legendre(order):
    if int(order) < 16:
        raise ValueError("Quadrature order must be >=16")
    return leggauss(int(order))


def _moment_record(inverse_temperature, norm, first, second, **settings):
    mean = first/norm
    moment2 = second/norm
    return dict(mean=float(mean), second=float(moment2),
                variance=float(moment2-mean*mean),
                ratio=float(moment2/free_moment(float(inverse_temperature), 2)),
                normalization=float(norm), **settings)


def strong_ratio(beta):
    """Normalized pair strong limit; its radial measure is r exp[2 beta h(r)]."""
    beta = float(beta)
    if beta < 0:
        raise ValueError("Require beta >= 0")
    if beta == 0:
        return 2.0**(-2.0/3.0)
    return free_moment(2*beta, 2)/free_moment(beta, 2)


def theory_moments(beta, k, order=128, rmax=None, tail_log=48.0):
    """Stable exact two-radius quadrature after analytical angular integration.

    s=(r1+r2)/2, delta=r1-r2 resolve the narrow coupling ridge. i0e removes
    the exponentially growing part of I_0. The domain bounds BOTH radii;
    an additional Gaussian difference tail is omitted below exp(-tail_log).
    The k derivative is the covariance with the exact logarithmic score.
    """
    beta, k = float(beta), float(k)
    if beta <= 0 or k < 0:
        raise ValueError("Require beta > 0 and k >= 0")
    rmax = radial_domain(beta, tail_log) if rmax is None else float(rmax)
    if not math.isfinite(rmax) or rmax <= 0 or tail_log <= 0:
        raise ValueError("Require finite positive rmax and positive tail_log")
    nodes, weights = _legendre(order)
    s = rmax*(nodes+1)/2
    sw = rmax*weights/2
    upper = 2*np.minimum(s, rmax-s)
    a = beta*((k+s)/2-0.25)
    positive = a > 0
    upper[positive] = np.minimum(upper[positive], np.sqrt(tail_log/a[positive]))
    delta = upper[:, None]*(nodes[None, :]+1)/2
    dw = upper[:, None]*weights[None, :]/2
    ss = s[:, None]
    product = ss*ss-delta**2/4
    z = beta*k*product
    bessel = i0e(z)
    exponent = beta*(ss*ss-2*ss**3/3-1/3)-a[:, None]*delta**2
    joint = 2*product*np.exp(exponent)*bessel*sw[:, None]*dw
    norm = np.sum(joint)
    second_observable = ss*ss+delta**2/4
    second = np.sum(joint*second_observable)/norm
    score = -beta*delta**2/2+beta*product*(i1e(z)/bessel-1)
    derivative = (np.sum(joint*second_observable*score)/norm
                  - second*np.sum(joint*score)/norm)/free_moment(beta, 2)
    return _moment_record(beta, norm, np.sum(joint*ss), np.sum(joint*second_observable),
                          beta=beta, k=k, order=int(order), rmax=rmax,
                          tail_log=float(tail_log), derivative=float(derivative))


def radial_marginal(beta, k, rgrid, order=128, rmax=None, tail_log=48.0):
    """Density with respect to dr, without renormalizing the displayed grid."""
    beta, k = float(beta), float(k)
    record = theory_moments(beta, k, order, rmax, tail_log)
    rmax = record['rmax']
    r = np.asarray(rgrid, dtype=float)
    flat = r.ravel()
    nodes, weights = _legendre(order)
    width = math.sqrt(2*tail_log/(beta*k)) if k > 0 else rmax
    lo = np.maximum(0.0, flat-width)
    hi = np.minimum(rmax, flat+width)
    valid = (flat >= 0) & (flat <= rmax)
    result = np.zeros_like(flat)
    rr = flat[valid, None]
    half = (hi[valid]-lo[valid])[:, None]/2
    other = (hi[valid]+lo[valid])[:, None]/2+half*nodes
    exponent = beta*(_potential(rr)+_potential(other)-1/3-k*(rr-other)**2/2)
    result[valid] = np.sum(rr*other*np.exp(exponent)*i0e(beta*k*rr*other)
                           *half*weights, axis=1)/record['normalization']
    return result.reshape(r.shape)


def reference_moments(beta, k, rmax=None):
    """Independent adaptive integration in the original two radii, without difference cutoff."""
    beta, k = float(beta), float(k)
    if beta <= 0 or k < 0:
        raise ValueError("Require beta > 0 and k >= 0")
    rmax = radial_domain(beta) if rmax is None else float(rmax)
    width = 1/math.sqrt(beta*k) if k > 0 else rmax
    def outer(r1):
        points = sorted(set([0.0, rmax, 1.0]+[
            min(rmax, max(0.0, r1+j*width)) for j in (-8, -2, 0, 2, 8)]))
        def inner(r2):
            return r2*math.exp(beta*(_potential(r2)-1/6-k*(r1-r2)**2/2))*i0e(beta*k*r1*r2)
        val, _ = quad(inner, 0, rmax, points=[p for p in points if 0 < p < rmax],
                        epsabs=1e-12, epsrel=1e-12, limit=200)
        base = r1*math.exp(beta*(_potential(r1)-1/6))*val
        return base*np.array([1.0, r1, r1*r1])
    vals, _ = quad_vec(outer, 0, rmax, epsabs=1e-12, epsrel=1e-12,
                            points=[1.0], limit=200)
    return _moment_record(beta, *vals)


def find_minimum(beta, kmax=40.0, order=128, rmax=None):
    """Search finite k, retaining boundaries; compare against strong_ratio separately."""
    def evaluate(k):
        return theory_moments(beta, k, order, rmax)
    grid = np.unique(np.r_[0.0, np.geomspace(1e-5, kmax, 65)])
    records = [evaluate(k) for k in grid]
    values = np.array([record['ratio'] for record in records])
    candidates = [(float(grid[i]), float(values[i])) for i in (0, len(grid)-1)]
    for i in range(len(grid)-1):
        if records[i]['derivative'] < 0 < records[i+1]['derivative']:
            location = brentq(lambda x: evaluate(x)['derivative'],
                              grid[i], grid[i+1], xtol=1e-10)
            candidates.append((float(location), evaluate(location)['ratio']))
    k, ratio = min(candidates, key=lambda item: item[1])
    return dict(beta=float(beta), k=k, ratio=ratio, depth=1-ratio,
                boundary=bool(k == 0 or k == kmax), kmax=float(kmax),
                order=int(order), strong_ratio=strong_ratio(beta))


@njit(cache=True, parallel=True)
def _simulate_cartesian_replicas(beta, k, omega, dt, burn_steps, measure_steps,
                                seeds, edges, hist_stride):
    """Cartesian additive-noise Heun; each row has its own pair and RNG stream."""
    nrep = len(seeds)
    nbins = max(0, len(edges)-1)
    out = np.full((nrep, 3), np.nan)
    hist = np.zeros((nrep, nbins), dtype=np.int64)
    tails = np.zeros((nrep, 2), dtype=np.int64)
    flags = np.zeros(nrep, dtype=np.int64)
    noise = math.sqrt(2*dt/beta)
    for rep in prange(nrep):
        np.random.seed(seeds[rep])
        theta1 = 2*math.pi*np.random.random()
        theta2 = 2*math.pi*np.random.random()
        x1, y1 = math.cos(theta1), math.sin(theta1)
        x2, y2 = math.cos(theta2), math.sin(theta2)
        work = sum_mean = sum_second = 0.0
        for step in range(burn_steps+measure_steps):
            r1, r2 = math.hypot(x1, y1), math.hypot(x2, y2)
            f1x = omega*y1+(1-r1)*x1+k*(x2-x1)
            f1y = -omega*x1+(1-r1)*y1+k*(y2-y1)
            f2x = omega*y2+(1-r2)*x2+k*(x1-x2)
            f2y = -omega*x2+(1-r2)*y2+k*(y1-y2)
            n1x, n1y = noise*np.random.normal(), noise*np.random.normal()
            n2x, n2y = noise*np.random.normal(), noise*np.random.normal()
            p1x, p1y = x1+dt*f1x+n1x, y1+dt*f1y+n1y
            p2x, p2y = x2+dt*f2x+n2x, y2+dt*f2y+n2y
            rp1, rp2 = math.hypot(p1x, p1y), math.hypot(p2x, p2y)
            g1x = omega*p1y+(1-rp1)*p1x+k*(p2x-p1x)
            g1y = -omega*p1x+(1-rp1)*p1y+k*(p2y-p1y)
            g2x = omega*p2y+(1-rp2)*p2x+k*(p1x-p2x)
            g2y = -omega*p2x+(1-rp2)*p2y+k*(p1y-p2y)
            x1n, y1n = x1+dt*(f1x+g1x)/2+n1x, y1+dt*(f1y+g1y)/2+n1y
            x2n, y2n = x2+dt*(f2x+g2x)/2+n2x, y2+dt*(f2y+g2y)/2+n2y
            if not (math.isfinite(x1n) and math.isfinite(y1n)
                    and math.isfinite(x2n) and math.isfinite(y2n)):
                flags[rep] = 2
                break
            if step >= burn_steps:
                r1n, r2n = math.hypot(x1n, y1n), math.hypot(x2n, y2n)
                xm1, ym1 = (x1+x1n)/2, (y1+y1n)/2
                xm2, ym2 = (x2+x2n)/2, (y2+y2n)/2
                rm1, rm2 = math.hypot(xm1, ym1), math.hypot(xm2, ym2)
                dx1, dy1, dx2, dy2 = x1n-x1, y1n-y1, x2n-x2, y2n-y2
                angle_step = omega*(ym1*dx1-xm1*dy1+ym2*dx2-xm2*dy2)
                conservative = ((1-rm1)*(xm1*dx1+ym1*dy1)
                    +(1-rm2)*(xm2*dx2+ym2*dy2)
                    +k*((xm2-xm1)*dx1+(ym2-ym1)*dy1
                         +(xm1-xm2)*dx2+(ym1-ym2)*dy2))
                work += angle_step+conservative
                sum_mean += (r1+r2+r1n+r2n)/4
                sum_second += (r1*r1+r2*r2+r1n*r1n+r2n*r2n)/4
                measured_step = step-burn_steps+1
                if nbins and measured_step % hist_stride == 0:
                    for radius in (r1n, r2n):
                        ibin = np.searchsorted(edges, radius, side='right')-1
                        if radius == edges[-1]:
                            ibin = nbins-1
                        if ibin < 0:
                            tails[rep, 0] += 1
                        elif ibin >= nbins:
                            tails[rep, 1] += 1
                        else:
                            hist[rep, ibin] += 1
            x1, y1, x2, y2 = x1n, y1n, x2n, y2n
        if flags[rep] == 0:
            duration = measure_steps*dt
            out[rep, 0] = beta*work/duration
            out[rep, 1] = sum_mean/measure_steps
            out[rep, 2] = sum_second/measure_steps
    return out, hist, tails, flags


def _integer_steps(duration, dt, name, allow_zero=False):
    if not math.isfinite(duration) or duration < 0 or (not allow_zero and duration == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    steps = int(round(duration/dt))
    if (steps < (0 if allow_zero else 1)
            or not math.isclose(steps*dt, duration, rel_tol=1e-11, abs_tol=1e-12)):
        raise ValueError(f"{name}={duration} must be an integer number of dt={dt} steps")
    return steps


def simulate_replicas(beta, k, omega, dt, burn, measure, replicas, seed,
                      hist_edges=None, hist_interval=0.05):
    """Return midpoint EPR, radial moments and RNG seeds for independent pairs.

    Radial moments pool both oscillators using trapezoidal endpoint values at
    each measured Heun step. If requested, histogram counts use accepted states
    sampled at the given cadence, and the sample total includes radii outside
    the bins. The two tail fractions give the mass below and above the bins.

    Seeds depend on physical/integration settings, not replicas or histogram
    settings; extending replica count preserves all existing trajectories.
    """
    beta, k, omega, dt = map(float, (beta, k, omega, dt))
    burn, measure = float(burn), float(measure)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if not math.isfinite(k) or k < 0 or not math.isfinite(omega):
        raise ValueError("k must be finite/nonnegative and omega finite")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(replicas, bool) or int(replicas) != replicas or replicas < 1:
        raise ValueError("replicas must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    replicas, seed = int(replicas), int(seed)
    burn_steps = _integer_steps(burn, dt, "burn", allow_zero=True)
    measure_steps = _integer_steps(measure, dt, "measure")
    edges = np.array([], dtype=np.float64)
    hist_stride = 1
    if hist_edges is not None:
        edges = np.asarray(hist_edges, dtype=np.float64)
        if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
                or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
            raise ValueError("hist_edges must be increasing finite nonnegative bin edges")
        hist_stride = _integer_steps(float(hist_interval), dt, "hist_interval")
        if hist_stride > measure_steps:
            raise ValueError("hist_interval must not exceed measure")
    settings = {"beta": beta, "k": k, "omega": omega, "dt": dt,
                "burn": burn, "measure": measure}
    digest = hashlib.blake2b(json.dumps(settings, sort_keys=True).encode(), digest_size=16).digest()
    entropy = [seed, *np.frombuffer(digest, dtype="<u4").astype(np.uint32).tolist()]
    sequences = np.random.SeedSequence(entropy).spawn(replicas)
    seeds = np.asarray([s.generate_state(1, dtype=np.uint32)[0] for s in sequences], dtype=np.uint32)
    if np.unique(seeds).size != replicas:
        raise ValueError('Duplicate 32-bit replica RNG streams; choose another master seed')
    raw, counts, tails, flags = _simulate_cartesian_replicas(
        beta, k, omega, dt, burn_steps, measure_steps, seeds, edges, hist_stride)
    if not np.all(np.isfinite(raw)) and not np.any(flags):
        raise FloatingPointError("Nonfinite trajectory observable; refine timestep or inspect settings")
    if np.any(flags):
        bad = np.flatnonzero(flags)
        raise FloatingPointError(f"Invalid Cartesian trajectory in replicas {bad.tolist()}; "
                                 f"flags={flags[bad].tolist()} (2: nonfinite Heun state). Refine timestep; no clipping applied.")
    result = dict(work=raw[:, 0], mean=raw[:, 1], second=raw[:, 2], replica_seeds=seeds)
    if len(edges):
        observations = 2*(measure_steps//hist_stride)
        result.update(histogram_counts=counts,
                      histogram_tails=tails/observations,
                      histogram_samples=np.asarray(observations))
    return result


def interval(values):
    """Pointwise 95% Student CI over independent pairs, never time samples."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError('At least two independent replicas are required for uncertainty')
    mean = np.mean(values, axis=0)
    sem = np.std(values, axis=0, ddof=1)/np.sqrt(n)
    half = student_t.ppf(0.975, n-1)*sem
    return mean, sem, half


def variance_interval(first, second):
    """Bias-corrected replica jackknife for Var(r), preserving moment covariance."""
    first, second = np.asarray(first), np.asarray(second)
    n = len(first)
    full = second.mean()-first.mean()**2
    leave_one_out = (second.sum()-second)/(n-1)-((first.sum()-first)/(n-1))**2
    pseudovalues = n*full-(n-1)*leave_one_out
    return interval(pseudovalues)


