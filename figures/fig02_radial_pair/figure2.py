#!/usr/bin/env python3
"""Figure 2: radial coupling and numerical validation (m=n=C=r_c=1).

Use compute, validate, theory, or plot; run settings and seeds are in data/parameters.json.
"""

import argparse
import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit, prange
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, quad_vec
from scipy.optimize import brentq
from scipy.special import gamma
from scipy.stats import t as student_t

BETAS = (10/3, 7., 10.)

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


def strong_ratio(beta):
    """Pair strong-coupling EPR divided by 2*sigma0; beta=0 is its limit."""
    beta = float(beta)
    if beta == 0:
        return float(gamma(2.0/3.0)*gamma(5.0/6.0)/math.sqrt(math.pi))
    if beta < 0:
        raise ValueError("Require beta >= 0")
    strong_second = _free_integral(2*beta, 4)/_free_integral(2*beta, 2)
    return strong_second/free_moment(beta, 2)


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


def theory_moments(beta, k, order=128, rmax=None, tail_log=48.0):
    """Tensor Gauss-Legendre integration in s=(r1+r2)/2, delta=r1-r2.

    Returns the moments, EPR ratio and derivative, joint normalization and cutoff.
    normalization uses exp[-beta/3] times the unnormalized joint weight.
    The symmetric delta tail is cut only where its Gaussian factor <exp(-tail_log).
    rmax bounds each original radius separately.
    """
    beta, k = float(beta), float(k)
    if beta <= 0 or k < 0:
        raise ValueError("Require beta > 0 and k >= 0")
    rmax = radial_domain(beta, tail_log) if rmax is None else float(rmax)
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
    exponent = beta*(ss*ss-2*ss**3/3-1.0/3.0)-a[:, None]*delta**2
    joint = 2*(ss*ss-delta**2/4)*np.exp(exponent)*sw[:, None]*dw
    norm = np.sum(joint)
    moment2 = ss*ss+delta**2/4
    cov = np.sum(joint*moment2*delta**2)/norm
    cov -= np.sum(joint*moment2)*np.sum(joint*delta**2)/norm**2
    derivative = -beta*cov/(2*free_moment(beta, 2))
    return _moment_record(beta, norm, np.sum(joint*ss), np.sum(joint*moment2),
                          rmax=rmax, derivative=float(derivative))


def radial_marginal(beta, k, rgrid, order=128, rmax=None, tail_log=48.0):
    """Return marginal density wrt dr on rgrid, with no grid renormalization."""
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
    exponent = beta*(_potential(rr)+_potential(other)-1.0/3.0-k*(rr-other)**2/2)
    result[valid] = np.sum(rr*other*np.exp(exponent)*half*weights, axis=1)/record['normalization']
    return result.reshape(r.shape)


def reference_moments(beta, k, rmax=None):
    """Independent adaptive quadrature in original (r1,r2) coordinates.

    The inner integral splits around the narrow coupling ridge and covers
    the full radial interval, with absolute and relative tolerances of 1e-12.
    """
    beta, k = float(beta), float(k)
    if beta <= 0 or k < 0:
        raise ValueError("Require beta > 0 and k >= 0")
    rmax = radial_domain(beta) if rmax is None else float(rmax)
    width = 1/math.sqrt(beta*k) if k > 0 else rmax
    def outer(r1):
        points = sorted(set([0.0, rmax, 1.0]+[
            min(rmax, max(0.0, r1+j*width)) for j in (-8, -2, 0, 2, 8)]))
        def inner(r2):
            return r2*math.exp(beta*(_potential(r2)-1/6-k*(r1-r2)**2/2))
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
    return dict(k=k, ratio=ratio, boundary=bool(k == 0 or k == kmax),
                strong_ratio=strong_ratio(beta))


# Independent-pair Cartesian trajectories.

@njit(cache=True, parallel=True)
def _simulate_radial_replicas(beta, k, omega, dt, burn_steps, measure_steps,
                             seeds, edges, hist_stride):
    """One independent RNG stream and one pair of oscillators per row."""
    nrep = len(seeds)
    nbins = max(0, len(edges)-1)
    out = np.full((nrep, 3), np.nan)
    hist = np.zeros((nrep, nbins), dtype=np.int64)
    tails = np.zeros((nrep, 2), dtype=np.int64)
    flags = np.zeros(nrep, dtype=np.int64)
    noise = math.sqrt(2.0*dt/beta)
    for rep in prange(nrep):
        np.random.seed(seeds[rep])
        theta1 = 2.0*math.pi*np.random.random()
        theta2 = 2.0*math.pi*np.random.random()
        x1, y1 = math.cos(theta1), math.sin(theta1)
        x2, y2 = math.cos(theta2), math.sin(theta2)
        work = sum_mean = sum_second = 0.0
        for step in range(burn_steps + measure_steps):
            r1, r2 = math.hypot(x1, y1), math.hypot(x2, y2)
            if r1 == 0.0 or r2 == 0.0:
                flags[rep] = 1
                break
            a1 = 1.0-r1 + k*(r2-r1)/r1
            a2 = 1.0-r2 + k*(r1-r2)/r2
            fx1, fy1 = omega*y1+a1*x1, -omega*x1+a1*y1
            fx2, fy2 = omega*y2+a2*x2, -omega*x2+a2*y2
            nx1, ny1 = noise*np.random.normal(), noise*np.random.normal()
            nx2, ny2 = noise*np.random.normal(), noise*np.random.normal()
            xp1, yp1 = x1+fx1*dt+nx1, y1+fy1*dt+ny1
            xp2, yp2 = x2+fx2*dt+nx2, y2+fy2*dt+ny2
            rp1, rp2 = math.hypot(xp1, yp1), math.hypot(xp2, yp2)
            if rp1 == 0.0 or rp2 == 0.0:
                flags[rep] = 1
                break
            ap1 = 1.0-rp1+k*(rp2-rp1)/rp1
            ap2 = 1.0-rp2+k*(rp1-rp2)/rp2
            x1n = x1 + .5*(fx1+omega*yp1+ap1*xp1)*dt + nx1
            y1n = y1 + .5*(fy1-omega*xp1+ap1*yp1)*dt + ny1
            x2n = x2 + .5*(fx2+omega*yp2+ap2*xp2)*dt + nx2
            y2n = y2 + .5*(fy2-omega*xp2+ap2*yp2)*dt + ny2
            if not (math.isfinite(x1n) and math.isfinite(y1n)
                    and math.isfinite(x2n) and math.isfinite(y2n)):
                flags[rep] = 2
                break
            if step >= burn_steps:
                r1n, r2n = math.hypot(x1n, y1n), math.hypot(x2n, y2n)
                xm1, ym1 = 0.5*(x1+x1n), 0.5*(y1+y1n)
                xm2, ym2 = 0.5*(x2+x2n), 0.5*(y2+y2n)
                rm1, rm2 = math.hypot(xm1, ym1), math.hypot(xm2, ym2)
                if rm1 == 0.0 or rm2 == 0.0:
                    flags[rep] = 1
                    break
                am1 = 1.0-rm1 + k*(rm2-rm1)/rm1
                am2 = 1.0-rm2 + k*(rm1-rm2)/rm2
                dx1, dy1, dx2, dy2 = x1n-x1, y1n-y1, x2n-x2, y2n-y2
                angle_step = omega*(ym1*dx1 - xm1*dy1 + ym2*dx2 - xm2*dy2)
                work += (angle_step + am1*(xm1*dx1 + ym1*dy1)
                         + am2*(xm2*dx2 + ym2*dy2))
                # Pool the two oscillators within each independent replica;
                # trapezoidal time moments use every measurement step.
                sum_mean += 0.25*(r1+r2+r1n+r2n)
                sum_second += 0.25*(r1*r1+r2*r2+r1n*r1n+r2n*r2n)
                measured_step = step-burn_steps+1
                if nbins and measured_step % hist_stride == 0:
                    for radius in (r1n, r2n):
                        ibin = np.searchsorted(edges, radius, side="right")-1
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
    """Return per-pair simulation measurements and replica seeds.

    work: full-force Stratonovich EPR estimate (per unit measurement time).
    mean/second: first/second radial time moments pooled over two oscillators.
    histogram: bin counts, bin edges, total scheduled samples, and
      the fractions below and above the bin edges.

    Cartesian additive-noise Heun reuses each Gaussian increment in the
    predictor and corrector, updating both oscillators synchronously. EPR uses
    the full drift at the arithmetic midpoint of accepted states.
    Moments use trapezoidal endpoint values at every accepted step. Histograms
    use endpoints at an integer cadence in physical time. Neither treats the
    two coupled oscillators or correlated time samples as independent replicas.
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
    raw, counts, tails, flags = _simulate_radial_replicas(
        beta, k, omega, dt, burn_steps, measure_steps, seeds, edges, hist_stride)
    if not np.all(np.isfinite(raw)) and not np.any(flags):
        raise FloatingPointError("Nonfinite trajectory observable; refine timestep or inspect settings")
    if np.any(flags):
        bad = np.flatnonzero(flags)
        raise FloatingPointError(f"Invalid Cartesian trajectory in replicas {bad.tolist()}; "
                                 f"flags={flags[bad].tolist()} (1: undefined zero-radius force, "
                                 "2: nonfinite Heun state). Refine timestep; no clipping applied.")
    result = dict(work=raw[:, 0], mean=raw[:, 1], second=raw[:, 2], replica_seeds=seeds)
    if len(edges):
        observations = 2*(measure_steps//hist_stride)
        result.update(histogram_counts=counts, hist_edges=edges,
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

def publication_style(plt):
    """LaTeX typography and tick styling shared with the phase-coupling figures."""
    return plt.rc_context({'text.usetex': True, 'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'], 'font.size': 12,
        'axes.labelsize': 18, 'xtick.labelsize': 15, 'ytick.labelsize': 15,
        'legend.fontsize': 11, 'legend.framealpha': 1,
        'axes.spines.top': True, 'axes.spines.right': True, 'axes.linewidth': .8,
        'xtick.major.size': 6, 'ytick.major.size': 6,
        'xtick.major.width': 1.1, 'ytick.major.width': 1.1,
        'xtick.minor.size': 4, 'ytick.minor.size': 4,
        'axes.grid': False, 'savefig.bbox': 'tight',
        'savefig.transparent': False, 'savefig.facecolor': 'white'})

def plot_main(figure, theory, outdir, plt, scale):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    points = [dict(zip(figure['point_fields'], row)) for row in figure['points']]
    # Original Fig. 2 theory palette: lightsteelblue and two Reds_d entries.
    # Explicit hex values avoid adding seaborn as a plotting dependency.
    colors = ('#b0c4de', '#c03029', '#f78a70')
    legend_size = 24*scale
    middle_label_size = 28*scale
    moment_sample_k = (0., 2., 5., 10., 20., 40.)
    fig = plt.figure(figsize=(9.6*scale, 16.5*scale), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=(1.7, 1, 1.7))
    ax = fig.add_subplot(grid[0, :])
    density_ax = fig.add_subplot(grid[1, 0])
    mean_ax = fig.add_subplot(grid[1, 1])
    limit_ax = fig.add_subplot(grid[2, :])
    for panel in (ax, limit_ax):
        panel.set_box_aspect(.7)
    for panel in (density_ax, mean_ax):
        panel.set_box_aspect(1)
    # Place the inset in data coordinates, below 1 and above the lower curve.
    inset = ax.inset_axes((24.5, .982, 13., .016), transform=ax.transData, zorder=6)
    inset_bounds = [-2.2, .5]
    for i, (beta, color, label_offset) in enumerate(zip(BETAS, colors, (-7, 3, -3))):
        ax.plot(theory['k'], theory['ratio'][i], color=color, lw=1.8, zorder=4)
        label_k = 34.
        label_ratio = np.interp(label_k, theory['k'], theory['ratio'][i])
        beta_label = '10/3' if np.isclose(beta, 10/3) else f'{beta:g}'
        ax.annotate(rf'$\beta={beta_label}$', (label_k, label_ratio),
                    xytext=(0, label_offset), textcoords='offset points',
                    color='black', fontsize=24*scale, ha='center',
                    va='top' if label_offset < 0 else 'bottom')
        infinity = theory['limits'][i]
        ax.axhline(infinity, color=color, linestyle='--', lw=1)
        if beta == 7:
            inset.plot(theory['k'], 1e3*(theory['ratio'][i]-1), color=color, zorder=6)
        for row in points:
            if row['beta'] != beta:
                continue
            k = row['k']
            m, m_sem = row['radial_moment_epr'], row['radial_moment_epr_sem']
            w, w_sem = row['stratonovich_epr'], row['stratonovich_epr_sem']
            ax.errorbar(k, w, yerr=w_sem, fmt='x', ms=4, color=color, capsize=2,
                        lw=.8, zorder=2)
            ax.errorbar(k, m, yerr=m_sem, fmt='o', ms=4, color=color, mfc='white',
                        mew=1, capsize=2, lw=.9, zorder=3)
            if beta == 7 and k <= 1:
                inset.errorbar(k, 1e3*(m-1), yerr=1e3*m_sem, fmt='o', ms=3,
                               mfc='white', color=color, capsize=1.5, lw=.7, zorder=3)
                inset_bounds.extend((1e3*(m-m_sem-1), 1e3*(m+m_sem-1)))
            if beta == BETAS[0] and k in (0, 40):
                index = 0 if k == 0 else 1
                histmean, histsem = figure['histogram_mean'][index], figure['histogram_sem'][index]
                centers = figure['histogram_centers']
                density_ax.fill_between(centers, histmean-histsem, histmean+histsem,
                                        color='#dce8f4')
                density_ax.plot(centers, histmean, '.', color=color, ms=3)
    ax.axhline(1, color='.6', lw=.7)
    ax.set(xlabel=r'$k\;(\tau_r^{-1})$', ylabel=r'$\sigma\;(2\sigma_0)$', xlim=(-.4, 40.5))
    handles = [Line2D([], [], color='black', lw=1.8, label='theory'),
               Line2D([], [], marker='o', mfc='white', ls='', color='.2', label='radial moments'),
               Line2D([], [], marker='x', ls='', color='.2', label='Stratonovich')]
    for k, value, sem in zip(figure['inset_k'], figure['inset_stratonovich_epr'],
                             figure['inset_stratonovich_epr_sem']):
        inset.errorbar(k, 1e3*(value-1), yerr=1e3*sem, fmt='D', ms=4.5,
                       color=colors[1], mec='white', mew=.5,
                       capsize=2, lw=1, zorder=5)
        inset_bounds.extend((1e3*(value-sem-1), 1e3*(value+sem-1)))
    inset.legend(handles=[Line2D([], [], marker='D', ls='', color=colors[1],
                                 markeredgecolor='white', markeredgewidth=.5,
                                 label='Stratonovich')],
                 loc='lower center', fontsize=legend_size, frameon=False,
                 handlelength=.8, handletextpad=.25, borderpad=.1,
                 borderaxespad=.1)
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(.06, .40),
              ncol=1, fontsize=legend_size, frameon=False, handlelength=1.2)
    inset.axhline(0, color='.5', lw=.7)
    inset.set(xlim=(-.04, 1.04),
              ylabel=r'$\sigma-2\sigma_0\;(2\times10^{-3}\sigma_0)$')
    inset.set_xticks((0, .5, 1))
    inset.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=(1, 2, 5, 10)))
    inset.tick_params(labelsize=14*scale, length=5*scale, width=.9)
    inset.yaxis.label.set_size(15*scale)
    padding = .06*(max(inset_bounds)-min(inset_bounds))
    inset.set_ylim(min(inset_bounds)-padding, max(inset_bounds)+padding)
    for i, k in enumerate((0, 40)):
        density_ax.plot(theory['radius'], theory['marginal'][i], color='.15',
                        lw=1.6, ls=':' if k == 0 else '-', label=f'$k={k}$')
    density_ax.set(xlabel=r'$r\;(r_c)$', ylabel=r'$r p(r)$', xlim=(0, 3))
    density_ax.legend(frameon=False, fontsize=middle_label_size, loc='upper right',
                      handlelength=.6, handletextpad=.25, borderpad=.1,
                      borderaxespad=.15)
    variance_ax = mean_ax.twinx(); variance_ax.spines['right'].set_visible(True)
    variance_ax.set_box_aspect(1)
    mean_ax.plot(theory['moment_k'], theory['mean'], color='lightcoral', lw=1.8, zorder=4)
    variance_ax.plot(theory['moment_k'], theory['variance'], color='lightblue', lw=1.8, zorder=4)
    mean_ax.set(xlabel=r'$k\;(\tau_r^{-1})$', ylabel='')
    variance_ax.set_ylabel('')
    mean_ax.tick_params(axis='y', colors='#c03029')
    variance_ax.tick_params(axis='y', colors='#527d9a')
    for panel in (mean_ax, variance_ax):
        panel.spines['left'].set_color('#c03029')
        panel.spines['right'].set_color('#527d9a')
    label_k = 23.
    mean_ax.annotate(r'$\langle r\rangle_k$',
                     (label_k, np.interp(label_k, theory['moment_k'], theory['mean'])),
                     xytext=(0, -9), textcoords='offset points', color='#c03029',
                     fontsize=middle_label_size, ha='center', va='top')
    variance_ax.annotate(r'$\mathrm{Var}_k(r)$',
                         (label_k, np.interp(label_k, theory['moment_k'], theory['variance'])),
                         xytext=(0, 7), textcoords='offset points', color='#527d9a',
                         fontsize=middle_label_size, ha='center', va='bottom')
    for row in points:
        if row['beta'] == BETAS[0] and row['k'] in moment_sample_k:
            mean_ax.errorbar(row['k'], row['mean_radius'], yerr=row['mean_radius_sem'],
                             fmt='o', color='lightcoral', mfc='white',
                             ms=4.5, capsize=3, zorder=3)
            variance_ax.errorbar(row['k'], row['radial_variance'],
                                yerr=row['radial_variance_sem'], fmt='s',
                                color='lightblue', mfc='white', ms=4.5,
                                capsize=3, zorder=3)
    beta, limit = theory['strong_beta'], theory['strong_ratio']
    limit_ax.plot(beta, limit, color='.1', lw=1.6,
                  label=r'$\sigma(k\to\infty)$')
    limit_ax.fill_between(beta, limit, 1, where=limit < 1, color='lightblue', interpolate=True)
    limit_ax.fill_between(beta, limit, 1, where=limit > 1, color='lightcoral', interpolate=True)
    envelope = np.minimum(theory['envelope_ratio'], np.interp(theory['envelope_beta'], beta, limit))
    upper = np.minimum(1, np.interp(theory['envelope_beta'], beta, limit))
    limit_ax.fill_between(theory['envelope_beta'], envelope, upper,
                           where=envelope < upper, color='lightblue', interpolate=True)
    limit_ax.axhline(1, color='.5', ls='--', lw=.8)
    limit_ax.set(xlabel=r'$\beta$', ylabel=r'$\sigma\;(2\sigma_0)$', xlim=(0, 100))
    limit_ax.legend(frameon=False, loc='lower right', fontsize=legend_size,
                    labelspacing=.3)
    for panel in (ax, density_ax, mean_ax, variance_ax, limit_ax):
        panel.tick_params(labelsize=22*scale, length=7.5*scale, width=1.2)
        panel.xaxis.label.set_size(26*scale)
        panel.yaxis.label.set_size(26*scale)
    for panel, label, left in ((ax, '(a)', .025), (density_ax, '(b)', .04),
                               (mean_ax, '(c)', .10), (limit_ax, '(d)', .025)):
        panel.text(left, .99 if panel is ax else .975, label, transform=panel.transAxes,
                   fontsize=24*scale, color='black', ha='left', va='top', zorder=10)
    # Resolve the text margins, then bring the square middle panels inward
    # without changing their size or the proportions of the other panels.
    fig.canvas.draw()
    fig.set_layout_engine('none')
    density_ax.set_position(density_ax.get_position().translated(.01, 0))
    mean_position = mean_ax.get_position().translated(-.01, 0)
    mean_ax.set_position(mean_position)
    variance_ax.set_position(mean_position)
    save_figure(fig, outdir, 'figure2')
    plt.close(fig)

RAW_FIELDS = ('work', 'mean', 'second', 'replica_seeds')
HIST_FIELDS = ('histogram_counts', 'hist_edges', 'histogram_samples', 'histogram_tails')
POINT_FIELDS = ('beta', 'k', 'stratonovich_epr', 'stratonovich_epr_sem',
                'radial_moment_epr', 'radial_moment_epr_sem', 'mean_radius', 'mean_radius_sem',
                'radial_variance', 'radial_variance_sem')


def load_parameters(root):
    return json.loads((Path(root)/'data/parameters.json').read_text())


def run_configs(parameters):
    """Expand the common settings and overrides for each recorded run."""
    histogram = parameters['histogram']
    histogram_runs = set(histogram['run_ids'])
    histogram_settings = {key: value for key, value in histogram.items() if key != 'run_ids'}
    configs = {}
    for run in parameters['runs']:
        name = run['id']
        config = {**parameters['run_defaults'], **{k: v for k, v in run.items() if k != 'id'}}
        if name in histogram_runs:
            config['histogram'] = histogram_settings
        configs[name] = config
    return configs


def quadrature_settings(parameters):
    rule = parameters['quadrature']
    return dict(order=rule['nodes'], rmax=rule['radial_cutoff'])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def read_run(archive, configs, reference):
    """Recover per-pair observables; a role may use a stored run's prefix."""
    name, count = reference['id'], reference['count']
    config = configs[name]
    if not 2 <= count <= config['replicas']:
        raise ValueError('Invalid independent-pair count')
    data = {field: archive[name+'_'+field][:count] for field in RAW_FIELDS}
    data['moment'] = 2.0*config['beta']*config['omega']*config['omega']*data['second']
    if 'histogram' in config:
        for field in HIST_FIELDS:
            values = archive[name+'_'+field]
            data[field] = values[:count] if field in ('histogram_counts', 'histogram_tails') else values
        data['histogram'] = data['histogram_counts']/(data['histogram_samples']*
                                                    np.diff(data['hist_edges'])[None, :])
    return data


def find_run(parameters, configs, beta, k, count=None, **overrides):
    defaults = parameters['run_defaults']
    wanted = {key: defaults[key] for key in ('omega', 'dt', 'burn', 'measure', 'seed')}
    wanted.update(beta=float(beta), k=float(k), **overrides)
    count = parameters['analysis_replica_counts']['refinements'] if count is None else count
    for name, config in configs.items():
        if config['replicas'] >= count and all(config[key] == value for key, value in wanted.items()):
            return dict(id=name, count=count)
    raise ValueError(f'No recorded run for {wanted}, N={count}')


def pooled_runs(archive, configs, references, seen=None):
    """Pool matching settings and retain each recorded RNG stream once."""
    selected = [configs[ref['id']] for ref in references]
    if any(any(c[key] != selected[0][key] for key in ('beta', 'k', 'omega', 'dt', 'burn', 'measure'))
           for c in selected[1:]):
        raise ValueError('Cannot pool different physical or integration settings')
    batches = [read_run(archive, configs, reference) for reference in references]
    names = ('work', 'moment', 'replica_seeds')
    merged = {name: np.concatenate([batch[name] for batch in batches]) for name in names}
    seen = set() if seen is None else seen
    keep = []
    for index, seed in enumerate(merged['replica_seeds']):
        if int(seed) not in seen:
            keep.append(index)
            seen.add(int(seed))
    return {name: values[keep] for name, values in merged.items()}, selected[0]


def estimate_row(data, config, normalization, case=''):
    row = dict(case=case, **{key: config[key] for key in ('beta', 'k', 'dt', 'burn', 'measure')},
               replicas=len(data['work']), sigma_uncoupled=normalization)
    for name in ('work', 'moment'):
        mean, sem, half = interval(data[name]/normalization)
        row.update({name: float(mean), name+'_sem': float(sem), name+'_ci95': float(half)})
    mean, sem, half = interval((data['work']-data['moment'])/normalization)
    row.update(work_minus_moment=float(mean), work_minus_moment_sem=float(sem),
               work_minus_moment_ci95=float(half))
    return row


def welch_half_width(sem_a, count_a, sem_b, count_b):
    """Pointwise 95% interval half-width for the difference of independent means."""
    va, vb = sem_a**2, sem_b**2
    variance = va+vb
    if not variance:
        return 0.
    df = variance**2/(va**2/(count_a-1)+vb**2/(count_b-1))
    return float(student_t.ppf(.975, df)*math.sqrt(variance))


def depth_interval(minimum, zero, estimator):
    """Welch interval for independent simulated baseline minus minimum."""
    half = welch_half_width(minimum[estimator+'_sem'], minimum['replicas'],
                            zero[estimator+'_sem'], zero['replicas'])
    value = zero[estimator]-minimum[estimator]
    return dict(depth=value, ci95_half=half, ci95=[value-half, value+half])


def prepare_summaries(root):
    """Calculate saved plotting estimates from the recorded independent pairs."""
    parameters = load_parameters(root)
    defaults = parameters['run_defaults']
    configs = run_configs(parameters)
    normalizations = parameters['uncoupled_pair_epr']
    minimum_count = parameters['analysis_replica_counts']['minimum']
    kstar = parameters['reference_minimum']['k']
    numerical = quadrature_settings(parameters)
    def norm(beta):
        return normalizations[str(float(beta))]
    def reference(beta, k):
        return theory_moments(beta, k, **numerical)['ratio']
    points, histograms = [], []
    with np.load(root/'data/replicas.npz', allow_pickle=False) as archive:
        for name in parameters['figure_runs']['run_ids']:
            ref = dict(id=name, count=parameters['figure_runs']['replicas'])
            data, config = read_run(archive, configs, ref), configs[name]
            normalization = norm(config['beta'])
            mean, mean_sem, _ = interval(data['mean'])
            variance, variance_sem, _ = variance_interval(data['mean'], data['second'])
            work, work_sem, _ = interval(data['work']/normalization)
            moment, moment_sem, _ = interval(data['moment']/normalization)
            points.append((config['beta'], config['k'], work, work_sem, moment, moment_sem,
                           mean, mean_sem, variance, variance_sem))
            if 'histogram' in data:
                density, sem, _ = interval(data['histogram'])
                histograms.append((config['k'], (data['hist_edges'][:-1]+data['hist_edges'][1:])/2,
                                   density, sem))
        long_rows, used_streams = [], set()
        for case in ('minimum', 'uncoupled'):
            data, config = pooled_runs(archive, configs, parameters['long_runs'][case], used_streams)
            row = estimate_row(data, config, norm(7), case)
            row['reference'] = parameters['reference_minimum']['ratio'] if case == 'minimum' else 1.
            long_rows.append(row)
        cases = ((10/3, 0), (10/3, 40), (7., 0), (7., kstar), (7., 40), (10., 40))
        benchmarks, convergence = [], []
        for beta, k in cases:
            ref = find_run(parameters, configs, beta, k)
            base = read_run(archive, configs, ref)
            row = estimate_row(base, configs[ref['id']], norm(beta))
            row['reference_ratio'] = reference(beta, k)
            benchmarks.append(row)
            for changed, value in (('dt', defaults['dt']/2), ('burn', 2*defaults['burn']),
                                   ('measure', 2*defaults['measure'])):
                other = read_run(archive, configs, find_run(parameters, configs, beta, k, **{changed: value}))
                if np.intersect1d(base['replica_seeds'], other['replica_seeds']).size:
                    raise ValueError('Refinement comparisons require independent random streams')
                for estimator in ('work', 'moment'):
                    a, asem, _ = interval(base[estimator]/norm(beta))
                    b, bsem, _ = interval(other[estimator]/norm(beta))
                    half = welch_half_width(asem, len(base[estimator]), bsem, len(other[estimator]))
                    convergence.append(dict(beta=beta, k=k, changed=changed, value=value,
                        estimator=estimator, baseline=float(a), refined=float(b),
                        difference=float(b-a), difference_ci95=float(half)))
            for count in sorted({max(2, ref['count']//4), max(2, ref['count']//2), ref['count']}):
                for estimator in ('work', 'moment'):
                    value, _, half = interval(base[estimator][:count]/norm(beta))
                    convergence.append(dict(beta=beta, k=k, changed='replica_prefix', value=count,
                        estimator=estimator, baseline=float(value), refined=row['reference_ratio'],
                        difference=float(value-row['reference_ratio']), difference_ci95=float(half)))
        followup = []
        followup_refs = dict(production_minimum=find_run(parameters, configs, 7, kstar, count=minimum_count),
            production_strong_dt_0_0005=find_run(parameters, configs, 10/3, 40, dt=defaults['dt']/2),
            **parameters['followup_runs'])
        for case, ref in followup_refs.items():
            config = configs[ref['id']]
            row = estimate_row(read_run(archive, configs, ref), config, norm(config['beta']), case)
            row['reference'] = reference(config['beta'], config['k'])
            followup.append(row)
        pooled, config = pooled_runs(archive, configs,
            [followup_refs['production_minimum'], followup_refs['minimum_seed']])
        row = estimate_row(pooled, config, norm(7), 'minimum_pooled_seeds')
        row['reference'] = reference(7, kstar)
        followup.append(row)
        location_k = kstar*np.array([.35, .65, 1., 1.5, 2.])
        samples = [read_run(archive, configs, find_run(parameters, configs, 7, float(k),
                   count=minimum_count))['moment']/norm(7) for k in location_k]
        rng = np.random.default_rng(defaults['seed']+918)
        locations = [float(location_k[np.argmin([rng.choice(x, len(x), replace=True).mean()
                     for x in samples])]) for _ in range(500)]
        location = dict(k=location_k.tolist(), mean_ratio=[float(interval(x)[0]) for x in samples],
            ci95=[float(interval(x)[2]) for x in samples],
            grid_argmin_interval=np.quantile(locations, [.025, .975]).tolist(),
            grid_argmin_frequencies=[locations.count(float(k))/500 for k in location_k])
    histograms.sort(key=lambda record: record[0])
    save_npz(root/'data/figure.npz', point_fields=np.array(POINT_FIELDS), points=np.asarray(points),
        histogram_centers=histograms[0][1], histogram_mean=np.array([h[2] for h in histograms]),
        histogram_sem=np.array([h[3] for h in histograms]),
        inset_k=np.array([row['k'] for row in long_rows]),
        inset_stratonovich_epr=np.array([row['work'] for row in long_rows]),
        inset_stratonovich_epr_sem=np.array([row['work_sem'] for row in long_rows]))
    payload = dict(benchmarks=benchmarks, convergence=convergence, followup=followup,
        long_work=long_rows, depth={name: depth_interval(*long_rows, name) for name in ('work', 'moment')},
        reference_depth=parameters['reference_minimum']['depth'], location=location)
    write_json(root/'data/validation.json', validation_report(payload))


def validation_report(results):
    """Name the reported quantities without changing their calculations."""
    observable_names = {
        'work': 'stratonovich_epr',
        'moment': 'radial_moment_epr',
        'work_minus_moment': 'stratonovich_minus_radial_moment',
    }
    row_names = {
        'dt': 'timestep', 'burn': 'burn_in', 'measure': 'measurement_duration',
        'sigma_uncoupled': 'uncoupled_pair_epr',
        'reference': 'quadrature_epr', 'reference_ratio': 'quadrature_epr',
    }
    for old, new in observable_names.items():
        row_names.update({old: new, old+'_sem': new+'_sem',
                          old+'_ci95': new+'_ci95_half_width'})
    case_names = {
        'production_minimum': 'minimum_base_seed',
        'production_strong_dt_0_0005': 'strong_coupling_half_timestep',
        'minimum_seed': 'minimum_alternate_seed',
        'minimum_dt': 'minimum_half_timestep',
        'strong_dt': 'strong_coupling_quarter_timestep',
    }

    def estimate(row):
        named = {row_names.get(key, key): value for key, value in row.items()
                 if key != 'case'}
        if row['case']:
            named = {'case': case_names.get(row['case'], row['case']), **named}
        return named

    benchmarks = [estimate(row) for row in results['benchmarks']]
    additional_runs = []
    for row in results['followup']:
        named = estimate(row)
        if {key: value for key, value in named.items() if key != 'case'} not in benchmarks:
            additional_runs.append(named)

    refinements, replica_counts = [], []
    for row in results['convergence']:
        point = dict(beta=row['beta'], k=row['k'], estimator=observable_names[row['estimator']])
        if row['changed'] == 'replica_prefix':
            replica_counts.append(dict(point, replicas=row['value'], estimate=row['baseline'],
                quadrature_epr=row['refined'], estimate_minus_quadrature=row['difference'],
                difference_ci95_half_width=row['difference_ci95']))
        else:
            refinements.append(dict(point, changed=row_names[row['changed']], refined_setting=row['value'],
                baseline=row['baseline'], refined=row['refined'], refined_minus_baseline=row['difference'],
                difference_ci95_half_width=row['difference_ci95']))

    location = results['location']
    return dict(
        normalization='EPR estimates and differences are divided by the uncoupled pair EPR.',
        uncertainty='Pointwise 95% Student intervals across independent oscillator pairs. '
            'Refinement and minimum-depth differences use Welch intervals for independent runs. '
            'Stratonovich and radial-moment estimates share paths; their differences are paired. '
            'Replica-count estimates use nested subsets.',
        estimator_benchmarks=benchmarks,
        timestep_burn_duration_refinements=refinements,
        replica_count_checks=replica_counts,
        additional_runs=additional_runs,
        shallow_minimum=dict(
            long_run_estimates=[estimate(row) for row in results['long_work']],
            depth={observable_names[name]: dict(depth=row['depth'], ci95_half_width=row['ci95_half'],
                                                ci95_limits=row['ci95'])
                   for name, row in results['depth'].items()},
            quadrature_depth=results['reference_depth'],
            sampled_minimizer=dict(
                sampled_k=location['k'], radial_moment_epr=location['mean_ratio'],
                ci95_half_width=location['ci95'],
                sampled_minimizer_ci95_limits=location['grid_argmin_interval'],
                sampled_minimizer_frequencies=location['grid_argmin_frequencies'])))


def calculate_theory(settings):
    numerical = dict(order=settings['order'], rmax=settings['rmax'])
    ks = np.unique(np.r_[np.linspace(0, 1, 61), np.geomspace(1, 40, 100)])
    records = [[theory_moments(beta, k, **numerical) for k in ks] for beta in BETAS]
    moment_k = np.linspace(0, 40, 80)
    moments = [theory_moments(BETAS[0], k, **numerical) for k in moment_k]
    radius = np.linspace(0, 3.5, 701)
    strong_beta = np.unique(np.r_[0, np.geomspace(.005, 1, 45), np.linspace(1, 100, 250)])
    envelope_beta = np.unique(np.r_[np.linspace(.1, 4.5, 35),
        4.16, 4.1688, 4.17463644, 4.18, 4.2, 4.3, 4.4, np.linspace(4.5, 8.7, 85)])
    minima = [find_minimum(beta, kmax=4000, **numerical) for beta in envelope_beta]
    minimum = find_minimum(7, **numerical)
    return dict(beta=np.array(BETAS), k=ks,
        ratio=np.array([[r['ratio'] for r in row] for row in records]), moment_k=moment_k,
        mean=np.array([r['mean'] for r in moments]), variance=np.array([r['variance'] for r in moments]),
        radius=radius, marginal=np.array([radial_marginal(BETAS[0], k, radius, **numerical) for k in (0, 40)]),
        strong_beta=strong_beta, strong_ratio=np.array([strong_ratio(beta) for beta in strong_beta]),
        limits=np.array([strong_ratio(beta) for beta in BETAS]), envelope_beta=envelope_beta,
        envelope_ratio=np.array([min(m['ratio'], m['strong_ratio']) for m in minima]),
        envelope_k=np.array([m['k'] for m in minima]),
        minimum_k=np.asarray(minimum['k']), minimum_ratio=np.asarray(minimum['ratio']))


def verify_checksums(root, checksums):
    for name, digest in checksums.items():
        if hashlib.sha256((root/'data'/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Saved data checksum mismatch: {name}')


def compute(root, threads=None, cases=None, theory_dir=None):
    """Calculate selected runs from the recorded manuscript parameters."""
    from shutil import copyfile
    from numba import set_num_threads

    root = Path(root)
    bundled = Path(__file__).resolve().parent
    parameters = load_parameters(bundled)
    configs = run_configs(parameters)

    cases = set(cases or ())
    unknown = cases - configs.keys()
    if unknown:
        raise ValueError("Unknown run IDs: " + ", ".join(sorted(unknown)))
    runs = {name: config for name, config in configs.items() if not cases or name in cases}

    theory_files = {}
    if not cases:
        names = ('theory.npz', 'numerics.csv')
        if theory_dir is None:
            theory_dir = bundled
            theory_files = {name: parameters['checksums'][name] for name in names}
            verify_checksums(theory_dir, theory_files)
        else:
            theory_files = {name: hashlib.sha256((theory_dir/'data'/name).read_bytes()).hexdigest() for name in names}

    if threads is not None:
        set_num_threads(threads)
    arrays = {}
    for name, config in runs.items():
        histogram = config.get('histogram')
        options = {}
        if histogram:
            options = dict(hist_edges=np.linspace(0, histogram['max_radius'], histogram['bins']+1),
                           hist_interval=histogram['sample_interval'])
        values = simulate_replicas(**{key: config[key] for key in
            ('beta', 'k', 'omega', 'dt', 'burn', 'measure', 'replicas', 'seed')}, **options)
        fields = (*RAW_FIELDS, *(HIST_FIELDS if histogram else ()))
        arrays.update({name+'_'+field: values[field] for field in fields})
        print(f"Calculated {name}: beta={config['beta']:g}, k={config['k']:g}", flush=True)
    data = root / 'data'
    data.mkdir(parents=True)
    save_npz(data/'replicas.npz', **arrays)
    parameters['checksums'] = {
        'replicas.npz': hashlib.sha256((data/'replicas.npz').read_bytes()).hexdigest(),
        **theory_files,
    }
    for name in theory_files:
        copyfile(theory_dir/'data'/name, data/name)
    if cases:
        parameters['completed_runs'] = list(runs)
    write_json(data/'parameters.json', parameters)
    if not cases:
        prepare_summaries(root)
    else:
        print("Selected runs only; this is not a complete figure data set.")


def numerical_checks(settings):
    """Refinements and independent integrals for the quantities in Figure 2."""
    order, rmax = settings['order'], settings['rmax']
    minimum = find_minimum(7, order=order, rmax=rmax)
    rows = []
    def record(check, beta, k, comparison, quantity, baseline, refined, **extra):
        rows.append(dict(check=check, beta=beta, k=k, comparison=comparison,
                         quantity=quantity, baseline=float(baseline), refined=float(refined),
                         absolute_difference=float(abs(refined-baseline)), **extra))

    for beta in BETAS:
        for k in ((0, minimum['k'], 40) if beta == 7 else (0, 40)):
            domain = rmax or radial_domain(beta)
            base = theory_moments(beta, k, order=order, rmax=rmax)
            comparisons = [
                ('double_nodes', theory_moments(beta, k, order=2*order, rmax=domain)),
                ('double_nodes_and_extend_cutoff',
                 theory_moments(beta, k, order=2*order, rmax=1.25*domain))]
            if k == (minimum['k'] if beta == 7 else 40):
                comparisons.append(('vs_adaptive', reference_moments(beta, k, rmax=1.25*domain)))
            quantities = [('ratio', 'normalized_epr')]
            if beta == BETAS[0]:
                quantities += [('mean', 'mean_radius'), ('variance', 'radial_variance')]
            for comparison, refined in comparisons:
                for key, quantity in quantities:
                    record('quadrature', beta, k, comparison, quantity, base[key], refined[key])

    domain = rmax or radial_domain(BETAS[0])
    nodes, weights = _legendre(2*order)
    radius, weights = domain*(nodes+1)/2, domain*weights/2
    for k in (0, 40):
        p = radial_marginal(BETAS[0], k, radius, order=order, rmax=domain)
        fine = radial_marginal(BETAS[0], k, radius, order=2*order, rmax=1.25*domain)
        record('marginal', BETAS[0], k, 'vs_unit_mass', 'density_integral', 1, np.dot(weights, p))
        i = np.argmax(abs(p-fine))
        record('marginal', BETAS[0], k, 'double_nodes_and_extend_cutoff',
               'radial_density', p[i], fine[i], radius=float(radius[i]))

    for beta in (.01, *BETAS, 100):
        domain = 1.25*(rmax or radial_domain(beta))
        x, w = _legendre(3*order)
        r, w = domain*(x+1)/2, domain*w/2
        free = w*r*np.exp(beta*(_potential(r)-1/6))
        locked = w*r*r*np.exp(2*beta*(_potential(r)-1/6))
        fixed_ratio = (np.dot(locked, r*r)/locked.sum()) / (np.dot(free, r*r)/free.sum())
        record('strong_limit', beta, '', 'vs_fixed_nodes', 'normalized_epr',
               strong_ratio(beta), fixed_ratio)

    for beta in (.1, 4.16, 4.18, 4.2, 5, 7, 8.6):
        base = find_minimum(beta, kmax=4000, order=order, rmax=rmax)
        fine = find_minimum(beta, kmax=40000, order=2*order, rmax=1.25*(rmax or radial_domain(beta)))
        comparison = 'double_nodes_extend_cutoff_and_search'
        k_role = 'search_boundary' if base['boundary'] else 'finite_search_minimizer'
        record('minimum_envelope', beta, base['k'], comparison, 'normalized_epr',
               min(base['ratio'], base['strong_ratio']), min(fine['ratio'], fine['strong_ratio']),
               k_role=k_role)
        if not base['boundary'] and base['ratio'] < base['strong_ratio']:
            record('minimum_envelope', beta, base['k'], comparison, 'minimizing_coupling',
                   base['k'], fine['k'], k_role=k_role)
    return rows


def validate(root):
    parameters = load_parameters(root)
    if 'completed_runs' in parameters:
        raise ValueError('Validation requires the complete simulation plan')
    verify_checksums(root, parameters['checksums'])
    prepare_summaries(root)
    if 'numerics.csv' not in parameters['checksums']:
        rows = numerical_checks(quadrature_settings(parameters))
        write_csv(root/'data/numerics.csv', rows)
    print(f'Saved validation estimates and numerical checks in {root/"data"}', flush=True)


def save_figure(fig, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(outdir/f'{name}.{extension}', dpi=300)


def plot(root):
    """Render Figure 2 from saved summaries."""
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'energetics-lc-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with publication_style(plt):
        figure = load_npz(root/'data/figure.npz')
        theory = load_npz(root/'data/theory.npz')
        scale = .64
        sizes = ('font.size', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize',
                 'legend.fontsize', 'xtick.major.size', 'ytick.major.size',
                 'xtick.minor.size', 'ytick.minor.size')
        with plt.rc_context({key: scale*plt.rcParams[key] for key in sizes}):
            plot_main(figure, theory, root/'figures', plt, scale)
    print(f'Saved Figure 2 in {root / "figures"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="plot", choices=("plot", "compute", "validate", "theory"))
    parser.add_argument("--outdir", type=Path, help="Data directory; compute requires a separate, empty directory")
    parser.add_argument("--threads", type=int, help="Numba threads during simulation")
    parser.add_argument("--theory-dir", type=Path, help="Run directory from theory, used by a full compute run")
    parser.add_argument("--case", action="append", help="Compute only this recorded run ID (repeatable)")

    args = parser.parse_args(argv)
    root = (args.outdir or Path(__file__).resolve().parent).resolve()
    if args.case and args.command != "compute":
        parser.error("--case requires a computation command")
    if args.theory_dir is not None and (args.command != "compute" or args.case):
        parser.error("--theory-dir requires a full compute run")
    if args.command == "compute":
        if args.outdir is None or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
            parser.error("Choose a separate, empty --outdir for new calculations")
        compute(root, args.threads, args.case, args.theory_dir)
    elif args.command == "validate":
        validate(root)
    elif args.command == "theory":
        if args.outdir is None or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
            parser.error("Choose a separate, empty --outdir for new quadrature")
        settings = quadrature_settings(load_parameters(Path(__file__).resolve().parent))
        data = root/'data'
        data.mkdir(parents=True)
        save_npz(data/'theory.npz', **calculate_theory(settings))
        write_csv(data/'numerics.csv', numerical_checks(settings))
    else:
        plot(root)


if __name__ == '__main__':
    main()
