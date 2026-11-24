"""Seeded Cartesian simulations for coherence and stochastic-process EPR.

Units are C=r_c=1 and the Cartesian diffusion constant is T_eff. Positive
intrinsic frequencies rotate clockwise. Coupling is an unnormalised
all-to-all sum.
"""

import hashlib
import json
import math

import numpy as np
from numba import njit, prange


COUPLINGS = ("radial", "phase", "cartesian", "cartesian_x")


@njit(cache=True)
def _fields(state, k, omegas, code):
    n = len(omegas)
    radii = np.empty(n)
    f = np.empty((n, 2))
    for i in range(n):
        x, y = state[i, 0], state[i, 1]
        radii[i] = math.hypot(x, y)
        f[i, 0] = omegas[i]*y + (1-radii[i])*x
        f[i, 1] = -omegas[i]*x + (1-radii[i])*y
    if code == 0:
        rsum = radii.sum()
        for i in range(n):
            if radii[i] > 0:
                force = k*(rsum-n*radii[i])/radii[i]
                f[i, 0] += force*state[i, 0]
                f[i, 1] += force*state[i, 1]
    elif code == 1:
        for i in range(n):
            angular = 0.0
            if radii[i] > 0:
                for j in range(n):
                    if j != i and radii[j] > 0:
                        angular += ((state[i, 0]*state[j, 1]
                                     - state[i, 1]*state[j, 0])
                                    / (radii[i]*radii[j]))
            angular *= k
            fx, fy = -angular*state[i, 1], angular*state[i, 0]
            f[i, 0] += fx
            f[i, 1] += fy
    else:
        xsum, ysum = state[:, 0].sum(), state[:, 1].sum()
        for i in range(n):
            f[i, 0] += k*(xsum-n*state[i, 0])
            if code == 2:
                f[i, 1] += k*(ysum-n*state[i, 1])
    return f


def _nonnegative_k(k):
    k = float(k)
    if not np.isfinite(k) or k < 0:
        raise ValueError("k must be finite and nonnegative")
    return k


@njit(cache=True)
def _readout_radii(state):
    """Return the first oscillator and Cartesian-mean radii."""
    n = len(state)
    zx, zy = 0.0, 0.0
    for i in range(n):
        zx += state[i, 0]
        zy += state[i, 1]
    return math.hypot(state[0, 0], state[0, 1]), math.hypot(zx/n, zy/n)


@njit(cache=True)
def _save_signals(state, row, scratch):
    for j in range(len(state)):
        scratch[j] = complex(state[j, 0], state[j, 1])
    row[0] = scratch[0]
    row[1] = scratch.mean()


@njit(cache=True)
def _one_path(k, T_eff, omegas, code, dt, nb, ns, stride, seed, signals):
    np.random.seed(seed)
    n = len(omegas)
    state = np.empty((n, 2))
    for i in range(n):
        theta = 2*np.pi*np.random.random()
        state[i, 0], state[i, 1] = math.cos(theta), math.sin(theta)
    f = _fields(state, k, omegas, code)
    noise_scale = math.sqrt(2*T_eff*dt)
    for unused in range(nb):
        noise = noise_scale*np.random.randn(n, 2)
        predictor = state+dt*f+noise
        fp = _fields(predictor, k, omegas, code)
        state = state+0.5*dt*(f+fp)+noise
        f = _fields(state, k, omegas, code)

    radii = _readout_radii(state)
    scratch = np.empty(n, dtype=np.complex128)
    _save_signals(state, signals[0], scratch)
    amplitude_sum = np.zeros(2)
    full_heat = radial_moment = 0.
    for step in range(ns):
        noise = noise_scale*np.random.randn(n, 2)
        predictor = state+dt*f+noise
        fp = _fields(predictor, k, omegas, code)
        new_state = state+0.5*dt*(f+fp)+noise
        fn = _fields(new_state, k, omegas, code)
        fm = _fields(0.5*(state+new_state), k, omegas, code)
        for i in range(n):
            for d in range(2):
                dx = new_state[i, d]-state[i, d]
                full_heat += fm[i, d]*dx
                if code == 2:
                    radial_moment += 0.5*omegas[i]**2*(state[i, d]**2+new_state[i, d]**2)
        rn = _readout_radii(new_state)
        for j in range(2):
            amplitude_sum[j] += 0.5*(radii[j]**2+rn[j]**2)
        if (step+1) % stride == 0:
            _save_signals(new_state, signals[(step+1)//stride], scratch)
        state, f = new_state, fn
        radii = rn
    return full_heat/(T_eff*ns*dt), np.sqrt(amplitude_sum/ns), radial_moment/(T_eff*ns)


@njit(cache=True, parallel=True)
def _replicas(k, T_eff, omegas, code, dt, nb, ns, stride, seeds):
    count = len(seeds)
    signals = np.empty((count, ns//stride+1, 2), dtype=np.complex128)
    epr = np.empty(count)
    rms = np.empty((count, 2))
    radial_moment = np.empty(count)
    for rep in prange(count):
        epr[rep], rms[rep], radial_moment[rep] = _one_path(k, T_eff, omegas, code, dt, nb, ns,
                                      stride, seeds[rep], signals[rep])
    return signals, epr, rms, radial_moment


def simulate(k, T_eff, omegas, coupling, n_replicas, burn, duration,
             dt_max, sample_dt, seed, step_scale=0.05):
    """Return sampled signals and full-step estimates of EPR and RMS amplitude.

    The two readouts are the first oscillator and the Cartesian mean. The
    post-burn initial state is included in each sampled signal.
    The step divides sample_dt; duration is rounded up to a sampling interval
    and burn is rounded up to a step. Returned times describe the actual run.

    EPR evaluates the full Cartesian drift at the arithmetic midpoint of each
    accepted Heun step, dotted with its displacement, divided by T_eff*duration.
    RMS amplitudes and the full Cartesian radial-moment benchmark use
    endpoint trapezoid time averages over all integration steps. The seed
    stream depends on the physical and integration settings.
    """
    k, T_eff = _nonnegative_k(k), float(T_eff)
    code = COUPLINGS.index(coupling)
    omegas = np.asarray(omegas, dtype=np.float64)
    if omegas.ndim != 1 or not len(omegas) or not np.all(np.isfinite(omegas)):
        raise ValueError("omegas must be a nonempty finite one-dimensional array")
    omegas = np.ascontiguousarray(omegas)
    if not np.isfinite(T_eff) or T_eff <= 0:
        raise ValueError("T_eff must be finite and strictly positive")
    for name, value in (("n_replicas", n_replicas), ("seed", seed)):
        if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                or value < (1 if name == "n_replicas" else 0)):
            raise ValueError(f"{name} must be an integer >= {1 if name == 'n_replicas' else 0}")
    burn, duration = float(burn), float(duration)
    dt_max, sample_dt, step_scale = float(dt_max), float(sample_dt), float(step_scale)
    for name, value in (("duration", duration), ("dt_max", dt_max),
                        ("sample_dt", sample_dt), ("step_scale", step_scale)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if not np.isfinite(burn) or burn < 0:
        raise ValueError("burn must be finite and nonnegative")
    dt_limit = min(dt_max, step_scale/(1+len(omegas)*k+np.max(np.abs(omegas))))
    stride = max(1, int(np.ceil(sample_dt/dt_limit)))
    dt = sample_dt/stride
    nb = int(np.ceil(burn/dt))
    intervals = max(1, int(np.ceil(duration/sample_dt)))
    ns = intervals*stride
    # The fixed final seed component preserves the recorded random streams.
    settings = [COUPLINGS[code] if k > 0 else "uncoupled", float(k).hex(), T_eff.hex(),
                [float(w).hex() for w in omegas], burn.hex(), duration.hex(),
                dt_max.hex(), sample_dt.hex(), step_scale.hex(), (0.1).hex()]
    digest = hashlib.sha256(json.dumps(settings, separators=(",", ":")).encode()).digest()
    tag = np.frombuffer(digest, dtype="<u4").astype(np.uint32).tolist()
    seeds = np.random.SeedSequence([int(seed), *tag]).generate_state(int(n_replicas))
    signals, epr, rms, radial_moment = _replicas(k, T_eff, omegas, code if k > 0 else 2, dt, nb, ns,
                                 stride, seeds)
    if not all(np.isfinite(x).all() for x in (signals, epr, rms, radial_moment)):
        raise RuntimeError(f"Nonfinite trajectory at k={k:g}; reduce dt_max/step_scale")
    return dict(signals=signals, epr=epr, rms_amplitude=rms, radial_moment_epr=radial_moment, seeds=seeds,
                dt=dt, sample_dt=sample_dt, duration=ns*dt, burn=nb*dt)
