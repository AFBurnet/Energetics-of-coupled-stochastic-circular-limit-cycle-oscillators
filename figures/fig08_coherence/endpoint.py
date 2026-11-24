"""Coherent-cycle counts from the correlation phase at its 1/e crossing.

Each ensemble or bootstrap correlation curve has its own crossing and phase
interpolation.
"""

from __future__ import annotations

import numpy as np


_THRESHOLD = np.exp(-1.0)


def endpoint_metrics(correlation, time):
    """Return net correlation rotations accumulated before the first 1/e crossing.

    ``correlation`` has shape ``(batch, lag, readout)`` and ``time`` is a finite,
    strictly increasing lag array starting at zero. The crossing time is linearly
    interpolated in the correlation magnitude. The unwrapped correlation phase,
    relative to its zero-lag value, is interpolated with the same fraction.
    ``n_coherent_cycles`` is the absolute interpolated phase divided by ``2*pi``.

    Unresolved curves have NaN time, phase, and cycle count. No cycle-count lower
    bound is inferred, because the accumulated phase need not be monotonic.
    ``phase_resolution_warning`` flags an adjacent principal angular increment
    exceeding pi/2 through the crossing sample (all samples if unresolved). This
    diagnostic cannot exclude all phase aliasing. A crossing within the first
    sampling interval also sets ``below_sample_resolution``.
    """
    values = np.asarray(correlation, dtype=complex)
    if values.ndim != 3 or min(values.shape) < 1 or values.shape[1] < 2:
        raise ValueError("correlation must have shape (batch, lag >= 2, readout)")
    if not np.all(np.isfinite(values)):
        raise ValueError("correlation must be finite")
    lags = np.asarray(time)
    if np.iscomplexobj(lags):
        raise ValueError("time must be real")
    lags = np.asarray(lags, dtype=float)
    if (lags.ndim != 1 or len(lags) != values.shape[1]
            or not np.all(np.isfinite(lags)) or lags[0] != 0
            or np.any(np.diff(lags) <= 0)):
        raise ValueError("time must match the lag axis, start at 0, and increase strictly")

    magnitude = np.abs(values)
    crossings = magnitude <= _THRESHOLD
    resolved = np.any(crossings, axis=1)
    first = np.argmax(crossings, axis=1)
    safe_index = np.maximum(first, 1)[:, None]
    before = np.take_along_axis(magnitude, safe_index - 1, axis=1)[:, 0]
    after = np.take_along_axis(magnitude, safe_index, axis=1)[:, 0]
    fraction = np.zeros_like(before)
    np.divide(before - _THRESHOLD, before - after, out=fraction,
              where=before != after)
    fraction = np.clip(fraction, 0.0, 1.0)
    left_time = lags[np.maximum(first, 1) - 1]
    right_time = lags[np.maximum(first, 1)]
    tau = left_time + fraction * (right_time - left_time)

    phase = np.unwrap(np.angle(values), axis=1)
    phase -= phase[:, :1]
    left_phase = np.take_along_axis(phase, safe_index - 1, axis=1)[:, 0]
    right_phase = np.take_along_axis(phase, safe_index, axis=1)[:, 0]
    crossing_phase = left_phase + fraction * (right_phase - left_phase)
    tau = np.where(resolved, np.where(first == 0, 0.0, tau), np.nan)
    crossing_phase = np.where(
        resolved, np.where(first == 0, 0.0, crossing_phase), np.nan)

    angle_steps = np.abs(np.diff(phase, axis=1))
    last_step = np.where(resolved, first, len(lags) - 1)
    through_crossing = np.arange(1, len(lags))[None, :, None] <= last_step[:, None]
    warning = np.any(through_crossing & (angle_steps > np.pi / 2), axis=1)
    return {
        "tau_coherence": tau,
        "phase_at_coherence": crossing_phase,
        "n_coherent_cycles": np.abs(crossing_phase) / (2 * np.pi),
        "resolved": resolved,
        "below_sample_resolution": resolved & (first <= 1),
        "phase_resolution_warning": warning,
    }
