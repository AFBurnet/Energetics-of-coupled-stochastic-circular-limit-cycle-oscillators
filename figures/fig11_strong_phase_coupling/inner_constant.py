#!/usr/bin/env python3
"""Deterministic reproduction of the universal inner constant.

For R = r*sqrt(k/T_eff), the normalized inner density satisfies

    p_RR + p_R/R + p_phiphi/R**2 + 2*d_phi(sin(phi)*p) = 0.

Its real cosine moments c_l = integral cos(l*phi)*p dphi obey
    c_l'' + c_l'/R - l**2*c_l/R**2 + l*(c_(l-1)-c_(l+1)) = 0,
with c_0 = 1. The quantity reproduced here, independently of any EPR data, is
    C = integral_0^1 c_1/R dR + integral_1^infinity (c_1-1)/R dR.

Centered second differences in s=log(R), regular-origin Robin conditions,
and a sparse linear solve determine c_1,...,c_M; c_(M+1)=0 closes the modes.
At the outer endpoint, normalized von Mises moments impose the leading
locked solution. Domain and mode refinement check this approximate boundary.

Near zero, c_1=alpha*R-R**2/3+..., so the omitted origin integral is
c_1(Rmin)+Rmin**2/6+O(Rmin**3). At infinity the moment recurrence gives
c_1=1-1/(4*R**2)-9/(32*R**4)-87/(64*R**6)+O(R**-8).
Integrating these terms supplies the omitted tail. Domain and mode
refinements assess the numerical sensitivity.

Use --output to save the radial-grid, domain, Fourier-mode and origin comparisons.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
from scipy.special import ive


TAIL_COEFFICIENTS = (-1/4, -9/32, -87/64)


def tail_correction(radius):
    """Integral of the fixed-mode outer series after subtracting unity."""
    return sum(a/(2*n*radius**(2*n))
               for n, a in enumerate(TAIL_COEFFICIENTS, 1))


def solve_inner(modes=144, intervals=4096, rmax=24., rmin=1.e-6):
    """Solve the inner moment hierarchy and return C with its grid settings."""
    if (not isinstance(modes, int) or modes < 2
            or not isinstance(intervals, int) or intervals < 8
            or intervals % 2 or not 0 < rmin < 1 < rmax
            or not math.isfinite(rmax)):
        raise ValueError('Require M >= 2, even intervals >= 8, and 0 < Rmin < 1 < Rmax.')
    s = np.linspace(np.log(rmin), np.log(rmax), intervals+1)
    radius, h = np.exp(s), s[1]-s[0]
    n = len(s)
    ell = np.arange(1, modes+1, dtype=float)
    radial = diags((np.full(n-1, 1/h**2), np.full(n, -2/h**2),
                    np.full(n-1, 1/h**2)), (-1, 0, 1), format='lil')
    radial[0, :] = 0
    radial[0, :3] = np.array([-3., 4., -1.])/(2*h)
    radial[-1, :] = 0
    radial[-1, -1] = 1
    interior = np.ones(n)
    interior[[0, -1]] = 0
    matrix = (kron(eye(modes), radial.tocsr())
              - kron(diags(ell**2), diags(interior)))
    angular = diags((ell[1:], -ell[:-1]), (-1, 1), shape=(modes, modes))
    matrix += kron(angular, diags(radius**2*interior))
    robin = np.zeros(modes*n)
    robin[::n] = -ell
    matrix = (matrix+diags(robin)).tocsc()
    rhs = np.zeros((modes, n))
    rhs[0] = -radius**2*interior
    # c_1,s-c_1 = -Rmin**2/3 + O(Rmin**3).
    rhs[0, 0] = -rmin**2/3
    rhs[:, -1] = ive(ell, 2*rmax**2)/ive(0, 2*rmax**2)
    moments = spsolve(matrix, rhs.ravel()).reshape(modes, n)
    origin = float(moments[0, 0]+rmin**2/6)
    finite = float(simpson(moments[0], x=s)-np.log(rmax))
    return dict(modes=modes, intervals=intervals, rmin=rmin, rmax=rmax,
                C=finite+origin+tail_correction(rmax))


def run_checks():
    """Refine radial and angular resolution and integration limits."""
    rows = []
    # Halving h on the same endpoints permits a second-order Richardson check.
    specifications = (
        [('radial', 192, n, 32., 1.e-6) for n in (1024, 2048, 4096, 8192)]
        + [('modes', m, 4096, 24., 1.e-6) for m in (48, 64, 96, 128, 144, 192)]
        + [('domain', int(6*r), 4096, float(r), 1.e-6)
           for r in (8, 12, 16, 24, 32, 48)]
        + [('origin', 144, 4096, 24., lo) for lo in (1.e-4, 1.e-5, 1.e-6, 1.e-7)]
    )
    cache = {}
    for family, modes, intervals, rmax, rmin in specifications:
        key = modes, intervals, rmax, rmin
        if key not in cache:
            cache[key] = solve_inner(*key)
        row = dict(family=family, **cache[key])
        rows.append(row)
        print(f'{family}: M={modes} intervals={intervals} Rmax={rmax:g} '
              f'Rmin={rmin:g} C={row["C"]:.12f}', flush=True)
    radial = [row for row in rows if row['family'] == 'radial']
    extrapolated = [(4*b['C']-a['C'])/3 for a, b in zip(radial, radial[1:])]
    final = radial[-1]
    summary = dict(reference_C=0.58592, finest_C=final['C'],
                   extrapolated_C=extrapolated[-1],
                   second_order_extrapolations=extrapolated,
                   last_extrapolation_change=extrapolated[-1]-extrapolated[-2],
                   comparisons=rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'validation')
    args = parser.parse_args()
    summary = run_checks()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'inner_constant.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Five-decimal reproduction: C={summary["extrapolated_C"]:.5f}')


if __name__ == '__main__':
    main()
