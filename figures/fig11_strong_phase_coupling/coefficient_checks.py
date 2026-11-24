#!/usr/bin/env python3
"""Independent radial quadratures for the strong-coupling coefficients.

Radial quadrature uses the saved inner-PDE constant. Finite-coupling
accuracy is assessed separately against the stationary PDE.
"""
import argparse
import json
import math
from pathlib import Path

import mpmath as mp


QUANTITIES = ('m2', 'p00', 'H', 'A', 'B', 'sigma_psi')
PRECISIONS = (35, 55)


def check_inner_constant(value, evidence):
    extrapolated = float(evidence['extrapolated_C'])
    if (float(evidence['reference_C']) != float(value) or not math.isfinite(extrapolated)
            or f'{extrapolated:.5f}' != f'{value:.5f}'):
        raise ValueError('Inner-PDE calculation does not reproduce the supplied five-decimal constant.')
    return extrapolated


def _integrals(T_eff, omega_bar, inner_C, precision):
    """Use the unshifted exponential, independently of numerics.radial_weight."""
    with mp.workdps(precision):
        temperature = mp.mpf(str(T_eff))
        frequency = mp.mpf(str(omega_bar))
        constant = mp.mpf(str(inner_C))
        scale = max(mp.mpf(1), temperature**(mp.mpf(1)/3))
        breaks = sorted(set((mp.mpf(0), mp.mpf(1), 2*scale, 4*scale, 8*scale)))

        def exponent(radius):
            return radius**2/(2*temperature) - radius**3/(3*temperature)

        def weight(radius):
            return mp.exp(exponent(radius))

        z = mp.quad(lambda radius: radius*weight(radius), breaks + [mp.inf])
        second_integral = mp.quad(lambda radius: radius**3*weight(radius),
                                  breaks + [mp.inf])
        # expm1 keeps the explicitly subtracted integrand regular at zero.
        H_integral = (mp.quad(lambda radius: mp.expm1(exponent(radius))/radius,
                              [0, 1])
                      + mp.quad(lambda radius: weight(radius)/radius,
                                breaks[1:] + [mp.inf]))

        def coefficients(normalizer, second, finite_H):
            m2, p00 = second/normalizer, 1/normalizer
            H = finite_H/normalizer
            return dict(m2=m2, p00=p00, H=H, A=m2*p00/2,
                        B=m2*(H+p00*constant)-1,
                        sigma_psi=2*frequency**2*m2/temperature)

        return coefficients(z, second_integral, H_integral)


def run_checks(theory, T_eff=1., omega_bar=1., inner_evidence=None):
    """Compare the supplied coefficients with independent radial quadratures."""
    T_eff, omega_bar = float(T_eff), float(omega_bar)
    if not math.isfinite(T_eff) or T_eff <= 0 or not math.isfinite(omega_bar):
        raise ValueError('Require finite T_eff > 0 and finite omega_bar.')
    supplied = {name: float(theory[name]) for name in (*QUANTITIES, 'inner_C')}
    if not all(math.isfinite(value) for value in supplied.values()):
        raise ValueError('Supplied coefficients must be finite.')
    if inner_evidence is None:
        source = Path(__file__).resolve().parent / 'validation/inner_constant.json'
        inner_evidence = json.loads(source.read_text())
    check_inner_constant(supplied['inner_C'], inner_evidence)
    coarse = _integrals(T_eff, omega_bar, supplied['inner_C'], PRECISIONS[0])
    fine = _integrals(T_eff, omega_bar, supplied['inner_C'], PRECISIONS[1])
    with mp.workdps(PRECISIONS[-1]):
        comparisons = {}
        for name in QUANTITIES:
            difference = mp.mpf(str(supplied[name]))-fine[name]
            comparisons[name] = dict(
                supplied=supplied[name], independent=float(fine[name]),
                difference=float(difference),
                relative_difference=(float(difference/abs(fine[name])) if fine[name] else None),
            )
        precision_differences = {name: mp.nstr(fine[name]-coarse[name], 15)
                                 for name in QUANTITIES}
        result = dict(model=dict(T_eff=T_eff, omega_bar=omega_bar),
                      comparisons=comparisons,
                      precision_refinement=dict(from_dps=PRECISIONS[0], to_dps=PRECISIONS[1],
                                                fine_minus_coarse=precision_differences))
    # Reject non-finite numerical output before it can enter saved records.
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path,
                        default=Path(__file__).parent/'data/settings.json')
    parser.add_argument('--output', '-o', type=Path,
                        default=Path(__file__).parent/'validation/coefficients.json')
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    result = run_checks(settings['theory'], settings['T_eff'], settings['omega_bar'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Saved {len(QUANTITIES)} coefficient comparisons in {args.output}.')


if __name__ == '__main__':
    main()
