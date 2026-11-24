"""Compare the radial EPR expansion with quadrature at beta=7."""

import argparse
from pathlib import Path

from figure2 import find_minimum, load_parameters, quadrature_settings, theory_moments, write_csv

HERE = Path(__file__).resolve().parent


def series_ratio(beta, k, terms=150, dps=60):
    """EPR expansion with n=0,...,terms-1, using arbitrary-precision moments."""
    import mpmath as mp
    with mp.workdps(int(dps)):
        b, coupling = mp.mpf(str(beta)), mp.mpf(str(k))
        if b <= 0 or coupling < 0:
            raise ValueError("Require beta > 0 and k >= 0")
        def base_integral(p):
            return mp.quad(lambda r: r**p*mp.exp(b*(r*r/2-r**3/3)), [0, 1, mp.inf])
        moments = {-1: base_integral(0), 0: base_integral(1)}
        moments[1] = moments[0]+1/b
        for n in range(2, 2*int(terms)+1):
            moments[n] = moments[n-1]+mp.mpf(n-1)/b*moments[n-3]
        i0 = moments[0]
        moments = {n: value/i0 for n, value in moments.items()}
        numerator, denominator, coefficient = mp.mpf(0), mp.mpf(0), mp.mpf(1)
        for n in range(int(terms)):
            if n:
                coefficient *= -b*coupling/(2*n)
            s = mp.fsum((-1)**j*mp.binomial(2*n, j)*moments[2*n-j]*moments[j]
                         for j in range(2*n+1))
            t = mp.fsum((-1)**j*mp.binomial(2*n, j)*moments[2*n-j+2]*moments[j]
                         for j in range(2*n+1))
            denominator += coefficient*s
            numerator += coefficient*t
        ratio = numerator/denominator/moments[2]
        return float(ratio)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'data/series.csv')
    args = parser.parse_args()
    settings = quadrature_settings(load_parameters(HERE))
    minimum = find_minimum(7, **settings)
    rows = []
    for k in (minimum['k'], .65):
        reference = theory_moments(7, k, **settings)['ratio']
        for terms, digits in ((100, 60), (150, 30), (150, 60)):
            value = series_ratio(7, k, terms, digits)
            rows.append(dict(beta=7, k=k, series_terms=terms, decimal_digits=digits,
                             series_epr_ratio=value, quadrature_epr_ratio=reference,
                             absolute_difference=abs(value-reference)))
    write_csv(args.output, rows)
    print(f'Saved series comparisons in {args.output}')


if __name__ == '__main__':
    main()
