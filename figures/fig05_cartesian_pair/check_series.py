"""Compare the Cartesian EPR expansion with quadrature near two minima."""

import argparse
from pathlib import Path

from figure5 import load_parameters, theory_moments, write_csv

HERE = Path(__file__).resolve().parent


def _series_coefficients(beta_string, terms, dps):
    """All recurrence, binomial and moment arithmetic stays at requested precision."""
    import mpmath as mp
    with mp.workdps(int(dps)):
        b = mp.mpf(beta_string)
        def base_integral(p):
            return mp.quad(lambda r: r**p*mp.exp(b*(r*r/2-r**3/3)), [0, 1, mp.inf])
        moments = {-1: base_integral(0), 0: base_integral(1)}
        moments[1] = moments[0]+1/b
        for n in range(2, 2*int(terms)+1):
            moments[n] = moments[n-1]+mp.mpf(n-1)/b*moments[n-3]
        i0 = moments[0]
        moments = {n: value/i0 for n, value in moments.items()}
        a, c = [], []
        for n in range(int(terms)):
            a.append(mp.fsum(mp.binomial(n,j)**2*moments[2*j]*moments[2*(n-j)]
                             for j in range(n+1)))
            c.append(mp.fsum(mp.binomial(n,j)**2*moments[2*j+2]*moments[2*(n-j)]
                             for j in range(n+1)))
        return tuple(a), tuple(c), moments[2]


def series_ratio(beta, k, terms=150, dps=100):
    """EPR expansion with n=0,...,terms-1, using arbitrary-precision moments."""
    import mpmath as mp
    if beta <= 0 or k < 0 or int(terms) != terms or terms < 1 or int(dps) < 20:
        raise ValueError("Require beta>0, k>=0, positive integer terms and dps>=20")
    with mp.workdps(int(dps)):
        b, coupling = mp.mpf(str(beta)), mp.mpf(str(k))
        a, c, m2 = _series_coefficients(str(beta), int(terms), int(dps))
        numerator = denominator = mp.mpf(0)
        coefficient = mp.mpf(1)
        for n in range(int(terms)):
            if n:
                coefficient *= -b*coupling/(2*n)
            denominator += coefficient*a[n]
            numerator += coefficient*c[n]
        ratio = numerator/denominator/m2
        return float(ratio)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'data/series.csv')
    args = parser.parse_args()
    params = load_parameters(HERE)
    rows = []
    for beta,settings in ((7.,((100,50),(150,50),(150,100))),
                          (10/3,((150,50),(150,100),(750,100)))):
        k = params['minima'][f'{beta:g}']['k']
        reference = theory_moments(beta,k,order=2*params['theory']['order'])['ratio']
        for terms,digits in settings:
            value = series_ratio(beta,k,terms,digits)
            rows.append(dict(beta=beta,k=k,series_terms=terms,decimal_digits=digits,
                series_epr_ratio=value,quadrature_epr_ratio=reference,
                absolute_difference=abs(value-reference)))
    write_csv(args.output,rows)
    print(f'Saved series comparisons in {args.output}')


if __name__=='__main__':
    main()
