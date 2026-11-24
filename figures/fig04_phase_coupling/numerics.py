"""Corrected two-oscillator phase-coupling numerics (dimensionless, C=r_c=1).

No plotting or file management here. Delta_omega = omega_2 - omega_1.
Radial probability is r*p0(r)*dr, not p0(r)*dr. No positive radius floor.
"""
import hashlib
import warnings
import numpy as np
from numba import njit, prange
from scipy.integrate import quad
from scipy.sparse import block_diag, diags, eye, kron
from scipy.sparse.linalg import LinearOperator, gmres, spilu

MEASURES = ('stratonovich', 'observable', 'intrinsic', 'cross',
            'coupling_squared', 'phase_derivative', 'mean_r2')


@njit(cache=True)
def fields(x, k, T_eff, omega_bar, delta_omega):
    """Cartesian drift, tangential force Q, four observable terms, mean r^2."""
    r1, r2 = np.hypot(x[0], x[1]), np.hypot(x[2], x[3])
    den = r1*r2
    s, c = 0.0, 0.0
    if den > 0:  # define only the value at exactly zero; never floor a radius
        s = (x[0]*x[3] - x[1]*x[2])/den
        c = (x[0]*x[2] + x[1]*x[3])/den
    w1, w2 = omega_bar-delta_omega/2, omega_bar+delta_omega/2
    v1, v2 = -w1+k*s, -w2-k*s
    q = np.array([-v1*x[1], v1*x[0], -v2*x[3], v2*x[2]])
    f = q + np.array([(1-r1)*x[0], (1-r1)*x[1],
                      (1-r2)*x[2], (1-r2)*x[3]])
    terms = np.array([(w1*w1*r1*r1+w2*w2*r2*r2)/T_eff,
                      2*k*s*(w2*r2*r2-w1*r1*r1)/T_eff,
                      k*k*(r1*r1+r2*r2)*s*s/T_eff, -2*k*c])
    return f, q, terms, (r1*r1+r2*r2)/2


@njit(cache=True)
def one_path(k, T_eff, omega_bar, delta_omega, dt, burn, duration, seed):
    """Streaming Heun path; return averages in MEASURES order (no saved path)."""
    np.random.seed(seed)  # seed compiled RNG, independently for each replica
    th1, th2 = 2*np.pi*np.random.random(), 2*np.pi*np.random.random()
    x = np.array([np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2)])
    nb, ns = int(np.ceil(burn/dt)), int(np.ceil(duration/dt))
    heat, r2_sum = 0.0, 0.0
    term_sum = np.zeros(4)
    f, q, terms, r2 = fields(x, k, T_eff, omega_bar, delta_omega)
    for n in range(nb+ns):
        noise = np.sqrt(2*T_eff*dt)*np.random.randn(4)
        xp = x+dt*f+noise
        fp, _, _, _ = fields(xp, k, T_eff, omega_bar, delta_omega)
        xn = x+0.5*dt*(f+fp)+noise  # SAME noise in predictor and corrector
        fn, qn, tn, r2n = fields(xn, k, T_eff, omega_bar, delta_omega)
        if n >= nb:
            # Full drift at the arithmetic midpoint of the accepted states.
            midpoint_force, _, _, _ = fields((x+xn)/2, k, T_eff, omega_bar, delta_omega)
            heat += np.dot(midpoint_force, xn-x)
            term_sum += 0.5*(terms+tn)
            r2_sum += 0.5*(r2+r2n)
        x, f, q, terms, r2 = xn, fn, qn, tn, r2n
    result = np.empty(7)
    result[0] = heat/(T_eff*ns*dt)
    result[2:6] = term_sum/ns
    result[1] = result[2:6].sum()  # keep covariance of the observable terms
    result[6] = r2_sum/ns
    return result


@njit(cache=True, parallel=True)
def _replicas(k, T_eff, omega_bar, delta_omega, dt, burn, duration, seeds):
    result = np.empty((len(seeds), 7))
    for j in prange(len(seeds)):
        result[j] = one_path(k, T_eff, omega_bar, delta_omega,
                             dt, burn, duration, seeds[j])
    return result


def simulate(k, T_eff, omega_bar, delta_omega, n_replicas, burn, duration,
             dt_max, step_scale, seed):
    """Independent replica estimates and their exact seeds/time step."""
    wmax = max(abs(omega_bar-delta_omega/2), abs(omega_bar+delta_omega/2))
    dt = min(dt_max, step_scale/(1+2*k+wmax))
    # Seeds depend on k, not its index: extending/reordering the sweep is safe.
    tag = int.from_bytes(hashlib.sha256(float(k).hex().encode()).digest()[:8], 'little')
    seeds = np.random.SeedSequence([seed, tag]).generate_state(n_replicas)
    values = _replicas(k, T_eff, omega_bar, delta_omega, dt, burn, duration, seeds)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f'Nonfinite trajectory at k={k:g}; reduce the timestep.')
    return dict(values=values, seeds=seeds, dt=dt,
                actual_duration=np.ceil(duration/dt)*dt)


def radial_weight(r, T_eff):
    """Unnormalised p0, shifted by its maximum to avoid overflow."""
    return np.exp(-(r-1)**2*(2*r+1)/(6*T_eff))


def free_second(T_eff):
    """Uncoupled radial second moment from continuum quadrature."""
    p = lambda r: radial_weight(r, T_eff)
    integrate = lambda f: (quad(f, 0, 1, epsabs=1e-11)[0]
                           + quad(f, 1, np.inf, epsabs=1e-11)[0])
    z = integrate(lambda r: r*p(r))
    if not np.isfinite(z) or z <= 0:
        raise ValueError('Radial normalisation failed at this T_eff.')
    return integrate(lambda r: r**3*p(r))/z


def radial_mesh(T_eff, nr, rmax):
    """Reversible finite volumes: S = sqrt(mu) L_r / sqrt(mu), symmetric.

    Cell edges include zero; positive cell centres are NOT a radius cutoff.
    Boundary conductivities vanish. mu approximates the measure r*p0(r)*dr.
    """
    edges = rmax*(np.arange(nr+1)/nr)**2
    r = (edges[1:]+edges[:-1])/2
    mass = r*radial_weight(r, T_eff)*np.diff(edges)
    if np.any(mass <= 0) or not np.all(np.isfinite(mass)):
        raise ValueError('Radial weights underflow; adjust rmax/T_eff (no weight clipping).')
    mu = mass/mass.sum()
    g = T_eff*edges[1:-1]*radial_weight(edges[1:-1], T_eff)/(np.diff(r)*mass.sum())
    diag = -(np.r_[g, 0]+np.r_[0, g])/mu
    offdiag = g/(np.sqrt(mu[:-1])*np.sqrt(mu[1:]))
    return r, mu, diags([offdiag, diag, offdiag], [-1, 0, 1], format='csr')


def solve_pde(k, T_eff, omega_bar, delta_omega, nr, rmax, modes, return_solution=False):
    """FULL stationary Fourier PDE, c0=1, c_(L+1)=0; both radii evolve.

    (L_r - ell^2 D_phiphi + i ell Delta_omega)c_ell
        + k ell (c_(ell-1)-c_(ell+1)) = 0.
    Uses the observable EPR and checks an independent discrete moment identity.
    return_solution also returns Fourier coefficients and the radial mesh.
    """
    modes = int(modes or max(12, np.ceil(8*np.sqrt(k/T_eff))))
    r, mu, S = radial_mesh(T_eff, nr, rmax)
    root = np.outer(np.sqrt(mu), np.sqrt(mu)).ravel()
    weight = root**2
    a = (r[:, None]**2+r[None, :]**2).ravel()
    D = (T_eff/r[:, None]**2+T_eff/r[None, :]**2).ravel()
    L = (kron(S, eye(nr))+kron(eye(nr), S)).tocsr()
    c = np.zeros((modes, nr*nr), dtype=complex)
    c1, c2 = c[0], c[1]
    residual, tail = 0.0, 0.0
    if k > 0:
        # y_ell = sqrt(weight)*c_ell; multiply each equation by -1/ell.
        blocks = [-L/ell+diags(ell*D-1j*delta_omega) for ell in range(1, modes+1)]
        coupling = diags([-np.ones(modes-1), np.ones(modes-1)], [-1, 1])
        M = block_diag(blocks, format='csc')+k*kron(coupling, eye(nr*nr), format='csc')
        rhs = np.zeros((modes, nr*nr), dtype=complex)
        rhs[0] = k*root  # forcing from c0=1
        rhs = rhs.ravel()
        ilu = spilu(M, drop_tol=1e-4, fill_factor=12,
                    permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0)
        P = LinearOperator(M.shape, matvec=ilu.solve, dtype=complex)
        y, info = gmres(M, rhs, M=P, rtol=1e-10, atol=0, restart=40, maxiter=100)
        residual = np.linalg.norm(M@y-rhs)/np.linalg.norm(rhs)
        if info != 0 or not np.isfinite(residual) or residual > 1e-8:
            raise RuntimeError(f'PDE solve failed at k={k:g}: info={info}, residual={residual:g}')
        c = y.reshape(modes, nr*nr)/root
        c1, c2 = c[0], c[1]
        tail = weight@np.abs(c[-1])
        if tail > 1e-3:
            warnings.warn(f'k={k:g}: last Fourier mode appreciable; increase modes.')
        if r[0] > 0.1*np.sqrt(T_eff/k):
            warnings.warn(f'k={k:g}: increase nr to resolve sqrt(T_eff/k).')
    # Re c1=<cos phi>; -Im c1=<sin phi>; (1-Re c2)/2=<sin^2 phi>.
    sigma0 = (2*omega_bar**2+delta_omega**2/2)*(mu@(r*r))/T_eff
    terms = np.array([sigma0, -delta_omega*k/T_eff*((weight*a)@c1.imag),
                      k*k/(2*T_eff)*((weight*a)@(1-c2.real)), -2*k*(weight@c1.real)])
    # Discrete self-adjointness, not the continuum polynomial, for this check.
    La = (L@(root*a))/root
    G = a*D/T_eff-4
    check = (sigma0+k/2*((weight*G)@c1.real)-k/(2*T_eff)*((weight*La)@c1.real)
             -delta_omega*k/(2*T_eff)*((weight*a)@c1.imag))
    sigma = terms.sum()
    if not np.isfinite(sigma) or abs(sigma-check) > 1e-6*max(1, abs(sigma)):
        raise RuntimeError('The two discrete EPR identities disagree.')
    result = dict(epr=sigma, terms=terms, epr_check=check, n_modes=modes,
                  relative_residual=residual, last_mode_weight=tail)
    if return_solution:
        conductance = S.diagonal(1) * np.sqrt(mu[:-1] * mu[1:])
        result.update(r=r, radial_mass=mu, radial_conductance=conductance,
                      fourier=c.reshape(modes, nr, nr))
    return result


def decompose_pde_currents(solution, k, T_eff, omega_bar, delta_omega, phase_points=512):
    """Positive current/Fisher decomposition, not the signed observable terms.

    D_phi_phi=T*(1/r1^2+1/r2^2), D_psi_phi=T*(1/r2^2-1/r1^2)/2.
    (D^{-1})_psi_psi=(r1^2+r2^2)/T is a matrix-inverse entry.
    The omega_bar cross term integrates to zero by radial exchange symmetry.
    Radial Fisher terms use the reversible mesh's discrete entropy dissipation,
    sum_edges g*(p_next-p)*(log(p_next)-log(p)), converging to T*int (p_r)^2/p.
    """
    coefficients = solution['fourier']
    if phase_points < max(64, 4 * (len(coefficients) + 1)):
        raise ValueError('Increase phase_points to resolve the Fourier solution.')
    r, mu, conductance = (solution[name] for name in
                          ('r', 'radial_mass', 'radial_conductance'))
    phase = 2 * np.pi * np.arange(phase_points) / phase_points
    ell = np.arange(1, len(coefficients) + 1)
    harmonics = np.exp(1j * ell[:, None] * phase)
    density = (1 + 2 * np.einsum('lij,lp->ijp', coefficients, harmonics).real) / (2 * np.pi)
    derivative = np.einsum('lij,lp->ijp', coefficients,
                           1j * ell[:, None] * harmonics).real / np.pi
    weight = np.outer(mu, mu)
    active = np.all(density > 0, axis=2)
    omitted_mass = float(weight[~active].sum())
    if omitted_mass > 1e-12:
        raise ValueError(f'Nonpositive conditional density on radial mass {omitted_mass:g}; '
                         'increase Fourier modes for the current decomposition.')
    # Exclude only machine-negligible radial cells; never clip a density or a radius.
    density = np.where(active[:, :, None], density, 1 / (2 * np.pi))
    weight = np.where(active, weight, 0.)
    dphase = 2 * np.pi / phase_points
    a = r[:, None]**2 + r[None, :]**2
    diffusion = T_eff * (r[:, None]**-2 + r[None, :]**-2)
    drift = -delta_omega - 2 * k * np.sin(phase)
    current = drift * density - diffusion[:, :, None] * derivative
    sigma_phi = dphase * np.sum(weight[:, :, None] * current**2 /
                                (diffusion[:, :, None] * density))
    mixing = (r[:, None]**2 - r[None, :]**2)**2 / (4 * T_eff * a)
    sigma_mixed = dphase * np.sum(weight[:, :, None] * mixing[:, :, None] *
                                  drift**2 * density)
    log_density = np.log(density)
    radial1 = conductance[:, None] * mu[None, :] * (active[:-1] & active[1:])
    radial2 = mu[:, None] * conductance[None, :] * (active[:, :-1] & active[:, 1:])
    sigma_r1 = dphase * np.sum(radial1[:, :, None] * np.diff(density, axis=0) *
                               np.diff(log_density, axis=0))
    sigma_r2 = dphase * np.sum(radial2[:, :, None] * np.diff(density, axis=1) *
                               np.diff(log_density, axis=1))
    sigma_psi = 2 * omega_bar**2 * (mu @ (r * r)) / T_eff
    terms = np.array([sigma_psi, sigma_phi, sigma_mixed, sigma_r1, sigma_r2])
    total = float(terms.sum())
    relative_error = abs(total - solution['epr']) / max(1., abs(solution['epr']))
    if not np.all(np.isfinite(terms)) or np.any(terms < 0) or relative_error > 1e-6:
        raise RuntimeError(f'Current decomposition does not resolve the PDE EPR '
                           f'(relative error {relative_error:g}); increase modes/phase_points.')
    return dict(current_terms=terms, current_epr_sum=total,
                current_relative_error=relative_error, omitted_radial_mass=omitted_mass,
                phase_points=phase_points)


def solve_pde_contributions(k, T_eff, omega_bar, delta_omega, nr, rmax,
                            modes=48, phase_points=512):
    """Solve with enough harmonics for density derivatives and decompose its EPR."""
    solution = solve_pde(k, T_eff, omega_bar, delta_omega, nr, rmax, modes,
                         return_solution=True)
    contributions = decompose_pde_currents(solution, k, T_eff, omega_bar,
                                           delta_omega, phase_points)
    for name in ('r', 'radial_mass', 'radial_conductance', 'fourier'):
        solution.pop(name)
    return dict(**solution, **contributions)


def mean_variance_decomposition(result, T_eff, omega_bar, delta_omega, m2):
    """Weighted angular means and variations from already cached PDE moments.

    m2 must be the second radial moment on the PDE's mesh. Phase derivatives
    integrate to zero in <r_i^2 v_theta_i>. Radial exchange symmetry gives
    <r_1^2 sin(phi)>=<r_2^2 sin(phi)>=b, and the signed observable cross term
    is 2*Delta*k*b/T. Thus these means use the full PDE, with no frozen radius.
    Ordinary solve_pde moments suffice for M and total R. If current_terms
    are available, also return R's radial Fisher/angular-variation split.
    """
    if m2 <= 0 or not np.isfinite(m2):
        raise ValueError('The radial second moment must be finite and positive.')
    intrinsic = (2 * omega_bar**2 + delta_omega**2 / 2) * m2 / T_eff
    if not np.isclose(result['terms'][0], intrinsic, rtol=1e-10, atol=1e-12):
        raise ValueError('PDE moments and the supplied model/radial mesh do not match.')
    # At zero detuning b=0 by phase-reflection symmetry; avoid division by zero.
    transfer = T_eff * result['terms'][1] / (2 * delta_omega * m2) if delta_omega else 0.
    velocities = np.array([-omega_bar + delta_omega/2 + transfer,
                           -omega_bar - delta_omega/2 - transfer])
    mean = m2 * (velocities[1] - velocities[0])**2 / (2 * T_eff)
    tolerance = 1e-9 * max(1., abs(result['epr']))
    psi = 2 * omega_bar**2 * m2 / T_eff
    if 'current_terms' not in result:
        variation = float(result['epr'] - psi - mean)
        if variation < -tolerance:
            raise RuntimeError('Negative current-variation remainder; refine the PDE solution.')
        variation = max(0., variation)
        return dict(weighted_velocities=velocities, sigma_psi=psi,
                    mean_differential_epr=mean, variation_epr=variation,
                    mean_variance_epr=psi + mean + variation)
    current_terms = result['current_terms']
    angular_variation = float(current_terms[1:3].sum() - mean)
    if angular_variation < -tolerance:
        raise RuntimeError('Negative angular-current variance; refine the PDE solution.')
    angular_variation = max(0., angular_variation)  # only roundoff can reach this branch
    radial_variation = float(current_terms[3:5].sum())
    variation = angular_variation + radial_variation
    reconstructed = float(current_terms[0] + mean + variation)
    relative_error = abs(reconstructed - result['epr']) / max(1., abs(result['epr']))
    if relative_error > 1e-6:
        raise RuntimeError('Mean/variation decomposition does not recover the PDE EPR.')
    return dict(weighted_velocities=velocities, sigma_psi=psi, mean_differential_epr=mean,
                angular_variation_epr=angular_variation, radial_variation_epr=radial_variation,
                variation_epr=variation, mean_variance_epr=reconstructed,
                mean_variance_relative_error=relative_error)


def trajectory_mean_variance_decomposition(values, T_eff, omega_bar, delta_omega,
                                          m2, estimator='stratonovich'):
    """Estimate M and total R from independent replicas in MEASURES order.

    The unchanged radial marginal supplies m2 (a continuum radial quadrature).
    Radial exchange symmetry gives V2-V1=-Delta-T*<cross>/(Delta*m2),
    where cross is the signed observable term, not a current variance. At
    Delta=0, phase reflection gives V1=V2=-omega_bar exactly.

    Independent-replica jackknife pseudovalues remove the square-of-a-sample-
    mean bias in M. They also retain the EPR/mean covariance in the SEM of
    R=sigma-sigma_psi-M. Finite-sample estimates may be negative; do not clip
    them. No density scores, PDE solution, or instantaneous noisy angular
    velocities enter this estimator. It does not split R into its Fisher
    and angular-current-variation contributions.
    """
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != len(MEASURES) or len(values) < 2
            or not np.all(np.isfinite(values))):
        raise ValueError('Require at least two finite replicas in MEASURES order.')
    if not np.all(np.isfinite([T_eff, omega_bar, delta_omega, m2])) or min(T_eff, m2) <= 0:
        raise ValueError('Require finite parameters, T_eff>0 and m2>0.')
    if estimator not in ('stratonovich', 'observable'):
        raise ValueError('Choose the stratonovich or observable EPR estimator.')
    count = len(values)
    cross = values[:, MEASURES.index('cross')]
    difference = (-delta_omega - T_eff * cross / (delta_omega * m2)
                  if delta_omega else np.zeros(count))
    mean_difference = difference.mean()
    leave_one_out = (difference.sum() - difference) / (count - 1)
    prefactor = m2 / (2 * T_eff)
    mean_pseudo = prefactor * (count * mean_difference**2 -
                               (count - 1) * leave_one_out**2)
    epr_pseudo = values[:, MEASURES.index(estimator)]
    psi = 2 * m2 * omega_bar**2 / T_eff
    pseudo = np.column_stack([epr_pseudo, -omega_bar - difference/2,
                              -omega_bar + difference/2, mean_pseudo,
                              epr_pseudo - psi - mean_pseudo])
    estimate = pseudo.mean(axis=0)
    sem = pseudo.std(axis=0, ddof=1) / np.sqrt(count)
    return dict(epr=estimate[0], epr_sem=sem[0], sigma_psi=psi,
                weighted_velocities=estimate[1:3], weighted_velocities_sem=sem[1:3],
                mean_differential_epr=estimate[3], mean_differential_sem=sem[3],
                variation_epr=estimate[4], variation_sem=sem[4])


# Weak-coupling pair response used by the equal-population bimodal panels.
# Uses the radial_mesh and radial_weight definitions above; no root imports.
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve


def radial_extent(T, specified=None, tail_log=40.0):
    """Outer edge where the shifted radial weight equals exp(-tail_log)."""
    if not np.isfinite(T) or T <= 0:
        raise ValueError('Require finite T>0.')
    if specified is not None:
        if not np.isfinite(specified) or specified <= 1:
            raise ValueError('Require a finite outer radius greater than one.')
        return float(specified)
    if not np.isfinite(tail_log) or tail_log <= 0:
        raise ValueError('Require finite tail_log>0.')
    return float(brentq(lambda r: (r-1)**2*(2*r+1)/6 - tail_log*T,
                        1., 1. + np.cbrt(3*tail_log*T)))


def weak_pair_response(T, deltas, nr=80, rmax=None):
    """Full two-radius first-harmonic response C2 for each supplied detuning.

    This is the existing weak-coupling radial resolvent, with the same radial
    finite-volume discretisation as the full stationary PDE. The intrinsic
    second moment uses continuum quadrature. Arrays retain the input detuning
    shape/order; negative detunings use symmetry. Grid error is signed relative
    error in the second moment, not an error bound for C2 itself.
    """
    if not np.isfinite(T) or T <= 0:
        raise ValueError('Require finite T>0.')
    if not isinstance(nr, (int, np.integer)) or nr < 12:
        raise ValueError('Require integer nr>=12.')
    delta = np.asarray(deltas, dtype=float)
    if not np.all(np.isfinite(delta)):
        raise ValueError('Require finite detunings.')
    extent = radial_extent(T, rmax)
    integrate = lambda f: (quad(f, 0, 1, epsabs=1e-11)[0]
                            + quad(f, 1, np.inf, epsabs=1e-11)[0])
    z = integrate(lambda r: r*radial_weight(r, T))
    if not np.isfinite(z) or z <= 0:
        raise RuntimeError('Radial normalisation failed.')
    m2 = integrate(lambda r: r**3*radial_weight(r, T))/z
    r, mu, S = radial_mesh(T, nr, extent)
    L = kron(S, eye(nr)) + kron(eye(nr), S)
    D = (T/r[:, None]**2 + T/r[None, :]**2).ravel()
    A = (diags(D)-L).tocsc()
    root = np.sqrt(np.outer(mu, mu)).ravel()
    weight = root**2
    rsum = (r[:, None]**2 + r[None, :]**2).ravel()
    identity = eye(nr*nr, format='csc')
    unique, inverse = np.unique(np.abs(delta), return_inverse=True)
    response, residual = np.empty(len(unique)), np.empty(len(unique))
    for j, detuning in enumerate(unique):
        matrix = A.astype(complex)-1j*detuning*identity
        y = spsolve(matrix, root, permc_spec='MMD_AT_PLUS_A')
        residual[j] = np.linalg.norm(matrix@y-root)/np.linalg.norm(root)
        if not np.isfinite(residual[j]) or residual[j] > 1e-8:
            raise RuntimeError('Pair resolvent residual too large.')
        h = y/root
        response[j] = float(m2/T - 2*weight@h.real
                            - detuning/T*((weight*rsum)@h.imag))
    if not np.all(np.isfinite(response)):
        raise RuntimeError('Nonfinite weak-coupling pair response.')
    return dict(rmax=extent, m2=float(m2),
                C2=response[inverse].reshape(delta.shape),
                relative_residual=residual[inverse].reshape(delta.shape),
                mean_r2_grid_error=float(mu@(r*r)/m2-1),
                radial_tail_mass=float(quad(lambda r: r*radial_weight(r, T),
                                             extent, np.inf, epsabs=1e-11)[0]/z))


def bimodal_coefficient(N, same, cross):
    """Second-order EPR coefficient for two equal all-to-all frequency groups."""
    N = np.asarray(N, dtype=float)
    if not np.all(np.isfinite(N)) or np.any(N < 2) or np.any(N % 2):
        raise ValueError('The equal bimodal population requires even N>=2.')
    return N*(N-2)/4 * np.asarray(same) + N*N/4 * np.asarray(cross)


def weak_zero_detuning(T, nr=160, rmax=None, upper=2., xtol=1e-9):
    """Locate the weak-response zero within a bracket of opposite signs."""
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError('Require a positive upper detuning.')
    ends = weak_pair_response(T, [0., upper], nr, rmax)
    if ends['C2'][0] <= 0 or ends['C2'][1] > 1e-10:
        raise ValueError('The weak-response zero is outside the detuning bracket.')
    if abs(ends['C2'][1]) <= 1e-10:
        root = upper
    else:
        root = brentq(lambda delta: float(weak_pair_response(T, [delta], nr, rmax)['C2'][0]),
                      0., upper, xtol=xtol)
    result = weak_pair_response(T, [root], nr, rmax)
    return dict(detuning=float(root), C2=float(result['C2'][0]), rmax=result['rmax'],
                radial_tail_mass=result['radial_tail_mass'],
                relative_residual=float(result['relative_residual'][0]))
