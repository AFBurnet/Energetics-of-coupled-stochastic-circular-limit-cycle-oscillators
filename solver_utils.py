import numpy as np
from numba import njit, prange
import math
import mpmath as mp

mp.mp.dps = 30  # high precision

from scipy.special import iv, ive
from joblib import Parallel, delayed



def EPR_full_cartesian_numeric_solver(k, omega, Teff):
    """
    Numerically computes the total entropy production rate σ
    for two double-cartesian-coupled oscillators by directly integrating the
    joint steady-state distribution.

    Parameters:
      k     -- coupling strength
      omega -- intrinsic rotation speed
      Teff  -- effective temperature for single oscillator

    Returns:
      σ_k as a mpmath.mpf
    """

    # single-oscillator unnormalised radial pdf
    def p0(r):
        return mp.e**((1/Teff)*(r**2/2 - r**3/3))

    # joint, unnormalised pdf with coupling
    def p_joint(r1, r2):
        return p0(r1) * p0(r2) * mp.e**(-k*(r1**2 + r2**2)/(2 * Teff)) * mp.besseli(0, k*r1*r2/Teff)

    # integrand for normalisation: p_joint * Jacobian r1*r2
    def integrand_N(r1, r2):
        return p_joint(r1, r2) * r1 * r2

    # compute normalization constant N
    N = mp.quad(lambda r2: mp.quad(lambda r1: integrand_N(r1, r2), [0, mp.inf]), [0, mp.inf])

    # integrand for ⟨r1^2⟩: r1^2 * p_joint * Jacobian
    def integrand_r2(r1, r2):
        return (r1**2) * p_joint(r1, r2) * r1 * r2

    # compute ⟨r1^2⟩
    mean_r2 = mp.quad(lambda r2: mp.quad(lambda r1: integrand_r2(r1, r2), [0, mp.inf]), [0, mp.inf]) / N

    # total EPR σ = 2 * ω^2 / Teff * ⟨r^2⟩_k
    return 2 * omega**2 * mean_r2 / Teff

def EPR_radial_numeric_solver(k, omega, Teff):
    """
    Numerically computes the total entropy production rate σ
    for two radially-coupled oscillators by directly integrating the
    joint steady-state distribution.

    Parameters:
      k     -- coupling strength
      omega -- intrinsic rotation speed
      Teff  -- effective temperature for single oscillator

    Returns:
      σ_k as a mpmath.mpf
    """

    # single-oscillator unnormalised radial pdf
    def p0(r):
        return mp.e**((1/Teff)*(r**2/2 - r**3/3))

    # joint, unnormalised pdf with coupling
    def p_joint(r1, r2):
        return p0(r1) * p0(r2) * mp.e**(-k*(r1 - r2)**2/(2 * Teff))

    # integrand for normalisation: p_joint * Jacobian r1*r2
    def integrand_N(r1, r2):
        return p_joint(r1, r2) * r1 * r2

    # compute normalization constant N
    N = mp.quad(lambda r2: mp.quad(lambda r1: integrand_N(r1, r2), [0, mp.inf]), [0, mp.inf])

    # integrand for ⟨r1^2⟩: r1^2 * p_joint * Jacobian
    def integrand_r2(r1, r2):
        return (r1**2) * p_joint(r1, r2) * r1 * r2

    # compute ⟨r1^2⟩
    mean_r2 = mp.quad(lambda r2: mp.quad(lambda r1: integrand_r2(r1, r2), [0, mp.inf]), [0, mp.inf]) / N

    # total angular EPR σ_θ = 2 * ω^2 / Teff * ⟨r^2⟩
    return 2 * omega**2 * mean_r2 / Teff


def EPR_phase_numeric_solver(k, omega1, omega2, T_eff, Rmax=10.0, Nr=400, N=5):
    """
    The total EPR for the case of non-zero frequency difference
    """
    Delta = omega1 - omega2
    s1 = 2 * (0.5 * (omega1 + omega2)) ** 2 * M_n(1/T_eff, 2) / T_eff
    s2 = phase_current_analytical(k, Delta, T_eff, Rmax=Rmax, Nr=Nr, N=N)
    s3 = phasefisher_analytical(k, Delta, T_eff, Rmax=Rmax, Nr=Nr, N=N)
    s4 = radial_fisher_averaged(k, Delta, T_eff, Rmax=Rmax, Nr=Nr, N=50)
    return s1,s2,s3,s4

def phase_current_analytical(k, Delta, T_eff, Rmax=10.0, Nr=400, N=5):
    """
    Compute sigma_phi contribution: -2πΔ * <D_varphi^{-1} j_phi>_0
    """
    # radial grid
    r = np.linspace(1e-6, Rmax, Nr)
    dr = r[1] - r[0]
    p0 = compute_p0_r(r, T_eff)
    R1, R2 = np.meshgrid(r, r, indexing='ij')
    P0 = np.outer(p0, p0)

    # helper to compute j_phi at given r1,r2
    def j_phi_at(r1, r2):
        # reuse user's compute_j_phi
        return jphi_analytical(1/T_eff, Delta, k, r1, r2, N=N)

    # vectorize j_phi
    j_phi_mat = np.array(Parallel()(delayed(j_phi_at)(R1[i,j], R2[i,j])
                                     for i in range(Nr) for j in range(Nr))).reshape(Nr, Nr)

    # D_varphi = T_eff * (r1^-2 + r2^-2)
    D_phi_inv = (R1**2 + R2**2) / (4*T_eff)

    # integrand
    integrand = (-2 * np.pi * Delta) * D_phi_inv * j_phi_mat * P0 * (R1 * R2)
    I = np.trapz(np.trapz(integrand, r, axis=1), r, axis=0)
    return I

def phasefisher_analytical(k, Delta, T_eff, Rmax=10.0, Nr=400, N=5):
    """
    Numerically compute
      ⟨ Dψψ^{-1} · (Dψφ^2 / Dφφ^2) · [ -2πΔω jφ - Δω^2 + 4k^2⟨sin²φ⟩ ] ⟩_0
    over r1,r2∈[0,Rmax], with radial weight p0(r) ∝ exp((r²/2 - r³/3)/T_eff).
    """
    # 1) build radial grid & weight p0(r)
    r = np.linspace(1e-6, Rmax, Nr)
    dr = r[1] - r[0]
    logp0 = (r ** 2 / 2 - r ** 3 / 3) / T_eff
    logp0 -= logp0.max()  # avoid overflow
    p0 = np.exp(logp0)
    p0 /= np.trapz(p0 * r, r)  # normalize ∫p0(r) r dr = 1

    # 2) mesh
    R1, R2 = np.meshgrid(r, r, indexing='ij')
    P0 = np.outer(p0, p0)

    # 3) diffusion‐matrix elements (T_eff cancels in ratio)
    ratio = (1 / T_eff) * 0.25 * (R2 ** 2 - R1 ** 2) ** 2 / (R1 ** 2 + R2 ** 2)

    # 4) for each (r1,r2) get j_phi and ⟨sin²φ⟩ by sampling p_phi
    sin2 = np.zeros_like(R1)
    jphi = np.zeros_like(R1)
    for i in range(Nr):
        for j in range(Nr):
            r1, r2 = R1[i, j], R2[i, j]
            # reduced current
            jphi[i, j] = jphi_analytical(1 / T_eff, Delta, k, r1, r2, N=N)
            # average sin^2
            sin2[i, j] = sin2phi_analytical(1 / T_eff, Delta, k, r1, r2, N=N)

    # 5) assemble the bracket term
    bracket = (-2 * np.pi * Delta * jphi
               - Delta ** 2
               + 4 * (k ** 2) * sin2)

    # 6) full integrand and double‐trapz
    integrand = ratio * bracket * P0 * (R1 * R2)
    I = np.trapz(np.trapz(integrand, r, axis=1), r, axis=0)

    return I

def jphi_analytical(beta, delta, k, r1, r2, N=5):
    Dphi = (1/beta) * (r1**2 + r2**2)/(r1**2 * r2**2)
    a = -2*k / Dphi
    b = delta / Dphi
    if b != 0:
        if iv(0,a)*iv(0,-a) != np.inf:
            def summand(n):
                return iv(n,a)*iv(n,-a) / (n**2 + b**2)
            SUM = np.sum([summand(n) for n in range(-N,N+1)])
            return -Dphi / (2*np.pi*b*SUM)
        else:
            return 0
    else:
        return 0

def sin2phi_analytical(beta, delta, k, r1, r2, N=5):
    Dphi = (1/beta) * (r1**2 + r2**2)/(r1**2 * r2**2)
    a = -2*k / Dphi
    b = delta / Dphi

    if ive(0, a) * ive(0, -a) != np.inf:
        def summand(n):
            return ive(n, a) * ive(n, -a) / (n ** 2 + b ** 2)
        SUM = np.sum([summand(n) for n in range(-N, N+1)])
        den =  2 * SUM
        def summand2(n):
            return ive(n, a) * (ive(n,-a) - 0.5*ive(n-2,-a)-0.5*ive(n+2,-a)) / (n ** 2 + b ** 2)

        SUM2 = np.sum([summand2(n) for n in range(-N, N+1)])
        num = SUM2
        return num/den
    else:
        return 2*Dphi / k - Dphi**2 / (8*k**2)

def radial_fisher_averaged(k, Delta, T_eff, Rmax=10.0, Nr=400, N=5):
    """
    Computes the EPR from the radial Fisher information
    """
    # 1) build radial grid & weight p0(r)
    r = np.linspace(1e-6, Rmax, Nr)
    dr = r[1] - r[0]
    logp0 = (r ** 2 / 2 - r ** 3 / 3) / T_eff
    logp0 -= logp0.max()  # avoid overflow
    p0 = np.exp(logp0)
    p0 /= np.trapz(p0 * r, r)  # normalise ∫p0(r) r dr = 1

    # 2) mesh
    R1, R2 = np.meshgrid(r, r, indexing='ij')
    P0 = np.outer(p0, p0)


    # 4) for each (r1,r2) get j_phi and ⟨sin²φ⟩ by sampling p_ph
    rad_fisher = np.zeros_like(R1)

    for i in range(Nr):
        for j in range(Nr):
            r1, r2 = R1[i, j], R2[i, j]
            rad_fisher[i, j] = radial_fisher_compute(Delta, k, r1, r2, T_eff, N=N)

    # 5) assemble the bracket term
    bracket = T_eff * rad_fisher

    # 6) full integrand and double‐trapz
    integrand = bracket * P0 * (R1 * R2)
    I = np.trapz(np.trapz(integrand, r, axis=1), r, axis=0)

    return I


def radial_fisher_compute(delta, k, r1, r2, Teff, N=5, nphi=2048):
    """
    Computes the Fisher information I_{r1} and I_{r2}, i.e.
    averages of [∂_{r_i} ln p_phi]^2 w.r.t. p_phi.
    """
    # 1. Precompute parameters and Bessel arrays
    Dphi = Teff * (r1**2 + r2**2) / (r1**2 * r2**2)
    a = -2 * k / Dphi
    b = delta / Dphi

    # apply condition for stability
    if np.abs(a)<30:

        n = np.arange(-N, N+1)
        denom_n = n**2 + b**2
        In_a = ive(n, a)
        In_ma = iv(n, -a)
        # Bessel derivative via recurrence
        dIn_da = 0.5 * (ive(n - 1, a) + ive(n + 1, a))
        dIn_dma = -0.5 * (iv(n - 1, -a) + iv(n + 1, -a))

        # 2. Build phi grid
        phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
        cos_phi = np.cos(phi)

        # 3. Numerator & denominator of p_phi
        # denom (scalar)
        D = 2 * np.pi * b * np.sum(In_a * In_ma / denom_n)

        # For vectorised sums over phi, broadcast n against phi
        n_phi = n[:, None] * phi[None, :]
        cos_nphi = np.cos(n_phi)
        sin_nphi = np.sin(n_phi)

        num_terms = In_a[:, None] * (b * cos_nphi + n[:, None] * sin_nphi) / denom_n[:, None]
        num_phi = np.sum(num_terms, axis=0)

        pphi = np.exp(-a * cos_phi) * num_phi / D

        # 4. Compute partial derivatives of numerator & denominator
        # dnum/da and dnum/db
        dnum_da = np.sum(dIn_da[:, None] * (b * cos_nphi + n[:, None] * sin_nphi) / denom_n[:, None], axis=0)
        d_term_db = ((n**2 - b**2)[:, None] * cos_nphi - 2 * b * n[:, None] * sin_nphi) / denom_n[:, None]**2
        dnum_db = np.sum(In_a[:, None] * d_term_db, axis=0)

        dden_da = 2 * np.pi * b * np.sum(dIn_da * In_ma / denom_n + dIn_dma * In_a / denom_n)
        dden_db = 2 * np.pi * (np.sum(In_a * In_ma / denom_n) + b * np.sum(In_a * In_ma * (-2*b) / denom_n**2))

        # 5. Compute derivatives of a and b w.r.t. r1 (same for r2 by symmetry)
        dD_dr1 = -2 * Teff * r1**-3

        da_dr1 = (2 * k / Dphi**2) * dD_dr1
        db_dr1 = (-delta / Dphi**2) * dD_dr1

        # 6. Score function S1(phi)
        S1 = (-da_dr1 * cos_phi +
              (dnum_da * da_dr1 + dnum_db * db_dr1) / num_phi -
              (dden_da * da_dr1 + dden_db * db_dr1) / D)

        # 7. Fisher information: integrate pphi * S^2 over phi
        I_r1 = np.trapz(pphi * S1**2, phi)
        # I_r2 = np.trapz(pphi * S2**2, phi)
        if np.abs(I_r1) < 8:
            return 2*I_r1
        else:
            # --- compute dimensionless kappa = 2 k * r1^2 r2^2 / ((r1^2+r2^2) T_eff) -----
            denom = r1 ** 2 + r2 ** 2
            kappa = 2 * k * r1 ** 2 * r2 ** 2 / (Teff * denom)
            # print('a', kappa)

            # --- use exponentially scaled Bessel functions for stability --------------
            I0 = ive(0, kappa)
            I1 = ive(1, kappa)
            I2 = ive(2, kappa)

            # --- derivatives dκ/dr1 and dκ/dr2 ---------------------------------------
            dk_dr1 = 4 * k * r1 * r2 ** 4 / (Teff * denom ** 2)
            dk_dr2 = 4 * k * r2 * r1 ** 4 / (Teff * denom ** 2)

            # --- radial Fisher factor: ½(1+I2/I0) - (I1/I0)^2 -------------------------
            F = 0.5 * (1.0 + I2 / I0) - (I1 / I0) ** 2

            # --- assemble the integrand: (dk/dr1)^2 + (dk/dr2)^2 times the Fisher factor
            fisher_r1 = (dk_dr1 ** 2) * F
            fisher_r2 = (dk_dr2 ** 2) * F
            return fisher_r1 + fisher_r2
    else:
        # --- compute dimensionless kappa = 2 k * r1^2 r2^2 / ((r1^2+r2^2) T_eff) -----
        denom = r1 ** 2 + r2 ** 2
        kappa =  2 * k * r1 ** 2 * r2 ** 2 / (Teff * denom)
        #print('a', kappa)

        # --- use exponentially scaled Bessel functions for stability --------------
        I0 = ive(0, kappa)
        I1 = ive(1, kappa)
        I2 = ive(2, kappa)

        # --- derivatives dκ/dr1 and dκ/dr2 ---------------------------------------
        dk_dr1 = 4 * k * r1 * r2 ** 4 / (Teff * denom ** 2)
        dk_dr2 = 4 * k * r2 * r1 ** 4 / (Teff * denom ** 2)

        # --- radial Fisher factor: ½(1+I2/I0) - (I1/I0)^2 -------------------------
        F = 0.5 * (1.0 + I2 / I0) - (I1 / I0) ** 2

        # --- assemble the integrand: (dk/dr1)^2 + (dk/dr2)^2 times the Fisher factor
        fisher_r1 = (dk_dr1 ** 2) * F
        fisher_r2 = (dk_dr2 ** 2) * F
        return fisher_r1 + fisher_r2


def compute_p0_r(r, T_eff):
    """Stationary radial distribution p0(r) normalized so that ∫ p0(r) r dr = 1."""
    log_p0 = (r**2 / 2 - r**3 / 3) / T_eff
    log_p0 -= np.max(log_p0)
    p0 = np.exp(log_p0)
    Z = np.trapz(p0 * r, r)
    return p0 / Z

def I_n(beta, n):
    """
    Computes the un-normalised moment integral I_n(beta) as the
    3F2-expression given in the derivation.
    """
    b = beta
    # shorthand
    z = b/6
    term0 = 8 * mp.gamma((2+n)/3) * mp.hyper([1/3 + n/6, 5/6 + n/6],
                                             [1/3, 2/3], z)
    term1 = 4 * 3**(2/3) * b**(1/3) * mp.gamma((4+n)/3) * mp.hyper(
                [2/3 + n/6, 7/6 + n/6], [2/3, 4/3], z)
    term2 = 3 * 3**(1/3) * b**(2/3) * mp.gamma(2 + n/3) * mp.hyper(
                [1 + n/6, 3/2 + n/6], [4/3, 5/3], z)
    prefactor = (1/8) * 3**((-1+n)/3) * b**(-2/3 - n/3)
    return prefactor * (term0 + term1 + term2)

def M_n(beta, n):
    """Dimensionless moment M_n = I_n / I_0."""
    return I_n(beta, n) / I_n(beta, 0)

# precompute M-array for n=0..2N+2, to be used in the S and T arrays for the epr expansions
def compute_M_array(beta, N):
    """
    Returns numpy array M[0..2N+2], where
      M[n] = I_n(beta) / I_0(beta).
    """
    # compute I[0..2N+2] in high precision then cast to float
    Ivals = [float(I_n(beta, n)) for n in range(2*N+3)]
    I0 = Ivals[0]
    return np.array(Ivals) / I0

# prep fn for radial coupling epr expansion, precomputing S and T arrays for the series coefficients
def compute_S_T(M, N):
    """
    Given M-array (length >= 2N+3), returns two numpy arrays S[0..N], T[0..N]:
      S[n] = sum_{m=0..2n} (-1)^m * comb(2n,m) * M[2n-m]   * M[m]
      T[n] = sum_{m=0..2n} (-1)^m * comb(2n,m) * M[2n-m+2] * M[m]
    """
    S = np.zeros(N+1, dtype=np.float64)
    T = np.zeros(N+1, dtype=np.float64)
    for n in range(N+1):
        Sm = 0.0
        Tm = 0.0
        for m in range(2*n+1):
            sign = -1.0 if (m % 2) else 1.0
            c    = math.comb(2*n, m)
            Sm += sign * c * M[2*n - m]   * M[m]
            Tm += sign * c * M[2*n - m + 2] * M[m]
        S[n] = Sm
        T[n] = Tm
    return S, T


# epr expansion of radial coupling case, with precomputed S and T arrays for the series coefficients
@njit(cache=True)
def epr_radial_jit(k_array, S, T, beta, omega):
    Teff = 1.0 / beta
    N    = S.shape[0] - 1
    out  = np.empty(k_array.shape[0], dtype=np.float64)

    for i in range(k_array.shape[0]):
        kval = k_array[i]
        x    = -kval / (2.0 * Teff)
        fac  = 1.0    # will accumulate n!
        pw   = 1.0    # will accumulate x**n
        norm = 0.0
        epr  = 0.0

        for n in range(N+1):
            if n > 0:
                fac *= n
                pw  *= x
            c = pw / fac
            norm += c * S[n]
            epr  += c * T[n]

        out[i] = 2.0 * omega**2 / Teff * (epr / norm)

    return out

 # prep fn for full cartesian coupling epr expansion
def compute_A_B(M, N):
    """
    Given M-array (len >= 2N+3), returns two arrays A[0..N], B[0..N] with
      A[n] = sum_{m=0..n} comb(n,m)^2 * M[2m]   * M[2(n-m)]
      B[n] = sum_{m=0..n} comb(n,m)^2 * M[2m+2] * M[2(n-m)]
    """
    A = np.zeros(N+1, dtype=np.float64)
    B = np.zeros(N+1, dtype=np.float64)
    for n in range(N+1):
        sA = 0.0
        sB = 0.0
        for m in range(n+1):
            c = math.comb(n, m)
            w = float(c * c)
            sA += w * M[2*m]     * M[2*(n-m)]
            sB += w * M[2*m + 2] * M[2*(n-m)]
        A[n] = sA
        B[n] = sB
    return A, B

# epr expansion of full cartesian coupling case, with precomputed A and B arrays for the series coefficients
@njit(cache=True)
def epr_double_cartesian_jit(k_array, A, B, beta, omega):
    """
    Computes sigma(k) for each k in k_array using the truncated series
      norm = sum_{n=0..N} (x^n / n!) * A[n]
      epr  = sum_{n=0..N} (x^n / n!) * B[n]
      sigma = (2*omega^2 / T) * (epr / norm)
    where x = -k/(2T), T = 1/beta, and N = len(A)-1.
    """
    T = 1.0 / beta
    N = A.shape[0] - 1
    out = np.empty(k_array.shape[0], dtype=np.float64)

    for i in range(k_array.shape[0]):
        k = k_array[i]
        x = -k / (2.0 * T)

        fac = 1.0   # n!
        pw  = 1.0   # x**n
        norm = 0.0
        epr  = 0.0

        for n in range(N+1):
            if n > 0:
                fac *= n
                pw  *= x
            c = pw / fac
            norm += c * A[n]
            epr  += c * B[n]

        out[i] = (2.0 * omega * omega / T) * (epr / norm)

    return out
