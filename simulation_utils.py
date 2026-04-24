import numpy as np
from numba import njit, prange
import math
import mpmath as mp

mp.mp.dps = 30  # high precision


@njit(cache=True, parallel=True)
def simulate_entropy_coupled_general(
    n_osc,        # number of oscillators (>=2)
    omega,        # base intrinsic rotation speed
    C,            # radial restoring coefficient
    rc,           # limit-cycle radius
    k,            # coupling strength
    D,            # diffusion coeff
    T_total,      # total sim time (after burn)
    dt,           # timestep
    n_traj,       # number of trajectories
    equilibrate=True,   # whether to do burn-in
    burn_time=0.0,      # if >0, override default
    radial_coupling=False,
    phase_coupling=False,
    cartesian_single=False,   # x-only coupling for ANY N (generalized)
    omegas=None               # optional array of per-oscillator frequencies, length n_osc
):
    # Build per-oscillator frequency array (defaults to 'omega')
    if omegas is None:
        omegas_arr = np.empty(n_osc)
        for i in range(n_osc):
            omegas_arr[i] = omega
    else:
        omegas_arr = np.array(omegas)
        if omegas_arr.shape[0] != n_osc:
            raise ValueError("Length of 'omegas' must equal n_osc")

    T_sys = D

    meas_steps = int(T_total / dt)
    if burn_time > 0.0:
        burn_steps = int(burn_time / dt)
    else:
        burn_steps = int((5.0 / C) / dt)

    sqrt2D   = math.sqrt(2.0 * D)
    inv_norm = 1.0 / (T_sys * T_total)

    sigma_sum = 0.0

    for _traj in prange(n_traj):
        # State arrays
        x  = np.empty(n_osc)
        y  = np.empty(n_osc)

        # Initialize on limit cycle with random phases
        for i in range(n_osc):
            theta = 2.0 * math.pi * np.random.random()
            x[i] = rc * math.cos(theta)
            y[i] = rc * math.sin(theta)

        # 1) Burn-in (equilibration)
        if equilibrate:
            sqrt_dt = math.sqrt(dt)
            for _b in range(burn_steps):
                # radii
                r = np.empty(n_osc)
                for i in range(n_osc):
                    r[i] = math.hypot(x[i], y[i])

                # single-oscillator drifts
                Ft_x = np.empty(n_osc)
                Ft_y = np.empty(n_osc)
                Fr_x = np.empty(n_osc)
                Fr_y = np.empty(n_osc)
                for i in range(n_osc):
                    w = omegas_arr[i]
                    Ft_x[i] =  w * y[i]
                    Ft_y[i] = -w * x[i]
                    coeff = C * (rc - r[i])
                    Fr_x[i] = coeff * x[i]
                    Fr_y[i] = coeff * y[i]

                # coupling drift (uniform for any N)
                Fc_x = np.zeros(n_osc)
                Fc_y = np.zeros(n_osc)

                if radial_coupling:
                    for i in range(n_osc):
                        ci = 0.0
                        ri = r[i]
                        for j in range(n_osc):
                            if j != i:
                                ci += (r[j] - ri) / ri
                        ci *= k
                        Fc_x[i] = ci * x[i]
                        Fc_y[i] = ci * y[i]

                elif phase_coupling:
                    theta = np.empty(n_osc)
                    for i in range(n_osc):
                        theta[i] = math.atan2(y[i], x[i])
                    for i in range(n_osc):
                        cpi = 0.0
                        thi = theta[i]
                        for j in range(n_osc):
                            if j != i:
                                cpi += math.sin(theta[j] - thi)
                        cpi *= k
                        Fc_x[i] = -cpi * y[i]
                        Fc_y[i] =  cpi * x[i]

                elif cartesian_single:
                    # x-only all-to-all
                    for i in range(n_osc):
                        sx = 0.0
                        xi = x[i]
                        for j in range(n_osc):
                            if j != i:
                                sx += (x[j] - xi)
                        Fc_x[i] = k * sx
                        Fc_y[i] = 0.0
                else:
                    # full vector all-to-all
                    for i in range(n_osc):
                        sx = 0.0
                        sy = 0.0
                        xi = x[i]; yi = y[i]
                        for j in range(n_osc):
                            if j != i:
                                sx += (x[j] - xi)
                                sy += (y[j] - yi)
                        Fc_x[i] = k * sx
                        Fc_y[i] = k * sy

                # EM update
                for i in range(n_osc):
                    Dx = Ft_x[i] + Fr_x[i] + Fc_x[i]
                    Dy = Ft_y[i] + Fr_y[i] + Fc_y[i]
                    dWx = np.random.normal(0.0, sqrt_dt)
                    dWy = np.random.normal(0.0, sqrt_dt)
                    x[i] += Dx * dt + sqrt2D * dWx
                    y[i] += Dy * dt + sqrt2D * dWy

        # 2) Measurement phase: accumulate entropy S
        S = 0.0
        sqrt_dt = math.sqrt(dt)
        for _i in range(meas_steps):
            # radii
            r = np.empty(n_osc)
            for i in range(n_osc):
                r[i] = math.hypot(x[i], y[i])

            # drifts at start of step
            Ft_x = np.empty(n_osc)
            Ft_y = np.empty(n_osc)
            Fr_x = np.empty(n_osc)
            Fr_y = np.empty(n_osc)
            for i in range(n_osc):
                w = omegas_arr[i]
                Ft_x[i] =  w * y[i]
                Ft_y[i] = -w * x[i]
                coeff = C * (rc - r[i])
                Fr_x[i] = coeff * x[i]
                Fr_y[i] = coeff * y[i]

            Fc_x = np.zeros(n_osc)
            Fc_y = np.zeros(n_osc)

            if radial_coupling:
                for i in range(n_osc):
                    ci = 0.0
                    ri = r[i]
                    for j in range(n_osc):
                        if j != i:
                            ci += (r[j] - ri) / ri
                    ci *= k
                    Fc_x[i] = ci * x[i]
                    Fc_y[i] = ci * y[i]

            elif phase_coupling:
                theta = np.empty(n_osc)
                for i in range(n_osc):
                    theta[i] = math.atan2(y[i], x[i])
                for i in range(n_osc):
                    cpi = 0.0
                    thi = theta[i]
                    for j in range(n_osc):
                        if j != i:
                            cpi += math.sin(theta[j] - thi)
                    cpi *= k
                    Fc_x[i] = -cpi * y[i]
                    Fc_y[i] =  cpi * x[i]

            elif cartesian_single:
                for i in range(n_osc):
                    sx = 0.0
                    xi = x[i]
                    for j in range(n_osc):
                        if j != i:
                            sx += (x[j] - xi)
                    Fc_x[i] = k * sx
                    Fc_y[i] = 0.0
            else:
                for i in range(n_osc):
                    sx = 0.0
                    sy = 0.0
                    xi = x[i]; yi = y[i]
                    for j in range(n_osc):
                        if j != i:
                            sx += (x[j] - xi)
                            sy += (y[j] - yi)
                    Fc_x[i] = k * sx
                    Fc_y[i] = k * sy

            # E-M tentative step for x_new, y_new
            x_new = np.empty(n_osc)
            y_new = np.empty(n_osc)
            for i in range(n_osc):
                Dx = Ft_x[i] + Fr_x[i] + Fc_x[i]
                Dy = Ft_y[i] + Fr_y[i] + Fc_y[i]
                dWx = np.random.normal(0.0, sqrt_dt)
                dWy = np.random.normal(0.0, sqrt_dt)
                x_new[i] = x[i] + Dx * dt + sqrt2D * dWx
                y_new[i] = y[i] + Dy * dt + sqrt2D * dWy

            # midpoints
            xm = np.empty(n_osc)
            ym = np.empty(n_osc)
            for i in range(n_osc):
                xm[i] = 0.5 * (x[i] + x_new[i])
                ym[i] = 0.5 * (y[i] + y_new[i])

            # midpoint single-oscillator drift
            rm = np.empty(n_osc)
            Fm_x = np.empty(n_osc)
            Fm_y = np.empty(n_osc)
            for i in range(n_osc):
                rm[i] = math.hypot(xm[i], ym[i])
                w = omegas_arr[i]
                Fm_x[i] =  w * ym[i] + C * (rc - rm[i]) * xm[i]
                Fm_y[i] = -w * xm[i] + C * (rc - rm[i]) * ym[i]

            # midpoint coupling
            if radial_coupling:
                for i in range(n_osc):
                    cri = 0.0
                    rmi = rm[i]
                    for j in range(n_osc):
                        if j != i:
                            cri += (rm[j] - rmi) / rmi
                    cri *= k
                    Fm_x[i] += cri * xm[i]
                    Fm_y[i] += cri * ym[i]

            elif phase_coupling:
                thm = np.empty(n_osc)
                for i in range(n_osc):
                    thm[i] = math.atan2(ym[i], xm[i])
                for i in range(n_osc):
                    cpi = 0.0
                    thi = thm[i]
                    for j in range(n_osc):
                        if j != i:
                            cpi += math.sin(thm[j] - thi)
                    cpi *= k
                    Fm_x[i] += -cpi * ym[i]
                    Fm_y[i] +=  cpi * xm[i]

            elif cartesian_single:
                for i in range(n_osc):
                    sx = 0.0
                    xmi = xm[i]
                    for j in range(n_osc):
                        if j != i:
                            sx += (xm[j] - xmi)
                    Fm_x[i] += k * sx
                    # y component stays unchanged (0 addition)

            else:
                for i in range(n_osc):
                    sx = 0.0
                    sy = 0.0
                    xmi = xm[i]; ymi = ym[i]
                    for j in range(n_osc):
                        if j != i:
                            sx += (xm[j] - xmi)
                            sy += (ym[j] - ymi)
                    Fm_x[i] += k * sx
                    Fm_y[i] += k * sy

            # increments & entropy
            for i in range(n_osc):
                dx = x_new[i] - x[i]
                dy = y_new[i] - y[i]
                S += Fm_x[i] * dx + Fm_y[i] * dy

            # advance
            for i in range(n_osc):
                x[i] = x_new[i]
                y[i] = y_new[i]

        sigma_sum += S * inv_norm

    return sigma_sum / n_traj


@njit(cache=True)
def simulate_coupled_circular_trajectories_general(
    N,
    C,
    rc,
    omega,
    D,               # diffusion coefficient
    T_total,         # total measurement time (after burn-in)
    dt,              # time step
    k=0.0,           # coupling strength
    equilibrate=True,
    burn_time=0.0,   # if >0, overrides default burn duration
    x0_array=None,   # optional initial x array (length N)
    y0_array=None,   # optional initial y array (length N)
    rng_seed=-1,     # optional seed (>=0 to set)
    # --- new options (match semantics of your general function) ---
    radial_coupling=False,     # couple radii: ci = k * sum_{j!=i} (rj - ri)/ri ; Fc = ci * (xi, yi)
    phase_coupling=False,      # Kuramoto-like: cpi = k * sum_{j!=i} sin(theta_j - theta_i)
    cartesian_single=False,    # x-only all-to-all (like your cartesian_single)
    omegas=None                # optional array of per-oscillator frequencies, length N
):
    """
    Simulate N noisy planar oscillators with limit-cycle restoring force and optional
    all-to-all coupling in *one* of four modes (priority order):
      1) radial_coupling
      2) phase_coupling
      3) cartesian_single (x-only)
      4) full vector cartesian (default if none of the above is True)

    Returns:
      xs, ys  with shape (meas_steps+1, N). Row 0 is the post-burn-in state.
    """

    # RNG seed
    if rng_seed >= 0:
        np.random.seed(rng_seed)

    # Steps
    meas_steps = int(T_total / dt)
    if burn_time > 0.0:
        burn_steps = int(burn_time / dt)
    else:
        # mirror your first function's fixed burn duration of ~5 time units
        burn_steps = int(5.0 / dt)

    sqrt2D = math.sqrt(2.0 * D)
    sqrt_dt = math.sqrt(dt)

    # Per-oscillator frequencies
    if omegas is None:
        omegas_arr = np.empty(N)
        for i in range(N):
            omegas_arr[i] = omega
    else:
        # Copy into a numba-friendly array and validate length
        omegas_arr = np.empty(N)
        if len(omegas) != N:
            raise ValueError("Length of 'omegas' must equal N")
        for i in range(N):
            omegas_arr[i] = omegas[i]

    # --- initialize state ---
    x = np.empty(N)
    y = np.empty(N)

    if (x0_array is not None) and (y0_array is not None):
        for i in range(N):
            x[i] = x0_array[i]
            y[i] = y0_array[i]
    else:
        for i in range(N):
            theta = 2.0 * math.pi * np.random.random()
            # start on unit circle like your original (not enforced to rc)
            x[i] = math.cos(theta)
            y[i] = math.sin(theta)

    x_new = np.empty(N)
    y_new = np.empty(N)

    # ---------- helper to compute coupling drift Fc_x, Fc_y ----------
    def compute_coupling(Fc_x, Fc_y, x_arr, y_arr):
        if radial_coupling:
            # radial: ci * (x_i, y_i) with ci = k * sum_{j!=i} (r_j - r_i)/r_i
            for i in range(N):
                ri = math.hypot(x_arr[i], y_arr[i])
                ci = 0.0
                for j in range(N):
                    if j != i:
                        rj = math.hypot(x_arr[j], y_arr[j])
                        # avoid 0/0; if ri==0, skip contribution (no direction)
                        if ri != 0.0:
                            ci += (rj - ri) / ri
                ci *= k
                Fc_x[i] = ci * x_arr[i]
                Fc_y[i] = ci * y_arr[i]

        elif phase_coupling:
            # Kuramoto-like on phases: Fc = (-cpi * y_i, cpi * x_i), cpi = k * sum sin(th_j - th_i)
            theta_arr = np.empty(N)
            for i in range(N):
                theta_arr[i] = math.atan2(y_arr[i], x_arr[i])
            for i in range(N):
                thi = theta_arr[i]
                cpi = 0.0
                for j in range(N):
                    if j != i:
                        cpi += math.sin(theta_arr[j] - thi)
                cpi *= k
                Fc_x[i] = -cpi * y_arr[i]
                Fc_y[i] =  cpi * x_arr[i]

        elif cartesian_single:
            # x-only all-to-all: Fc_x = k * sum_{j!=i} (x_j - x_i), Fc_y = 0
            for i in range(N):
                xi = x_arr[i]
                sx = 0.0
                for j in range(N):
                    if j != i:
                        sx += (x_arr[j] - xi)
                Fc_x[i] = k * sx
                Fc_y[i] = 0.0

        else:
            # full vector cartesian: both x and y
            for i in range(N):
                xi = x_arr[i]
                yi = y_arr[i]
                sx = 0.0
                sy = 0.0
                for j in range(N):
                    if j != i:
                        sx += (x_arr[j] - xi)
                        sy += (y_arr[j] - yi)
                Fc_x[i] = k * sx
                Fc_y[i] = k * sy

    # --- burn-in (optional) ---
    if equilibrate and burn_steps > 0:
        for _ in range(burn_steps):
            # Single-oscillator drifts: tangential rotation + radial restoring
            Ft_x = np.empty(N)
            Ft_y = np.empty(N)
            Fr_x = np.empty(N)
            Fr_y = np.empty(N)
            for i in range(N):
                w = omegas_arr[i]
                # tangential (CCW): +w*y, -w*x
                Ft_x[i] =  w * y[i]
                Ft_y[i] = -w * x[i]
                # radial restoring toward rc
                ri = math.hypot(x[i], y[i])
                coeff = C * (rc - ri)
                Fr_x[i] = coeff * x[i]
                Fr_y[i] = coeff * y[i]

            # Coupling drift
            Fc_x = np.zeros(N)
            Fc_y = np.zeros(N)
            compute_coupling(Fc_x, Fc_y, x, y)

            # EM update
            for i in range(N):
                Dx = Ft_x[i] + Fr_x[i] + Fc_x[i]
                Dy = Ft_y[i] + Fr_y[i] + Fc_y[i]
                dWx = np.random.normal(0.0, sqrt_dt)
                dWy = np.random.normal(0.0, sqrt_dt)
                x_new[i] = x[i] + Dx * dt + sqrt2D * dWx
                y_new[i] = y[i] + Dy * dt + sqrt2D * dWy

            for i in range(N):
                x[i] = x_new[i]
                y[i] = y_new[i]

    # --- allocate output (including initial post-burn-in point) ---
    xs = np.empty((meas_steps + 1, N))
    ys = np.empty((meas_steps + 1, N))
    for i in range(N):
        xs[0, i] = x[i]
        ys[0, i] = y[i]

    # --- measurement trajectory ---
    for t in range(meas_steps):
        # Single-oscillator drifts at the start of the step
        Ft_x = np.empty(N)
        Ft_y = np.empty(N)
        Fr_x = np.empty(N)
        Fr_y = np.empty(N)
        for i in range(N):
            w = omegas_arr[i]
            Ft_x[i] =  w * y[i]
            Ft_y[i] = -w * x[i]
            ri = math.hypot(x[i], y[i])
            coeff = C * (rc - ri)
            Fr_x[i] = coeff * x[i]
            Fr_y[i] = coeff * y[i]

        # Coupling drift
        Fc_x = np.zeros(N)
        Fc_y = np.zeros(N)
        compute_coupling(Fc_x, Fc_y, x, y)

        # EM update
        for i in range(N):
            Dx = Ft_x[i] + Fr_x[i] + Fc_x[i]
            Dy = Ft_y[i] + Fr_y[i] + Fc_y[i]
            dWx = np.random.normal(0.0, sqrt_dt)
            dWy = np.random.normal(0.0, sqrt_dt)
            x_new[i] = x[i] + Dx * dt + sqrt2D * dWx
            y_new[i] = y[i] + Dy * dt + sqrt2D * dWy

        # advance + record
        for i in range(N):
            x[i] = x_new[i]
            y[i] = y_new[i]
            xs[t + 1, i] = x[i]
            ys[t + 1, i] = y[i]

    return xs, ys
