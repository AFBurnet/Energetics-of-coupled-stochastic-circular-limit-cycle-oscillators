from simulation_utils import *
from solver_utils import *
import numpy as np
import matplotlib.pyplot as plt

# Example code for simulating the entropy production rate with full cartesian coupling
Example_1 = False

# Example code for using a solver function to compute the EPR for a range of coupling strengths
Example_2 = False

## Example code for computing the effective frequency with single cartesian coupling
Example_3 = False

if Example_1:
    # Parameters for the simulation
    T_total = 500 # Total simulation time
    dt = 1e-3  # Time step
    n_traj = 15000  # Number of trajectories
    omega = 0.2
    C = 1.0  # Restoring force coefficient
    r_c = 1.0  # Radius of limit cycle
    D = 1  # Diffusion coefficient
    k = 0.1  # Coupling strength
    T_sys = D  # System time scale
    beta = C * r_c ** 3 / D
    k_end = 40  # End of coupling strength range
    k_start = 0.0  # Start of coupling strength range

    n_osc = 4 # number of oscillators

    kstheory = np.linspace(k_start, k_end, 100)

    # JIT warm-up
    simulate_entropy_coupled_general(n_osc,
        omega, C, r_c, k, D, dt, dt, 1,
        equilibrate=False, burn_time=0, radial_coupling=False
    )
    EPR = []
    k1 = np.linspace(0, 1, 10)
    k2 = np.linspace(2, 20, 10)
    k3 = np.linspace(21, 40, 20)
    #concatinate ks
    ks = np.concatenate((k1, k2, k3))

    # run simulations for each k and compute EPR
    for k in ks:
        print(k)
        rate = simulate_entropy_coupled_general(n_osc,
            omega, C, r_c, k, D, T_total, dt, n_traj,
            equilibrate=True, burn_time=50, radial_coupling=False
        )
        EPR.append(rate)


    kstheory = np.linspace(k_start, k_end, 100)

    # strong coupling limit
    limit_theory = n_osc * omega ** 2 * M_n(n_osc*beta,2)

    plt.scatter(ks, EPR, marker='x')
    plt.xlabel('Coupling strength k')
    plt.ylabel('Entropy Production Rate σ')
    plt.axhline(limit_theory, color='blue', linestyle='--', label='Theoretical σ')
    plt.ylim(np.min(EPR) - 0.01, np.max(EPR) + 0.01)
    plt.title('Entropy Production Rate vs Coupling Strength')
    plt.show()

if Example_2:
    from joblib import Parallel, delayed

    omega = 0.2
    C = 1.0
    r_c = 1.0
    D = 1.0
    k_start, k_end = 2.0, 15.0

    kstheory = np.linspace(k_start, k_end, 60)

    eprs = Parallel(n_jobs=-1, prefer="processes", verbose=10)(
        delayed(EPR_full_cartesian_numeric_solver())(k, omega, C, D, r_c)
        for k in kstheory
    )

if Example_3:
    n_traj = 1000 # Number of trajectories
    N = 2 # Number of oscillators
    C, rc = 1.0, 1 # Restoring force coefficient and radius of limit cycle
    D = 0.2 # Diffusion coefficient
    beta = C * rc ** 3 / D # Dimensionless inverse temperature
    omega = 0.1 # Natural frequency of the oscillators
    T_total = 300 # Total simulation time
    dt = 1e-3 # Time step

    x0_array = None
    y0_array = None

    ks = np.linspace(0, 40, 20)
    Ds = np.linspace(1, 0.01, 20)

    freqs = []

    for k in ks:
        print(k)
        f_list = []
        for _ in range(n_traj):

            # x-coupling (0). For y-coupling use 1.
            xs, ys = simulate_coupled_circular_trajectories(
                N, C, rc, omega, D, T_total, dt,
                k=k, couple_on=0,
                equilibrate=True, burn_time=300.0, x0_array=x0_array, y0_array=y0_array
            )

            # for i in range(0,N,10):
            x_1 = xs[:, 0]
            x_2 = xs[:, 1]
            y_1 = ys[:, 0]
            y_2 = ys[:, 1]

            # Compute phase
            phi = np.arctan2(y_1, x_1)

            # Unwrap phase to remove 2π jumps
            phi_unwrapped = np.unwrap(phi)

            # Total phase change
            total_phase_change = phi_unwrapped[-1] - phi_unwrapped[0]

            # Number of full rotations around origin
            n_rotations = total_phase_change / (2 * np.pi)

            # Total time
            total_time = len(x_1) * dt

            # Mean angular frequency (rad/time)
            freq = total_phase_change / total_time

            f_list.append(freq)
        mean_freq = np.mean(f_list)
        freqs.append(mean_freq)

    plt.plot(ks, freqs)
    plt.show()
