# Numerical validation

Manuscript error bars show one standard error. Simulation comparisons in
validation use pointwise 95% confidence intervals. Deterministic checks
report the change under refinement.

| Figures | What the checks show |
| --- | --- |
| [2](figures/fig02_radial_pair/), [3](figures/fig03_radial_population/), [5](figures/fig05_cartesian_pair/), [6](figures/fig06_cartesian_population/) | Stationary radial moments benchmark trajectory EPR. The shallow radial dip in Figure 2 is better resolved in depth than in location. |
| [4](figures/fig04_phase_coupling/) | In panel (a), trajectory means differ from the plotted stationary PDE curve by less than 1%. The sampled minimum and the weak-coupling boundary are calculated separately. |
| [7](figures/fig07_single_coordinate/) | The checks measure residual EPR changes at large finite coupling and the sensitivity of effective frequency to initialization near locking. |
| [8](figures/fig08_coherence/) | The collective coherence gain at $N=50$ is much larger than the observed change under the final timestep halving. |
| [10](figures/fig10_radial_deformation/), [12](figures/fig12_cartesian_deformation/) | Coupling-grid changes are largest at the lower endpoints, where the derivatives use one-sided differences. The radial cutoff affects uncoupled quantities at the highest temperature. |
| [11](figures/fig11_strong_phase_coupling/) | The linear correction improves the strong-coupling approximation at every tested coupling. Trajectory refinement interval half-widths exceed the observed PDE grid shifts. |
| [13](figures/fig13_attractive_repulsive/) | Refinement changes are at roundoff for radial and full Cartesian quadrature and of order $10^{-6}$ in normalized EPR for the phase radial grid. |

The linked figure READMEs give the exact comparisons, validation plots and
replay commands.
