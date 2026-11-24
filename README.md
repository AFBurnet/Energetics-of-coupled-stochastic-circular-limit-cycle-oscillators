This repository contains the code and saved data for the paper:

> *Energetics of stochastic limit-cycle oscillators: when does coupling reduce dissipation?*

Preprint: [arXiv:2604.24553](https://arxiv.org/abs/2604.24553) · [DOI: 10.48550/arXiv.2604.24553](https://doi.org/10.48550/arXiv.2604.24553)

## Reproduce the numerical figures

Use Python 3.10 or later on macOS or Linux. Run these commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python plot_all.py
```

Rendering uses LaTeX (`latex`, `dvipng`, `dvips`, `type1cm`, `cm-super`),
Ghostscript for EPS export, and Cairo. `plot_all.py` draws Figures 2–8
and 10–13 from the saved data.

For a single figure, run `python figure#.py plot` from its folder and replace
`#` with the figure number. For Figure 7, use `python epr/figure7.py plot`.
The output goes into `figures/` within the figure folder (`epr/figures/` for Figure 7).

## Figures

| Figure | Code, data and settings |
| --- | --- |
| 2 | [Radial coupling: pairs](figures/fig02_radial_pair/) |
| 3 | [Radial coupling: populations](figures/fig03_radial_population/) |
| 4 | [Phase coupling](figures/fig04_phase_coupling/) |
| 5 | [Cartesian coupling: pairs](figures/fig05_cartesian_pair/) |
| 6 | [Cartesian coupling: populations](figures/fig06_cartesian_population/) |
| 7 | [Single-coordinate coupling and frequency](figures/fig07_single_coordinate/) |
| 8 | [Coherence and EPR](figures/fig08_coherence/) |
| 10 | [Radial deformation](figures/fig10_radial_deformation/) |
| 11 | [Strong phase coupling](figures/fig11_strong_phase_coupling/) |
| 12 | [Cartesian deformation](figures/fig12_cartesian_deformation/) |
| 13 | [Attractive and repulsive coupling](figures/fig13_attractive_repulsive/) |

Each figure README lists the inputs and commands for new calculations.

[Validation](VALIDATION.md) summarizes what the checks show. Run
`python plot_validation.py` from a figure folder to draw its validation plot.
Pass `--outdir` to use a recalculated run. The
[separate data bundle](data-bundles/README.md) supports replay of simulation checks.

## Validation data

[Download `validation-data.tar.gz` (301 MB)](https://github.com/AFBurnet/Energetics-of-coupled-stochastic-circular-limit-cycle-oscillators/releases/download/revision-update/validation-data.tar.gz). See the
[bundle instructions](data-bundles/README.md) to restore it and replay the
checks.
