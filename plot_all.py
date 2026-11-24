#!/usr/bin/env python3
"""Render the numerical manuscript figures (2–8 and 10–13) from saved data."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIGURES = (
    'fig02_radial_pair/figure2.py',
    'fig03_radial_population/figure3.py',
    'fig04_phase_coupling/figure4.py',
    'fig05_cartesian_pair/figure5.py',
    'fig06_cartesian_population/figure6.py',
    'fig07_single_coordinate/epr/figure7.py',
    'fig08_coherence/figure8.py',
    'fig10_radial_deformation/figure10.py',
    'fig11_strong_phase_coupling/figure11.py',
    'fig12_cartesian_deformation/figure12.py',
    'fig13_attractive_repulsive/figure13.py',
)

if __name__ == '__main__':
    for figure in FIGURES:
        print(figure, flush=True)
        subprocess.run([sys.executable, str(ROOT / 'figures' / figure), 'plot'],
                       cwd=ROOT, check=True)
