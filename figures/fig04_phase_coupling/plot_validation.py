"""Plot the saved Figure 4 EPR comparisons and trajectory refinements."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from figure4 import COLORS, load_parameters

HERE = Path(__file__).resolve().parent
ESTIMATORS = {'stratonovich_epr': ('Stratonovich', 'x'),
              'observable_epr': ('Observable', 'o')}


def interval(ax, x, y, half_width, marker, color, *, horizontal=False):
    errors = {'xerr' if horizontal else 'yerr': 1e3*half_width}
    ax.errorbar(x,y,**errors,fmt=marker,color=color,mfc='white',ms=4.5,lw=1,capsize=2)


def plot_benchmarks(ax, rows, colors):
    ax.axvline(0,color='.6',lw=.8,ls='--')
    for i,row in enumerate(rows):
        for offset,(name,(_,marker)) in zip((-.13,.13),ESTIMATORS.items()):
            difference = row[name]-row['stationary_pde_epr']
            interval(ax,1e3*difference,i+offset,row[name+'_ci95_half_width'],
                     marker,colors[row['delta_omega']],horizontal=True)
    labels = [rf"$({row['delta_omega']:g},\,{row['k']:g})$" for row in rows]
    ax.set_yticks(range(len(rows)),labels=labels)
    ax.invert_yaxis()
    ax.set(xlabel=r'$10^3(\sigma-\sigma_{\mathrm{PDE}})/\sigma(0)$',ylabel=r'$(\Delta\omega,k)$')


def plot_refinements(ax, rows, cases, colors, changes):
    ax.axhline(0,color='.6',lw=.8,ls='--')
    offsets = dict(zip(cases,np.linspace(-.29,.29,len(cases))))
    positions = {key:i for i,key in enumerate(changes)}
    for row in rows:
        if row['changed'] not in positions: continue
        x = positions[row['changed']]+offsets[row['delta_omega'],row['k']]
        for offset,(name,(_,marker)) in zip((-.02,.02),ESTIMATORS.items()):
            interval(ax,x+offset,1e3*row[name+'_change'],row[name+'_change_ci95_half_width'],
                     marker,colors[row['delta_omega']])
    ax.set_xticks(range(len(changes)),labels=list(changes.values()))
    ax.set_ylabel(r'$10^3(\sigma_{\mathrm{refined}}-\sigma_{\mathrm{base}})/\sigma(0)$')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=HERE, help='Run directory with data/validation.json')
    parser.add_argument('--output', type=Path, help='Output SVG (default: validation.svg in the run directory)')
    args = parser.parse_args()
    output = args.output or args.outdir/'validation.svg'
    params = load_parameters(args.outdir)
    checks = json.loads((args.outdir/'data/validation.json').read_text())
    by_id = {r['run_id']:r for r in checks['estimator_benchmarks']}
    benchmarks = [by_id[name] for name in params['refinements']['baseline_runs']]
    cases = [(r['delta_omega'],r['k']) for r in benchmarks]
    colors = dict(zip(params['model']['deltas'],COLORS))
    timestep = {'coarser_timestep':r'$2\Delta t$','finer_timestep':r'$\Delta t/2$'}
    duration = {'burn_in':r'Equilibration $\times2$','measurement_duration':r'Duration $\times2$'}
    with plt.rc_context({'text.usetex':True,'font.family':'serif','font.serif':['Computer Modern Roman'],
            'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,
            'axes.linewidth':.8,'xtick.major.size':6,'ytick.major.size':6,
            'svg.hashsalt':'figure4-validation','savefig.bbox':'tight','savefig.facecolor':'white'}):
        fig,axes = plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
        plot_benchmarks(axes[0],benchmarks,colors)
        plot_refinements(axes[1],checks['timestep_burn_duration_refinements'],cases,colors,timestep)
        plot_refinements(axes[2],checks['timestep_burn_duration_refinements'],cases,colors,duration)
        for letter,ax,title in zip('abc',axes,('EPR benchmarks','Timestep refinements','Equilibration and duration')):
            ax.set_title(f'({letter})  {title}',loc='left',fontsize=12)
        handles = [Line2D([],[],marker=marker,mfc='white',ls='',color='.2',label=label)
                   for label,marker in ESTIMATORS.values()]
        fig.legend(handles=handles,loc='outside upper center',ncols=2,frameon=False,
                   title=rf"$\beta={1/params['model']['T_eff']:g},\quad\bar\omega={params['model']['omega_bar']:g}$")
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=220,metadata={'Date':None})
        plt.close(fig)


if __name__=='__main__':
    main()
