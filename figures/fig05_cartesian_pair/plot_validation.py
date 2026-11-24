"""Plot the saved Figure 5 EPR comparisons, refinements and minimum depths."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

from figure5 import load_parameters, publication_style

HERE = Path(__file__).resolve().parent
ESTIMATORS = {'stratonovich_epr': ('Stratonovich','x'),
              'radial_moment_epr': ('Radial moments','o')}
COLORS = {1.: '#9fc1d6',10/3: '#6b9cbd',7.: '#397aa8',10.: '#15537e'}


def interval(ax,x,y,half_width,marker,color,*,horizontal=False):
    errors = {'xerr' if horizontal else 'yerr':1e3*half_width}
    ax.errorbar(x,y,**errors,fmt=marker,color=color,mfc='white',ms=4.5,lw=1,capsize=2)


def plot_benchmarks(ax,rows):
    ax.axvline(0,color='.6',lw=.8,ls='--')
    for i,row in enumerate(rows):
        for offset,(name,(_,marker)) in zip((-.13,.13),ESTIMATORS.items()):
            interval(ax,1e3*(row[name]-row['quadrature_epr']),i+offset,
                     row[name+'_ci95_half_width'],marker,COLORS[row['beta']],horizontal=True)
    labels = []
    for row in rows:
        beta = '10/3' if row['beta']==10/3 else f"{row['beta']:g}"
        labels.append(rf"$({beta},\,{row['k']:.3g})$")
    ax.set_yticks(range(len(rows)),labels=labels)
    ax.invert_yaxis()
    ax.set(xlabel=r'$10^3(\sigma-\sigma_{\mathrm{quad}})/(2\sigma_0)$',ylabel=r'$(\beta,k)$')


def plot_refinements(ax,rows,cases):
    changes = {'half_timestep':r'$\Delta t/2$','quarter_timestep':r'$\Delta t/4$',
               'burn_in':'Equilibration\n'+r'$\times2$','measurement_duration':'Duration\n'+r'$\times2$'}
    positions = {key:i for i,key in enumerate(changes)}
    offsets = dict(zip(cases,np.linspace(-.29,.29,len(cases))))
    ax.axhline(0,color='.6',lw=.8,ls='--')
    for row in rows:
        x = positions[row['changed']]+offsets[row['beta'],row['k']]
        for offset,(name,(_,marker)) in zip((-.025,.025),ESTIMATORS.items()):
            interval(ax,x+offset,1e3*row[name+'_change'],row[name+'_change_ci95_half_width'],
                     marker,COLORS[row['beta']])
    ax.set_xticks(range(len(changes)),labels=list(changes.values()))
    ax.set_ylabel(r'$10^3(\sigma_{\mathrm{refined}}-\sigma_{\mathrm{base}})/(2\sigma_0)$')


def plot_depths(ax,rows,minima):
    ax.axhline(0,color='.6',lw=.8,ls='--')
    for x,beta in enumerate((10/3,7.,10.)):
        for offset,(name,(_,marker)) in zip((-.15,0),ESTIMATORS.items()):
            row = next(r for r in rows if r['beta']==beta and r['estimator']==name)
            interval(ax,x+offset,1e3*row['depth_below_strong_coupling'],
                     row['depth_below_strong_coupling_ci95_half_width'],marker,COLORS[beta])
        m = minima[f'{beta:g}']
        ax.plot(x+.15,1e3*(m['strong_ratio']-m['ratio']),'s',color='.25',ms=4.5)
    ax.set_xticks(range(3),labels=[r'$10/3$',r'$7$',r'$10$'])
    ax.set(xlabel=r'$\beta$',ylabel=r'$10^3[\sigma(\infty)-\sigma(k_*)]/(2\sigma_0)$')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=HERE, help='Directory containing the calculation results')
    parser.add_argument('--output', type=Path, help='Destination SVG, defaults to validation.svg in --outdir')
    args = parser.parse_args()
    output = args.output if args.output is not None else args.outdir/'validation.svg'
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    params = load_parameters(args.outdir)
    checks = json.loads((args.outdir/'data/validation.json').read_text())
    ids = {r['baseline_run'] for r in checks['timestep_burn_duration_refinements']}
    benchmarks = sorted((r for r in checks['estimator_benchmarks'] if r['run_id'] in ids),
                        key=lambda r:(r['beta'],r['k']))
    cases = [(r['beta'],r['k']) for r in benchmarks]
    with publication_style(plt),plt.rc_context({'axes.labelsize':13,'xtick.labelsize':10,
            'ytick.labelsize':11,'svg.hashsalt':'figure5-validation'}):
        fig,axes = plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        plot_benchmarks(axes[0],benchmarks)
        plot_refinements(axes[1],checks['timestep_burn_duration_refinements'],cases)
        plot_depths(axes[2],checks['minimum_depths'],params['minima'])
        for letter,ax,title in zip('abc',axes,('EPR benchmarks','Trajectory refinements','Depth below strong coupling')):
            ax.set_title(f'({letter})  {title}',loc='left',fontsize=12)
        handles = [Line2D([],[],marker=marker,mfc='white',ls='',color='.2',label=label)
                   for label,marker in ESTIMATORS.values()]
        handles.append(Line2D([],[],marker='s',ls='',color='.25',label='Quadrature'))
        fig.legend(handles=handles,loc='outside upper center',ncols=3,frameon=False)
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=220,metadata={'Date':None})
        plt.close(fig)


if __name__=='__main__':
    main()
