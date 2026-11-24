"""Combined Figure 7: plot saved EPR, locking and effective frequency data only.

The trajectory file stores the plotted envelopes and their time coordinates.
The interval bars are illustrative display annotations, not state estimates.
"""
from copy import deepcopy
from pathlib import Path
import importlib.util
import io
import re
import xml.etree.ElementTree as ET

import numpy as np

HERE=Path(__file__).resolve().parent
PUBLICATION_STYLE={'text.usetex':True,'font.family':'serif','font.serif':['Computer Modern Roman'],
       'font.size':21,'axes.labelsize':24,'xtick.labelsize':21,'ytick.labelsize':21,
       'legend.fontsize':21,'axes.linewidth':1.5,'xtick.direction':'in',
       'ytick.direction':'in','xtick.major.size':6,'ytick.major.size':6,
       'svg.fonttype':'path','savefig.bbox':None}


WIDTH = 732.
TOP_HEIGHT = 677.6
ORANGE = '#c78323'
BLUE_FILL = '#b7d6e8'
CORRECTION_COLOR = '#8F5268'
BLUES = ('#9dbdd2', '#739fbd', '#4f85a9', '#2c678f', '#12466b')


def _coupling_points(ax, rows, color):
    rows = sorted(rows, key=lambda r: r['k'])
    x = np.array([r['k'] for r in rows])
    y = np.array([r['stratonovich_epr'] for r in rows])
    ax.errorbar(x, y, yerr=[r['stratonovich_epr_sem'] for r in rows],
                fmt='x', color=color, lw=1.25, ms=7.2, mew=1.3, capsize=2, zorder=3)
    ax.plot(x, y, color=color, lw=1.8, zorder=4)
    return x, y


def _shade_decomposition(ax, x, total, sem, radial, markersize=8, linewidth=2.2):
    ax.fill_between(x, radial, 1., color=BLUE_FILL, alpha=.30, lw=0, zorder=0)
    ax.fill_between(x, total, radial, color=BLUE_FILL, lw=0, zorder=0)
    ax.fill_between(x, total-sem, total+sem, color='.60', lw=0, zorder=2)
    radial_line, = ax.plot(x, radial, color=ORANGE, lw=3.5, zorder=4,
                           label=r'$\beta\omega^2\sum_{i=1}^{2}\langle r_i^2\rangle_k$')
    ax.plot(x, total, color='.15', marker='x', ms=markersize, lw=linewidth, zorder=4)
    return radial_line


def _epr_panel(plt, svg_utils, summary):
    """Panels (a-c), including the stationary EPR decomposition, from summaries."""
    from matplotlib.lines import Line2D

    indexed = {r['id']: r for r in summary['rows']}
    plateau = sorted((indexed[i] for i in summary['panels']['b']
                      if indexed[i]['k'] == summary['plateau_k']), key=lambda r: r['beta'])
    decomp = sorted((indexed[i] for i in summary['panels']['a']
                     if indexed[i]['number'] == 2 and np.isclose(indexed[i]['beta'], 10/3)),
                    key=lambda r: r['k'])
    if any('radial_reference' not in r or 'correction' not in r for r in plateau+decomp):
        raise ValueError('Missing decomposition summaries: validate the archived replicas with radial second moments.')
    fig = plt.figure(figsize=(WIDTH/72, TOP_HEIGHT/72), facecolor='white')
    # Two standalone plots, followed by one wide plot with a decomposition inset.
    a = fig.add_axes([70/WIDTH, (TOP_HEIGHT-259)/TOP_HEIGHT, 274/WIDTH, 225/TOP_HEIGHT])
    b = fig.add_axes([438/WIDTH, (TOP_HEIGHT-259)/TOP_HEIGHT, 274/WIDTH, 225/TOP_HEIGHT])
    c = fig.add_axes([68/WIDTH, 58/TOP_HEIGHT, 642/WIDTH, 289/TOP_HEIGHT])

    for beta, color in zip((10/3, 16., 25., 40., 60.), BLUES):
        rows = [indexed[i] for i in summary['panels']['a'] if indexed[i]['beta'] == beta]
        xs, ys = _coupling_points(a, rows, color)
        label = r'$\beta=10/3$' if np.isclose(beta, 10/3) else rf'$\beta={beta:g}$'
        a.annotate(label, xy=(25., np.interp(25., xs, ys)), xytext=(0, 2.5),
                   textcoords='offset points', color='black', fontsize=20,
                   ha='left', va='bottom', zorder=5)
    a.axhline(1., color='.5', ls='--', lw=.8, zorder=0)
    a.set(xlim=(-.5, 41), ylim=(.58, 1.035), xticks=[0, 20, 40], yticks=[.6, .8, 1.])
    a.set_xlabel(r'$k\;(\tau_r^{-1})$', fontsize=24, labelpad=2)
    a.set_ylabel(r'$\sigma\;(2\sigma_0)$', fontsize=24, labelpad=4)
    a.text(.96, .895, r'$\mathrm{(a)}$', transform=a.transAxes,
           ha='right', va='top', fontsize=25)

    for number, color in zip((2, 4, 8, 16), ('#9fc1d6', 'lightcoral', 'sandybrown', '#b69b3d')):
        rows = [indexed[i] for i in summary['panels']['c'] if indexed[i]['number'] == number]
        xs, ys = _coupling_points(b, rows, color)
        b.annotate(rf'$N={number}$', xy=(27., np.interp(27., xs, ys)), xytext=(0, 0 if number==16 else 2),
                   textcoords='offset points', color='black', fontsize=21,
                   ha='left', va='bottom', zorder=5)
    b.axhline(1., color='.5', ls='--', lw=.8, zorder=0)
    b.set(xlim=(-.5, 41), ylim=(.08, 1.035), xticks=[0, 20, 40], yticks=[.2, .6, 1.])
    b.set_xlabel(r'$k\;(\tau_r^{-1})$', fontsize=24, labelpad=2)
    b.set_ylabel(r'$\sigma\;(N\sigma_0)$', fontsize=24, labelpad=4)
    b.text(.5, .94, r'$\beta=10/3$', transform=b.transAxes,
           ha='center', va='top', fontsize=23)
    b.text(.96, .895, r'$\mathrm{(b)}$', transform=b.transAxes,
           ha='right', va='top', fontsize=25)
    for ax in (a, b):
        ax.tick_params(labelsize=21, length=5, width=1.3, pad=2)

    beta = np.array([r['beta'] for r in plateau])
    total = np.array([r['stratonovich_epr'] for r in plateau])
    sem = np.array([r['stratonovich_epr_sem'] for r in plateau])
    radial = np.array([r['radial_reference'] for r in plateau])
    radial_line = _shade_decomposition(c, beta, total, sem, radial)
    # The exact uncoupled pair EPR is unity in the plotted units at every beta.
    corrected = 1. - np.array([r['correction'] for r in plateau])
    corrected_error = np.array([r['correction_sem'] for r in plateau])
    c.fill_between(beta, corrected-corrected_error, corrected+corrected_error,
                   color=CORRECTION_COLOR, alpha=.18, lw=0, zorder=3)
    corrected_line, = c.plot(beta, corrected, '--', color=CORRECTION_COLOR, lw=3.3, zorder=5,
                             label=r'$1-\mathcal{R}/(2\sigma_0)$')
    c.set(xlim=(0, 100), ylim=(.54, 1.), xticks=[0, 20, 40, 60, 80, 100],
          yticks=[.6, .7, .8, .9, 1.])
    c.set_xlabel(r'$\beta$', fontsize=24, labelpad=2)
    c.set_ylabel(r'$\sigma\;(2\sigma_0)$', fontsize=24, labelpad=4)
    c.text(.018, .975, r'$\mathrm{(c)}$', transform=c.transAxes,
           ha='left', va='top', fontsize=25)
    c.text(.36, .85, r'$N=2,\quad\omega=0.2\;\tau_r^{-1}$',
           transform=c.transAxes, ha='center', va='top', fontsize=23)
    c.legend(handles=[
        Line2D([], [], marker='x', ls='none', color='black', ms=8, label=r'$\mathrm{Stratonovich}$'),
        radial_line, corrected_line],
        title=rf'$k={summary["plateau_k"]:g}$', title_fontsize=22,
        loc='lower left', bbox_to_anchor=(.235, .012), frameon=False,
        fontsize=21, handlelength=1.5, handletextpad=.35, labelspacing=.16)

    inset = c.inset_axes([.662, .155, .288, .405])
    k = np.array([r['k'] for r in decomp])
    work = np.array([r['stratonovich_epr'] for r in decomp])
    work_sem = np.array([r['stratonovich_epr_sem'] for r in decomp])
    ref = np.array([r['radial_reference'] for r in decomp])
    ref_sem = np.array([r['radial_reference_sem'] for r in decomp])
    stationary = np.array([r['observable_epr'] for r in decomp])
    stationary_sem = np.array([r['observable_epr_sem'] for r in decomp])
    correction = np.array([r['correction'] for r in decomp])
    correction_sem = np.array([r['correction_sem'] for r in decomp])
    if k.size == 0 or k[0] != 0:
        raise ValueError('The Fig. 7(c) inset requires the uncoupled simulation.')
    corrected_baseline = work[0] - correction
    # Each positive k uses a separate run. At k=0 the coupling correction is identically zero.
    corrected_error = np.hypot(work_sem[0], correction_sem)
    corrected_baseline[0] = work[0]
    corrected_error[0] = work_sem[0]
    for values, errors, color, linestyle, linewidth in [
            (ref, ref_sem, ORANGE, '-', 2.7),
            (stationary, stationary_sem, '#2879B5', '-', 1.8),
            (corrected_baseline, corrected_error, CORRECTION_COLOR, '--', 2.6)]:
        inset.fill_between(k, values-errors, values+errors, color=color,
                           alpha=.22, lw=0, zorder=2)
        inset.plot(k, values, color=color, ls=linestyle, lw=linewidth, zorder=3)
    inset.errorbar(k, work, yerr=work_sem, fmt='x', color='.15',
                   ms=8.5, mew=1.35, elinewidth=.65, capsize=0, zorder=4)
    inset.set_xscale('symlog', linthresh=.1, linscale=.45)
    inset.set(xlim=(0, 40), ylim=(.6, 1.))
    inset.set_yticks([.6, .8, 1.], labels=['0.6', '0.8', '1'])
    inset.set_xticks([0, .1, 1, 10, 40], labels=['0', '0.1', '1', '10', '40'])
    inset.set_xlabel(r'$k\;(\tau_r^{-1})$', fontsize=21, labelpad=1)
    inset.tick_params(labelsize=19, length=4.2, width=1.1, pad=2)
    inset.text(.5, 1.055, r'$\beta=10/3$', transform=inset.transAxes,
               ha='center', va='bottom', fontsize=20)
    top, _ = svg_utils._matplotlib_svg(fig, plt, 'five_panel_top_')

    return top


def plot_figure(output_dir,summary,frequency_summary):
    """Export the figure as PNG/SVG from saved results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend
    from PIL import Image
    import cairosvg

    helper_path=HERE.parent/'frequency/plotting.py'
    spec=importlib.util.spec_from_file_location('_figure7_dynamics_svg',helper_path)
    svg_utils=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(svg_utils)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    # Preserve the lower panels at their approved dimensions.
    width,top_height,gap=WIDTH,TOP_HEIGHT,-8.
    column_width=246.
    style=PUBLICATION_STYLE
    S=svg_utils.SVG;tag='{'+S+'}'

    def trace_coordinates():
     with np.load(HERE/'data/locking_trajectory_envelope.npz',allow_pickle=False) as data:
      traces=[(data[f't{i}'].copy(),data[f'y{i}'].copy()) for i in (1,2)]
      limits=tuple(data['y_limits'])
     return traces,limits

    def frequency_panel():
     summary8=frequency_summary
     f=svg_utils._frequency_panel(plt,summary8)
     f.set_layout_engine('none');f.set_size_inches(5.,304/72)
     a=f.axes[0];zoom=a.child_axes[0]
     a.set_position([61/360,49/304,280/360,247/304])
     a.set_xlabel(r'$k\;(\tau_r^{-1})$',fontsize=24,labelpad=3)
     a.set_ylabel(r'$\omega_{\mathrm{eff}}\;(\omega)$',fontsize=24,labelpad=3)
     a.set_xticks([0,50,100,150]);a.set_yticks([.4,.6,.8,1.])
     a.tick_params(labelsize=20,length=5,width=1.3,pad=3)
     for artist in list(a.get_children()):
      if isinstance(artist,Legend):artist.remove()
     for line in a.lines:
      if line.get_marker() not in (None,'None','',' ','_','|'):line.set_markersize(6.2)
      if line.get_linestyle()=='-':line.set_linewidth(1.65)
     for collection in a.collections:collection.set_linewidth(1.15)
     handles,labels=a.get_legend_handles_labels()
     reference=[(h,l) for h,l in zip(handles,labels) if l=='full Cartesian coupling']
     curves=[(h,l) for h,l in zip(handles,labels) if l!='full Cartesian coupling']
     curves[0]=(curves[0][0],r'$N\!=\!2,\!\beta\!=\!5,\!\omega\!=\!1$')
     assert len(reference)==1 and len(curves)==6
     curve_key=a.legend([h for h,_ in curves],[l for _,l in curves],
              loc='center left',bbox_to_anchor=(10.,.555),bbox_transform=a.transData,
              frameon=False,fontsize=14.5,handlelength=.85,handletextpad=.25,
              labelspacing=.02,borderaxespad=0.)
     a.add_artist(curve_key)
     a.legend([reference[0][0]],['full Cartesian coupling'],frameon=False,
              fontsize=15.5,loc='upper right',bbox_to_anchor=(150.,.995),
              bbox_transform=a.transData,handlelength=.8,handletextpad=.3,borderaxespad=0.)
     # Restore the zoom inside the main frequency plot, beside the curve key.
     zoom.set_axes_locator(None)
     zoom.set_position([221/360,86/304,117/360,66/304])
     zoom.set_xticks([0,1,2,3]);zoom.set_yticks([.75,.875,1.],labels=['0.75','0.875','1'])
     zoom.set_xlabel('')
     zoom.tick_params(labelsize=14.5,length=3.5,width=1.,pad=1.5)
     old=zoom.get_legend()
     if old is not None:old.remove()
     for line in zoom.lines:
      if line.get_marker() not in (None,'None','',' ','_','|'):line.set_markersize(5.3)
      if line.get_linestyle()=='-':line.set_linewidth(1.25)
     for collection in zoom.collections:collection.set_linewidth(1.)
     h,l=zoom.get_legend_handles_labels();assert len(h)==2
     zoom.legend(h,['sync. start','death start'],loc='lower left',
                 frameon=True,facecolor='white',edgecolor='none',framealpha=1.,
                 fontsize=12.5,borderpad=.1,labelspacing=.05,handlelength=.8,
                 handletextpad=.25,borderaxespad=.1)
     return svg_utils._matplotlib_svg(f,plt,'compact_frequency_')[0]

    with plt.rc_context(style):
     top=_epr_panel(plt,svg_utils,summary)
     frequency=frequency_panel()
     trace_data,trace_ylim=trace_coordinates()
     tf,ta=plt.subplots(figsize=(315/72,126/72),layout='constrained')
     for i,((times,values),color) in enumerate(zip(trace_data,('#ffc0cb','#5298c9'))):
      ta.plot(times,values,color=color,lw=1.05,label=rf'$y_{i+1}$')
     ta.set(xlabel=r'$t\;(\tau_r)$',ylabel=r'$y_i\;(r_c)$',xlim=(0,2300),ylim=trace_ylim)
     ta.set_xticks([0,1000,2000]);ta.set_yticks([-1,0,1])
     ta.tick_params(length=4.,width=1.2,pad=2.,labelsize=20)
     ta.set_xlabel(r'$t\;(\tau_r)$',fontsize=24,labelpad=2)
     ta.set_ylabel(r'$y_i\;(r_c)$',fontsize=24,labelpad=2)
     # The two trajectory colours identify the traces; omit their legend.
     trace,_=svg_utils._matplotlib_svg(tf,plt,'column_large_trace_')
     def label_svg(letter,prefix):
      f=plt.figure(figsize=(.55,.48),facecolor='none')
      f.text(.01,.97,rf'$\mathrm{{({letter})}}$',ha='left',va='top',fontsize=25)
      return svg_utils._matplotlib_svg(f,plt,prefix)[0]
     bottom_labels=[label_svg('d','five_panel_label_d_'),label_svg('e','five_panel_label_e_')]

    schematic,schematic_ratio=svg_utils._schematic()
    # Extract the exact trajectory drawing bounds from its exported vector panel.
    trace_rect=next(e for e in trace.iter() if e.get('id')=='column_large_trace_patch_2').find(tag+'path')
    trace_xy=np.array([float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',trace_rect.get('d'))]).reshape(-1,2)
    trace_left=11.+trace_xy[:,0].min();trace_right=11.+trace_xy[:,0].max()

    view=[float(x) for x in schematic.get('viewBox').split()];units=view[2]/310
    # Align the illustrative bars with visibly locked stretches of the saved trace.
    # These approximate display annotations are not a trajectory-state classifier.
    lookup={e.get('id'):e for e in schematic.iter() if e.get('id')}
    bar_ids=['schematic_path60666','schematic_path60666-3','schematic_path60666-7']
    text_ids=['schematic_g59131-1','schematic_g57694-7','schematic_g57694-3',
              'schematic_g72879','schematic_g72879-9']
    interval_times=np.array([[0.,165.],[190.,1880.],[1930.,2100.]])
    # Equal visual gaps centred on the same approximate transitions.
    transition_centres=(interval_times[:-1,1]+interval_times[1:,0])/2
    half_gap=12.5
    interval_times[:-1,1]=transition_centres-half_gap
    interval_times[1:,0]=transition_centres+half_gap
    assert np.allclose(interval_times[1:,0]-interval_times[:-1,1],2*half_gap)
    def bar_points(element):
     raw=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',element.get('d'))]
     assert len(raw)==4
     points=np.array([[raw[0],raw[1]],[raw[0]+raw[2],raw[1]+raw[3]]])
     matrix=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',element.get('transform'))]
     a,b,c,d,e,f=matrix
     return points@np.array([[a,b],[c,d]])+np.array([e,f])
    def rendered_label_bounds(identifier):
     # Measure actual outlined lettering, including SVG clip paths.
     clone=deepcopy(schematic)
     layer=next(e for e in clone.iter() if e.get('id')=='schematic_layer1')
     for element in list(layer):
      if element.get('id')!=identifier:layer.remove(element)
     x,y,w,h=240.,74.,145.,25.;pixels_per_unit=32
     clone.set('viewBox',f'{x} {y} {w} {h}');clone.set('width',str(w));clone.set('height',str(h))
     data=cairosvg.svg2png(bytestring=ET.tostring(clone),output_width=int(w*pixels_per_unit),
                          output_height=int(h*pixels_per_unit))
     bounds=Image.open(io.BytesIO(data)).getchannel('A').getbbox()
     assert bounds is not None,identifier
     return np.array([x+bounds[0]/pixels_per_unit,y+bounds[1]/pixels_per_unit,
                      x+bounds[2]/pixels_per_unit,y+bounds[3]/pixels_per_unit])
    label_bounds={identifier:rendered_label_bounds(identifier) for identifier in text_ids}
    bar_y=np.mean([bar_points(lookup[i])[:,1].mean() for i in bar_ids])
    target_global=trace_left+interval_times/2300.*(trace_right-trace_left)
    target_source=view[0]+(target_global-33.)*units
    for identifier,xs in zip(bar_ids,target_source):
     element=lookup[identifier]
     a,b,c,d,e,f=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',element.get('transform'))]
     desired=np.column_stack([xs,[bar_y,bar_y]])
     points=(desired-np.array([e,f]))@np.linalg.inv(np.array([[a,b],[c,d]]))
     delta=points[1]-points[0]
     element.set('d',f'm {points[0,0]},{points[0,1]} {delta[0]},{delta[1]}')
    parents={child:parent for parent in schematic.iter() for child in parent}
    left_anti=label_bounds['schematic_g57694-3'];left_lock=label_bounds['schematic_g72879-9']
    block_middle=(min(left_anti[1],left_lock[1])+max(left_anti[3],left_lock[3]))/2
    associations={'schematic_g57694-3':0,'schematic_g72879-9':0,
                  'schematic_g59131-1':1,'schematic_g57694-7':2,'schematic_g72879':2}
    label_alignment=[]
    for identifier in text_ids:
     element=lookup[identifier];bounds=label_bounds[identifier]
     interval=associations[identifier];centre=target_source[interval].mean()
     dx=centre-(bounds[0]+bounds[2])/2
     if identifier=='schematic_g57694-7':dy=left_anti[1]-bounds[1]
     elif identifier=='schematic_g72879':dy=left_lock[1]-bounds[1]
     elif identifier=='schematic_g59131-1':dy=block_middle-(bounds[1]+bounds[3])/2
     else:dy=0.
     parent=parents[element];position=list(parent).index(element)
     wrapper=ET.Element(tag+'g',id=identifier+'_centred',transform=f'translate({dx},{dy})')
     parent.remove(element);wrapper.append(element);parent.insert(position,wrapper)
     label_alignment.append(abs((bounds[0]+bounds[2])/2+dx-centre)/units)
    actual_bar_global=np.array([33.+(bar_points(lookup[i])[:,0]-view[0])/units for i in bar_ids])
    bar_alignment_error_pt=float(np.max(abs(actual_bar_global-target_global)))
    assert bar_alignment_error_pt<1e-9
    assert max(label_alignment)<1e-9
    schematic.set('overflow','visible')
    schematic_y=22.;schematic_height=310.*schematic_ratio
    trace_y=max(186.,schematic_y+schematic_height-2.)
    lower_height=max(322.,trace_y+126.+10.);height=top_height+gap+lower_height
    root=ET.Element(tag+'svg',width=f'{column_width}pt',height=f'{column_width*height/width}pt',
                     viewBox=f'0 0 {width} {height}',version='1.1')
    ET.SubElement(root,tag+'rect',width=str(width),height=str(height),fill='white')
    for panel,x,y,w,h in [(top,0.,0.,width,top_height),
                          (schematic,33.,top_height+gap+schematic_y,310.,schematic_height),
                          (trace,11.,top_height+gap+trace_y,315.,126.),
                          (frequency,364.,top_height+gap+8.,360.,304.),
                          (bottom_labels[0],2.,top_height+gap+3.,39.6,34.56),
                          (bottom_labels[1],355.,top_height+gap+3.,39.6,34.56)]:
     panel.set('x',str(x));panel.set('y',str(y));panel.set('width',str(w));panel.set('height',str(h))
     root.append(panel)
    svg=out/'figure7.svg';png=svg.with_suffix('.png')
    ET.ElementTree(root).write(svg,encoding='utf-8',xml_declaration=True)
    cairosvg.svg2png(url=str(svg),write_to=str(png),output_width=2400,
                     output_height=round(2400*height/width))
    return svg,png
