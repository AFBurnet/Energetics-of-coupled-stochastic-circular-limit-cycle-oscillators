"""Effective frequency panels and SVG composition helpers for Figure 7."""
from copy import deepcopy
import io
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
SVG = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', XLINK)
SCHEMATIC = HERE / 'assets/schematic_a.svg'


def _plot_points(ax, points, *, color, marker='o', label=None, line=True,
                 open_marker=False, markersize=4.6):
    points = sorted(points, key=lambda p: p['k'])
    if not points:
        return
    ax.errorbar([p['k'] for p in points], [p['mean'] for p in points],
        yerr=[p['sem'] for p in points], color=color,
        fmt=marker+('-' if line else ''), lw=1.35, ms=markersize, capsize=1.6,
        markerfacecolor='white' if open_marker else color, label=label)


def _frequency_panel(plt, summary):
    fig, ax = plt.subplots(figsize=(5., 3.9), layout='constrained')
    # Colors of the reference and comparison curves.
    colors = ('#ffc0cb', '#64a3cf', '#9ac5e0', '#5298c9', '#88b9da', '#408cc3')
    baseline = []
    broad_points = []
    for curve, color, marker in zip(summary['curves'], colors, ('o', 'D', '^', 'x', 'P', 's')):
        parameters = curve['parameters']
        points = curve['points']
        visible = [p for p in points if p['initial'] != 'death']
        broad_points.extend(visible)
        _plot_points(ax, visible, color=color, marker=marker, label=curve['label'])
        if (parameters['number'] == 2 and np.isclose(parameters['omega'], 1.)
                and np.isclose(parameters['beta'], 5.)):
            baseline.extend(points)
    ax.axhline(1., ls='--', lw=.8, color='.5', zorder=0)
    reference_k = np.linspace(0., 150., 20)
    reference, = ax.plot(reference_k, np.ones_like(reference_k), marker='v', ls='none',
                         color='silver', ms=4.6, zorder=1, label='full Cartesian coupling')
    ax.set(xlabel=r'$k\;(\tau_r^{-1})$', ylabel=r'$\omega_{\mathrm{eff}}\;(\omega)$',
           xlim=(-2., 152.), ylim=(.40, 1.04))
    if broad_points:
        lower = min(p['mean']-p['sem'] for p in broad_points)
        upper = max(p['mean']+p['sem'] for p in broad_points)
        ax.set_ylim(min(.40, lower-.02), max(1.04, upper+.02))
    handles, labels = ax.get_legend_handles_labels()
    curves = [(handle, label) for handle, label in zip(handles, labels) if handle is not reference]
    curve_handles, curve_labels = zip(*curves)
    if ax.get_ylim()[1]-ax.get_ylim()[0] < .9:
        # Put both annotations in the gap between the N=8 and omega=0.1 curves.
        curve_legend = ax.legend(curve_handles, curve_labels, frameon=False,
                                fontsize=7.4, loc='center left',
                                bbox_to_anchor=(26., .555), bbox_transform=ax.transData,
                                ncol=1, handlelength=1.5, borderaxespad=0.)
        zoom = ax.inset_axes([86., .49, 62., .14], transform=ax.transData)
    else:
        # Short test runs can have wide intervals: keep their inset legible.
        curve_legend = ax.legend(curve_handles, curve_labels, frameon=True,
                                facecolor='white', edgecolor='none', framealpha=1.,
                                fontsize=7.4, loc='upper right', ncol=2, handlelength=1.5)
        zoom = ax.inset_axes([.565, .105, .40, .26])
    ax.add_artist(curve_legend)
    ax.legend([reference], ['full Cartesian coupling'], frameon=False,
              fontsize=8., loc='upper right', bbox_to_anchor=(150., .995),
              bbox_transform=ax.transData, handlelength=1.2, borderaxespad=0.)
    sync = [p for p in baseline if p['initial'] != 'death' and p['k'] <= 3.2]
    death = [p for p in baseline if p['initial'] == 'death' and p['k'] <= 3.2]
    _plot_points(zoom, sync, color=colors[0], marker='o',
                 label='synchronized start', markersize=3.8)
    _plot_points(zoom, death, color='#5298c9', marker='s',
                 label='death-state start', line=False, open_marker=True, markersize=3.8)
    # For omega=1, the antipodal branch has positive radius only for k>1;
    # it is initially unstable and becomes stable above the separate threshold.
    zoom.axvline(1., color='#9b7333', ls='--', lw=.9)
    threshold = float(summary['threshold'])
    zoom.axvline(threshold, color='.35', ls=':', lw=1.)
    zoom.axhline(1., color='.6', ls='--', lw=.6, zorder=0)
    zoom.set_xlim(-.05, 3.15)
    zoom.set_xlabel(r'$k\;(\tau_r^{-1})$', fontsize=8, labelpad=1.)
    zoom.tick_params(labelsize=7)
    zoom.set_ylim(.75, 1.)
    zoom.set_yticks((.75, .875, 1.), labels=('0.75', '0.875', '1'))
    zoom.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1.,
                fontsize=6.4, loc='lower left',
                bbox_to_anchor=(.02, .02), borderpad=.1, labelspacing=.15,
                handlelength=1.2, borderaxespad=0.)
    return fig


def _prefix_ids(root, prefix):
    """Prevent collisions between independent SVG defs, paths and clip paths."""
    mapping = {e.attrib['id']: prefix+e.attrib['id'] for e in root.iter() if 'id' in e.attrib}
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if key == 'id':
                element.set(key, mapping[value])
            elif value.startswith('#') and value[1:] in mapping:
                element.set(key, '#'+mapping[value[1:]])
            else:
                element.set(key, re.sub(r'url\(#([^)]+)\)',
                    lambda m: 'url(#'+mapping.get(m.group(1), m.group(1))+')', value))
    return root


def _portable_markers(root):
    """Resolve SVG2 context colors explicitly for renderers lacking their support."""
    markers = {e.get('id'): e for e in root.iter() if e.tag == f'{{{SVG}}}marker'}
    defs = ET.Element(f'{{{SVG}}}defs')
    for element in list(root.iter()):
        style = dict(item.split(':', 1) for item in element.get('style', '').split(';') if ':' in item)
        for key in ('marker-start', 'marker-mid', 'marker-end'):
            value = element.get(key, style.get(key, ''))
            match = re.fullmatch(r'url\(#([^)]+)\)', value)
            if not match or match.group(1) not in markers:
                continue
            marker = markers[match.group(1)]
            if 'context-' not in ET.tostring(marker, encoding='unicode'):
                continue
            clone = deepcopy(marker)
            identifier = marker.get('id')+f'_resolved_{len(defs)}'
            _prefix_ids(clone, f'resolved{len(defs)}_')
            clone.set('id', identifier)
            stroke = element.get('stroke', style.get('stroke', '#000000'))
            fill = element.get('fill', style.get('fill', '#000000'))
            for part in clone.iter():
                for attribute, text in list(part.attrib.items()):
                    part.set(attribute, text.replace('context-stroke', stroke).replace('context-fill', fill))
            defs.append(clone)
            if key in element.attrib:
                element.set(key, f'url(#{identifier})')
            else:
                style[key] = f'url(#{identifier})'
                element.set('style', ';'.join(f'{k}:{v}' for k, v in style.items()))
    root.insert(0, defs)


def _schematic():
    root = ET.parse(SCHEMATIC).getroot()
    # The white rectangle sets the placement of the schematic above the traces.
    canvas = next((e for e in root.iter() if e.get('id') == 'rect14636'), None)
    if canvas is not None:
        viewbox = [float(canvas.get(key)) for key in ('x', 'y', 'width', 'height')]
    else:
        viewbox = [float(v) for v in root.get('viewBox', '').split()]
    if len(viewbox) != 4 or min(viewbox[2:]) <= 0:
        raise ValueError('The schematic needs a valid viewBox or its original canvas rectangle')
    root.set('viewBox', ' '.join(map(str, viewbox)))
    _portable_markers(root)
    return _prefix_ids(root, 'schematic_'), viewbox[3]/viewbox[2]


def _matplotlib_svg(fig, plt, prefix):
    stream = io.BytesIO()
    fig.savefig(stream, format='svg', facecolor='white')
    plt.close(fig)
    root = _prefix_ids(ET.fromstring(stream.getvalue()), prefix)
    # Scope Matplotlib's global stroke defaults to this panel alone.
    identifier = prefix+'panel'
    root.set('id', identifier)
    for element in root.iter(f'{{{SVG}}}style'):
        if element.text:
            element.text = element.text.replace('* {', f'#{identifier} * {{')
    viewbox = [float(v) for v in root.get('viewBox').split()]
    return root, viewbox[3]/viewbox[2]
