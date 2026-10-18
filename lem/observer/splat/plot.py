# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Phone-readable full and late-window learning curves; optional matplotlib."""
import argparse
import json
from pathlib import Path


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    root=Path(root)
    rows=[]
    for line in (root/'training.log').read_text().splitlines():
        if line.startswith('{"step":'):
            rows.append(json.loads(line))
    if not rows:raise ValueError('no saved training evaluations')
    initial=json.loads((root/'training/initial.json').read_text())
    x=[0]+[r['step']/1000 for r in rows]
    v=[initial]+[r['validation'] for r in rows]
    fig,axes=plt.subplots(3,1,figsize=(6.5,9.2))
    fig.subplots_adjust(left=.17,right=.96,top=.87,bottom=.19,hspace=.62)
    fig.text(.17,.965,'Gus learns consequences',fontsize=19,weight='bold',color='#163a43')
    fig.text(.17,.925,f"500k deals · 185,652 parameters · {rows[-1]['step']:,} updates",fontsize=10,color='#42545a')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#e8edee');ax.set_axisbelow(True)
        ax.tick_params(colors='#42545a',labelsize=9)
        ax.set_xlabel('Training updates (thousands)',fontsize=10)
    axes[0].plot(x,[a['tile_all_top1'] for a in v],color='#19788d',linewidth=2)
    axes[0].set_title('Predicting the recorded tile',loc='left',fontsize=12)
    axes[0].set_ylabel('Accuracy');axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylim(0,.65)
    axes[0].annotate(f"{v[-1]['tile_all_top1']:.1%}",(x[-1],v[-1]['tile_all_top1']),xytext=(-4,9),textcoords='offset points',ha='right',color='#19788d',weight='bold')
    axes[1].plot(x,[100*a['splat_consequential_make_mae'] for a in v],color='#cf762d',linewidth=2)
    axes[1].set_title('Forecasting make/set before settlement',loc='left',fontsize=12)
    axes[1].set_ylabel('Mean error (pp)');axes[1].set_ylim(0,30)
    axes[1].annotate(f"{100*v[-1]['splat_consequential_make_mae']:.1f} pp",(x[-1],100*v[-1]['splat_consequential_make_mae']),xytext=(-4,9),textcoords='offset points',ha='right',color='#a95318',weight='bold')
    late=[r for r in rows if r['step']>=20000] or rows
    axes[2].plot([r['step']/1000 for r in late],[r['validation']['splat_all_kl'] for r in late],color='#7854a3',linewidth=2)
    axes[2].set_title('Late window: outcome-distribution error',loc='left',fontsize=12)
    axes[2].set_ylabel('KL (nats)')
    fig.text(.17,.045,'Validation on 50,210 whole held-out deals.\nForecast errors compare with the named Walt teachers.\nThese curves are not game win rates.',fontsize=9,color='#42545a',linespacing=1.5)
    for ext in ('png','pdf'):fig.savefig(root/f'progress.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    return str(root/'progress.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    print(render(p.parse_args().root))
