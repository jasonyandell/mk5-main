# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Render a phone-readable campaign chart. Optional dependency: matplotlib."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def render(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter, FuncFormatter

    root = Path(root).resolve()
    state = json.loads((root / "state.json").read_text())
    baseline = json.loads((root / "baseline/report.json").read_text())
    rows = [{"generation":0,"layers":2,"parameters":107392,
             "win_rate":baseline["team_win_rate"],"checkpoint_sha256":baseline["checkpoint_sha256"]}]
    for generation in state["generations"][1:]:
        run = root / generation["id"]
        result = json.loads((run / "result.json").read_text())
        provenance = json.loads((run / "release/provenance.json").read_text())
        publication = json.loads((run / "published.json").read_text())
        rows.append({"generation":int(generation["id"][1:]),"layers":generation["layers"],
                     "parameters":provenance["parameters"],"win_rate":result["development"]["team_win_rate"],
                     "checkpoint_sha256":generation["sha256"],"model_revision":publication["model_revision"]})
    x, wins, parameters = ([r[key] for r in rows] for key in ("generation","win_rate","parameters"))
    retained = [max(wins[:i+1]) for i in range(len(wins))]
    fig, (top, bottom) = plt.subplots(2,1,figsize=(6.8,7.2),sharex=True,
                                     gridspec_kw={"height_ratios":[3,1.5]})
    fig.subplots_adjust(left=.15,right=.97,bottom=.19,top=.85,hspace=.15)
    fig.text(.15,.955,"Gus: growing a tiny player",fontsize=18,weight="bold",color="#163a43")
    fig.text(.15,.91,"Same 256-hand G0/Zeb evaluation table",fontsize=11,color="#42545a")
    for axis in (top,bottom):
        axis.spines[["top","right"]].set_visible(False)
        axis.spines[["bottom","left"]].set_color("#bdc8cb")
        axis.tick_params(colors="#42545a",labelsize=10)
        axis.grid(axis="y",color="#e8edee",linewidth=.8)
        axis.set_axisbelow(True)
    top.axhline(wins[0],color="#788487",linestyle=":",linewidth=1.5,label=f"G0 reference: {wins[0]:.2%}")
    top.plot(x,wins,color="#19788d",marker="o",markersize=5,linewidth=1.6,label="Each completed generation")
    top.step(x,retained,where="post",color="#cf762d",linewidth=1.8,linestyle="--",label="Best retained score")
    top.set_ylim(min(.45,min(wins)-.025),max(.60,max(wins)+.035))
    top.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
    top.set_ylabel("Development team win rate",fontsize=11,color="#163a43")
    top.legend(loc="upper left",frameon=False,fontsize=9)
    bottom.step(x,parameters,where="post",color="#19788d",linewidth=2)
    bottom.scatter(x,parameters,color="#19788d",s=15,zorder=3)
    bottom.axhline(757184,color="#788487",linestyle=":",linewidth=1)
    bottom.text(max(x),775000,"Target: 757k",ha="right",va="bottom",fontsize=9,color="#596b70")
    for i,row in enumerate(rows):
        if i == 0 or row["layers"] != rows[i-1]["layers"]:
            if row["parameters"] < 757184:  # The target line already labels 757k.
                bottom.annotate(f"{row['parameters']/1000:.0f}k",(x[i],parameters[i]),xytext=(3,8),
                                textcoords="offset points",fontsize=9,color="#163a43")
            if i:
                top.axvline(x[i],color="#dce4e6",linewidth=.8,zorder=0)
    bottom.set_ylim(0,850000)
    bottom.yaxis.set_major_locator(MaxNLocator(4))
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f"{v/1000:.0f}k"))
    bottom.xaxis.set_major_locator(MaxNLocator(8,integer=True))
    bottom.set_xlim(-.35,max(1,max(x))+.5)
    bottom.set_xlabel("Generation",fontsize=11,color="#163a43")
    bottom.set_ylabel("Parameters",fontsize=11,color="#163a43")
    data = {"scope":"Repeated development panel; one learner seat; legal argmax; bid30 pip",
            "completed_generations":len(rows)-1,"rows":rows,
            "plot_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    note = "Development scores are reused for selection.\nThey are not independent evidence of improvement.\nFresh confirmation has not run yet."
    if (root / "final.json").exists():
        final = json.loads((root / "final.json").read_text())
        data["confirmation"] = final
        chosen, base = final["confirmation"], final["baseline"]
        delta=chosen["vs_reference"];lo,hi=delta["ci95"]
        note=(f"Development scores above were reused for selection.\nFresh {chosen['hands']} hands: selected {chosen['team_win_rate']:.2%}, G0 {base['team_win_rate']:.2%}.\n"
              f"Difference {100*delta['win_rate_difference']:+.2f} pp; 95% paired CI [{100*lo:+.2f}, {100*hi:+.2f}].")
    fig.text(.15,.045,note,fontsize=9,color="#42545a",linespacing=1.6)
    for extension in ("png","pdf"):
        fig.savefig(root/f"progress.{extension}",dpi=180,facecolor="white")
    plt.close(fig)
    (root/"progress.json").write_text(json.dumps(data,indent=2)+"\n")
    return {"generations":len(rows)-1,"png":str(root/"progress.png")}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    args=parser.parse_args()
    print(json.dumps(render(args.root)),flush=True)
