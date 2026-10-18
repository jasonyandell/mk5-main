# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Execute the preregistered eight-pair plan once, with a 60s cap per faceoff."""
import argparse
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np


def run(root):
    root=Path(root)
    assert (root/'training/report.json').exists(),'Training has not completed'
    plan=json.load(open(root/'h2h-plan.json'))
    if (len(plan['pairs'])!=8 or plan['outer_cap_seconds']!=60
            or len({p['seed'] for p in plan['pairs']})!=8
            or any(sum(p['opponent']==name for p in plan['pairs'])!=4 for name in ('g0','zeb'))):
     raise ValueError('expected the registered four pairs per opponent and 60-second cap')
    out=root/'h2h';out.mkdir(exist_ok=False)
    attempts=[]
    for p in plan['pairs']:
     name=f"{p['opponent']}-pair-{p['pair']}";started=time.monotonic()
     cmd=['scripts/cpu-slot','--threads','2','--timeout','60','--',sys.executable,'-u','-m','lem.observer.splat.h2h',
     '--candidate',str(root/'training/best.pt'),'--opponent',plan['opponents'][p['opponent']],
     '--exclude-hands',str(root/'excluded-hands.npy'),'--seed',str(p['seed']),'--out',str(out/name)]
     with (out/f'{name}.log').open('w') as log:
      result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'MK5_CPU_SLOT_COST':str(out/f'{name}-cost.jsonl')})
     row={**p,'exit_code':result.returncode,'outer_seconds_including_queue':time.monotonic()-started,'command':cmd}
     attempts.append(row);(out/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
     print(name,result.returncode,flush=True)
    summary={'scope':'preregistered small pilot; four pairs per opponent; bootstrap intervals exploratory and suppressed if empirically degenerate; conservative bounded-pair Hoeffding intervals also reported; not a promotion test','scheduled_pairs':len(plan['pairs']),'completed_pairs':0,'opponents':{}}
    for name in ('g0','zeb'):
     reports=[]
     for p in plan['pairs']:
      path=out/f"{p['opponent']}-pair-{p['pair']}"/'report.json'
      if p['opponent']==name and path.exists():reports.append(json.load(open(path)))
     summary['completed_pairs']+=len(reports)
     values=np.array([r['game_win_rate']['estimate'] for r in reports]);marks=np.array([r['marks_per_game']['estimate'] for r in reports])
     item={'pairs':len(reports),'games':2*len(reports),'wins':int(values.sum()*2),'hands':sum(r['hands'] for r in reports),'incomplete_pairs':4-len(reports)}
     if len(reports)>1:
      idx=np.random.default_rng(42).integers(len(reports),size=(10000,len(reports)))
      radius=float(np.sqrt(np.log(40)/(2*len(values))))
      item.update(win_rate=float(values.mean()),
         conservative_pair_hoeffding_ci95=[max(0.,float(values.mean())-radius),min(1.,float(values.mean())+radius)],
         win_rate_ci95=np.quantile(values[idx].mean(-1),[.025,.975]).tolist() if len(np.unique(values))>1 else None,
         marks_per_game=float(marks.mean()),marks_ci95=np.quantile(marks[idx].mean(-1),[.025,.975]).tolist())
     summary['opponents'][name]=item
    summary['all_completed_games_replayed']=all(json.load(open(p))['audit']['all_plays_legal_and_scores_replayed'] for p in out.glob('*-pair-*/report.json'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    run(parser.parse_args().root)
