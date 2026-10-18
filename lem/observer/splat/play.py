# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Choose by forecast make/set tail, with the same clean prefix and legal mask."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from forge.zeb.game import current_player, legal_actions
from ..data import VERSION
from ..model import Config
from ..play import prefix
from .model import SplatObserver, TASK


class SplatPlay:
    def __init__(self, checkpoint, device='cpu', head='splat'):
        if head not in ('splat','tile'):raise ValueError('unknown policy head')
        path=Path(checkpoint)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['task']!=TASK or saved['tokenizer']!=VERSION:
            raise ValueError('checkpoint task or tokenizer mismatch')
        self.model=SplatObserver(Config(**saved['config'])).to(device).eval()
        self.model.load_state_dict(saved['model'])
        self.device,self.head=device,head
        self.identity={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'task':TASK,'head':head,'step':saved['step'],'data_sha256':saved['data_sha256'],
            'selection':'legal argmax of team make/set tail; lowest tile ID breaks exact ties' if head=='splat'
            else 'legal next-tile argmax; lowest tile ID breaks exact ties'}

    def plays(self,decl):return decl in (*range(8),9)

    @torch.inference_mode()
    def choose(self,states,bid_values,marks=None,marks_to_win=7):
        if len(states)!=len(bid_values):raise ValueError('one bid per state')
        if not states:return []
        rows=[prefix(s,bid) for s,bid in zip(states,bid_values)]
        x=np.zeros((len(rows),max(map(len,rows))),np.int64)
        for i,r in enumerate(rows):x[i,:len(r)]=r
        positions=torch.tensor([[len(r)-1] for r in rows],device=self.device)
        tile,splat=self.model(torch.tensor(x,device=self.device),positions)
        if self.head=='splat':
            scores=splat[:,0].softmax(-1)[:,:,30:].sum(-1).cpu().numpy()
            for i,s in enumerate(states):
                if current_player(s)%2!=s.bidder%2:scores[i]=1-scores[i]
        else:
            scores=tile[torch.arange(len(rows),device=self.device),positions[:,0]].cpu().numpy()
        if not np.isfinite(scores).all():raise ValueError('nonfinite scores')
        result=[]
        for s,row in zip(states,scores):
            hand=s.hands[current_player(s)]
            legal=sorted(legal_actions(s),key=lambda slot:hand[slot])
            result.append(max(legal,key=lambda slot:row[hand[slot]]))
        return result
