// SPDX-FileCopyrightText: Jason Yandell
// SPDX-License-Identifier: CC0-1.0
// Shipped Zeb bidding JS (0.4.0 or 0.4.1) with the identical Rust player compiled natively.
// Differs from the 0.4.0 trials host only in reading legalDeclarations and overcall from the participant config.
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {pathToFileURL} from 'node:url';
import {join} from 'node:path';
const [kind,dir,configText]=process.argv.slice(2);
if(kind!=='zeb')throw Error('only Zeb');
const config=JSON.parse(configText??'{}');
const z=await import(pathToFileURL(join(dir,'package/index.js')));
const child=spawn(config.nativeBinary,[],{stdio:['pipe','pipe','inherit']});
const exited=new Promise((resolve,reject)=>{child.on('error',reject);child.on('close',code=>code===0?resolve():reject(Error(`native Zeb exit ${code}`)));});
const output=createInterface({input:child.stdout})[Symbol.asyncIterator]();
async function ask(r){
  child.stdin.write(JSON.stringify(r)+'\n');
  const line=await output.next(); if(line.done)throw Error('native Zeb ended');
  const reply=JSON.parse(line.value);if(reply.error)throw Error(reply.error);return reply;
}
const worlds=config.worlds??160, cutoff=config.cutoff??.8;
const declarations=config.legalDeclarations??(config.matchAware?[0,1,2,3,4,5,6]:[0,1,2,3,4,5,6,7,9]);
const forecasts=new Map();
async function forecastOf(hand){
  const key=[...hand].sort((a,b)=>a-b).join(',');
  if(!forecasts.has(key)){
    if(forecasts.size>64)forecasts.clear();
    forecasts.set(key,await ask({op:'forecast',hand:[...hand],worlds,seed:0,declarations}));
  }
  return forecasts.get(key);
}
async function call(r){
  if(r.op==='version')return ask({op:'version'});
  if(r.op==='bid'){
    const partner=r.high_seat>=0 && r.high_seat%2===r.seat%2;
    if(!config.matchAware && partner)return {bid:0};
    const need=config.matchAware?Math.min(...r.legal):r.legal.find(v=>v<=42);
    if(need===undefined)return {bid:0};
    const team=r.seat%2;
    const match=!config.matchAware?undefined:{
      us:r.marks[team],them:r.marks[1-team],marksToWin:r.marks_to_win,
      weShakeNext:(r.dealer+1)%2===team,
      holder:r.high_seat<0?'none':partner?'partner':'opp',
      passesAfter:r.bids.filter(b=>b===-1).length-1,
      maxBid:Math.max(...r.legal),pointsToWin:r.points_to_win!==null,
    };
    const choice=z.chooseBid(await forecastOf(r.hand),{need,cutoff,match,legalDeclarations:declarations,...(config.overcall?{overcall:true}:{})});
    return {bid:choice && r.legal.includes(choice.bid)?choice.bid:0};
  }
  if(r.op==='declare'){
    const t=Math.min(r.bid,42)-30;
    let best=null;
    for(const f of await forecastOf(r.hand))if(!best || f.pmake[t]>best.pmake[t])best=f;
    return {decl:best.decl};
  }
  if(r.op==='play'){
    const out=await ask({op:'play',decl:r.decl,bidder:r.bidder,bid:Math.min(r.bid,42),seat:r.seat,
      hand:[...r.hand],plays:r.plays,worlds,seed:0,reading:true});
    return {tile:out.tile};
  }
  throw Error(`unknown op ${r.op}`);
}
try{
  for await(const line of createInterface({input:process.stdin})){
    try{console.log(JSON.stringify(await call(JSON.parse(line))));}
    catch(e){console.log(JSON.stringify({error:String(e)}));}
  }
}finally{child.stdin.end();await exited;}
