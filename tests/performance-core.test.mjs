import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {performance,normalizeBet,roi} from '../web/assets/js/performance-core.mjs';

const bet=(extra={})=>({Date:'01/10/2026',Player:'A',Tournament:'Open','Tour/Majors':'PGA',Stake:'£10',Return:'£30','Profit/Loss':'£18',Outcome:'Won',...extra});
test('accounting modes retain recorded P/L versus calculated ROI and pending treatment',()=>{
 const rows=[bet(),bet({Outcome:'Pending',Return:'','Profit/Loss':''})];
 const settled=performance(rows).total,all=performance(rows,{mode:'all'}).total;
 assert.equal(settled.bets,1);assert.equal(settled.pl,18);assert.equal(roi(settled),180);
 assert.equal(all.bets,2);assert.equal(all.pl,10);assert.equal(roi(all),50);
});
test('settled fallback, zero stakes and undated rows',()=>{
 assert.equal(normalizeBet(bet({'Profit/Loss':''}),'settled').pl,20);
 assert.equal(normalizeBet(bet({Date:''}),'settled'),null);
 const result=performance([bet({Date:'',Stake:'0'})],{mode:'all'});
 assert.equal(result.undated,1);assert.equal(result.total.bets,1);assert.equal(result.groups.length,0);
 assert.equal(roi({stake:0,pl:10}),null);
});
test('year/tour grouping, shared filters and chronological cumulative P/L',()=>{
 const rows=[bet(),bet({Date:'02/10/2026','Tour/Majors':'DPWT'}),bet({Date:'01/10/2025'})];
 const all=performance(rows,{view:'tournament'});
 assert.equal(all.groups.length,3);assert.deepEqual(all.groups.map(g=>g.running),[18,36,54]);
 assert.equal(performance(rows,{year:'2026',tour:'PGA'}).total.bets,1);
 for(const view of ['yearly','monthly','weekly','player','tournament'])assert.equal(performance(rows,{view}).total.pl,54);
});
test('complete CSV matches predecessor accounting totals and tournament groups',async()=>{
 const csv=readFileSync(new URL('../web/spreadsheet_data.csv',import.meta.url),'utf8');
 const context={window:{},fetch:async()=>({ok:true,headers:{get:()=>null},text:async()=>csv}),localStorage:{getItem:()=>null,setItem:()=>{}},console};
 vm.runInNewContext(readFileSync(new URL('../web/assets/js/betting-data.js',import.meta.url),'utf8'),context);
 const rows=await context.window.BettingData.load({forceRefresh:true});
 const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-7,`${a} != ${b}`);
 const legacyNumber=v=>parseFloat(String(v||'').replace('£','').replace(',',''));
 const stake=rows.reduce((n,r)=>n+(legacyNumber(r.Stake)||0),0),returned=rows.reduce((n,r)=>n+(legacyNumber(r.Return)||0),0);
 const all=performance(rows,{mode:'all'}).total;
 near(all.stake,stake);near(all.returned,returned);near(all.pl,returned-stake);assert.equal(all.bets,rows.length);
 const num=v=>{const n=parseFloat(String(v||'').replace(/[£,\s]/g,'').replace(/^\((.*)\)$/,'-$1'));return Number.isFinite(n)?n:NaN;};
 const old=new Map();
 for(const r of rows){
  const parts=String(r.Date||'').split('/').map(Number),d=parts.length===3&&parts.every(Number.isFinite)?new Date(parts[2],parts[1]-1,parts[0]):null;
  const s=num(r.Stake),ret=num(r.Return),p=num(r['Profit/Loss']),pl=Number.isFinite(p)?p:Number.isFinite(s)||Number.isFinite(ret)?(Number.isFinite(ret)?ret:0)-(Number.isFinite(s)?s:0):NaN;
  if(!d||!Number.isFinite(pl)||String(r.Outcome||'').toLowerCase().includes('pending'))continue;
  const key=JSON.stringify([String(d.getFullYear()),String(r['Tour/Majors']||'').trim()||'Other',String(r.Tournament||'').trim()||'Unspecified tournament']);
  if(!old.has(key))old.set(key,{bets:0,stake:0,returned:0,pl:0});
  const g=old.get(key);g.bets++;g.stake+=Math.max(0,s||0);g.returned+=Math.max(0,ret||0);g.pl+=pl;
 }
 const current=performance(rows,{view:'tournament'}).groups;
 assert.equal(current.length,old.size);
 for(const g of current){const previous=old.get(JSON.stringify([g.year,g.tour,g.name]));assert.ok(previous);for(const key of ['bets','stake','returned','pl'])near(g[key],previous[key]);}
});
