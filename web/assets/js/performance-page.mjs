import {performance,normalizeBet,roi} from './performance-core.mjs';
const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money=n=>new Intl.NumberFormat('en-GB',{style:'currency',currency:'GBP'}).format(n),tone=n=>n>0?'positive':n<0?'negative':'';
const params=new URLSearchParams(location.search);let view=['overview','period','player','tournament'].includes(params.get('view'))?params.get('view'):'overview',rows=[],data,sort='latest',direction=1;
function setView(){document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===view)));$('frequency').hidden=view!=='period';}
function renderTable(){
 if(!data)return;
 const q=$('search').value.trim().toLowerCase(),visible=data.groups.filter(g=>`${g.name} ${view==='tournament'?g.tour:''}`.toLowerCase().includes(q));
 visible.sort((a,b)=>{const value=g=>sort==='roi'?roi(g):g[sort];const x=value(a),y=value(b);if(x==null||y==null)return x==null?(y==null?0:1):-1;return direction*(typeof x==='string'?x.localeCompare(y,undefined,{numeric:true}):x-y);});
 const title=view==='tournament'?'Tournament':view==='player'?'Player':view==='period'?$('period').selectedOptions[0].textContent:'Year';
 const columns=[['name',title],['bets','Bets'],['stake','Stake'],['returned','Return'],['pl','P/L'],['roi','ROI'],...(view==='tournament'?[['running','Running P/L'],['latest','Latest bet']]:[])];
 $('status').textContent=`${visible.length} of ${data.groups.length} groups shown${data.undated?` · ${data.undated} undated rows in totals`:''}`;
 $('results').innerHTML=`<table><caption>${esc(title)} breakdown</caption><thead><tr>${columns.map(([key,label])=>`<th scope="col" aria-sort="${sort===key?(direction===1?'ascending':'descending'):'none'}"><button data-sort="${key}">${label}${sort===key?(direction===1?' ↑':' ↓'):''}</button></th>`).join('')}</tr></thead><tbody>${visible.map(g=>`<tr>${columns.map(([key])=>{
 if(key==='name'){let label=esc(g.name);if(view==='player'||view==='tournament'){const p=new URLSearchParams({[view]:g.name});if(view==='tournament'){if(g.year!=='Undated')p.set('year',g.year);p.set('tour',g.tour);}else{if($('year').value!=='all')p.set('year',$('year').value);if($('tour').value!=='all')p.set('tour',$('tour').value);}label=`<a href="spreadsheet.html?${esc(p.toString())}">${label}</a>`;}return `<td>${label}${view==='tournament'?`<small>${esc(g.tour)} · ${esc(g.year)}</small>`:''}</td>`;}
 const n=key==='roi'?roi(g):g[key];return `<td class="${['pl','roi','running'].includes(key)?tone(n):''}">${key==='latest'?(n?n.toLocaleDateString('en-GB'):'Undated'):key==='roi'?(n===null?($('mode').value==='all'?'0.00%':'—'):`${n.toFixed(2)}%`):key==='bets'?n:money(n)}</td>`;
 }).join('')}</tr>`).join('')||`<tr><td colspan="${columns.length}">No matching data. Try another search or filter.</td></tr>`}</tbody></table>`;
}
function render(){
 data=performance(rows,{mode:$('mode').value,year:$('year').value,tour:$('tour').value,view:view==='overview'?'yearly':view==='period'?$('period').value:view});
 const t=data.total,r=roi(t);$('basis').textContent=$('mode').value==='settled'?'Settled only · recorded Profit/Loss, matching the former Tournament P/L page.':'Includes pending · return minus stake, matching the former ROI page. Pending stakes can reduce the displayed ROI before settlement.';
 $('cards').innerHTML=[['ROI',r===null?($('mode').value==='all'?'0.00%':'—'):`${r.toFixed(2)}%`,r],['Profit / loss',money(t.pl),t.pl],['Total stake',money(t.stake),0],['Total return',money(t.returned),0],['Bets',`${t.bets}`,0]].map(([label,value,n])=>`<div class="card">${label}<strong class="${tone(n)}">${value}</strong></div>`).join('');
 setView();renderTable();
}
$('views').addEventListener('click',e=>{const b=e.target.closest('[data-view]');if(!b)return;view=b.dataset.view;sort=view==='player'?'roi':'latest';direction=view==='player'?-1:1;$('search').value='';render();});
$('results').addEventListener('click',e=>{const b=e.target.closest('[data-sort]');if(!b)return;direction=sort===b.dataset.sort?-direction:1;sort=b.dataset.sort;renderTable();});
for(const id of ['year','tour','mode','period'])$(id).addEventListener('change',render);
$('search').addEventListener('input',renderTable);
try{
 rows=await window.BettingData.load({forceRefresh:true});const normalized=rows.map(r=>normalizeBet(r,'all'));
 const years=[...new Set(normalized.map(b=>b.year).filter(y=>y!=='Undated'))].sort().reverse(),tours=[...new Set(normalized.map(b=>b.tour))].sort();
 years.forEach(y=>$('year').add(new Option(y,y)));tours.forEach(t=>$('tour').add(new Option(t,t)));
 $('mode').value=params.get('mode')==='all'?'all':'settled';
 $('year').value=years.includes(params.get('year'))?params.get('year'):params.get('year')==='all'?'all':years.includes(String(new Date().getFullYear()))?String(new Date().getFullYear()):'all';
 if(tours.includes(params.get('tour')))$('tour').value=params.get('tour');
 if(['yearly','monthly','weekly'].includes(params.get('period')))$('period').value=params.get('period');
 for(const id of ['year','tour','mode','search'])$(id).disabled=false;render();
}catch(e){$('status').textContent='Betting data unavailable. Please reload to retry.';console.error(e);}
