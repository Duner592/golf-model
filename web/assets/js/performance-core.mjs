// Two explicit accounting bases preserve the predecessor pages' totals.
export const moneyNumber = value => {
    const n = parseFloat(String(value || '').replace(/[£,\s]/g,'').replace(/^\((.*)\)$/,'-$1'));
    return Number.isFinite(n) ? n : NaN;
};
export function betDate(value) {
    const p = String(value || '').split('/').map(Number);
    if(p.length !== 3 || !p.every(Number.isFinite)) return null;
    const d = new Date(p[2],p[1]-1,p[0]);
    return Number.isFinite(d.getTime()) ? d : null;
}
function isoWeek(date) {
    const d = new Date(Date.UTC(date.getFullYear(),date.getMonth(),date.getDate()));
    d.setUTCDate(d.getUTCDate()+4-(d.getUTCDay()||7));
    return Math.ceil((((d-new Date(Date.UTC(d.getUTCFullYear(),0,1)))/86400000)+1)/7);
}
export function normalizeBet(row, mode) {
    const date = betDate(row.Date), pending = String(row.Outcome || '').toLowerCase().includes('pending');
    let stake, returned, pl;
    if(mode === 'settled') {
        const s=moneyNumber(row.Stake), r=moneyNumber(row.Return), recorded=moneyNumber(row['Profit/Loss']);
        pl=Number.isFinite(recorded)?recorded:Number.isFinite(s)||Number.isFinite(r)?(r||0)-(s||0):NaN;
        if(pending || !date || !Number.isFinite(pl)) return null;
        stake=Math.max(0,s||0);returned=Math.max(0,r||0);
    } else {
        // Deliberately retain the legacy ROI parser rather than change historical figures.
        const legacy=v=>parseFloat(String(v||'').replace('£','').replace(',',''));
        stake=legacy(row.Stake);returned=legacy(row.Return);
        stake=Number.isNaN(stake)?0:stake;returned=Number.isNaN(returned)?0:returned;pl=returned-stake;
    }
    return {date,year:date?String(date.getFullYear()):'Undated',tour:String(row['Tour/Majors']||'').trim()||'Other',player:row.Player||'Unknown',tournament:mode==='settled'?(String(row.Tournament||'').trim()||'Unspecified tournament'):(row.Tournament||'Unknown'),stake,returned,pl,pending};
}
export const roi = value => value.mode==='all' && value.stake<=0 ? 0 : value.stake ? value.pl/value.stake*100 : null;
export function performance(rows,{mode='settled',year='all',tour='all',view='yearly'}={}) {
    const bets=rows.map(r=>normalizeBet(r,mode)).filter(b=>b&&(year==='all'||b.year===year)&&(tour==='all'||b.tour===tour));
    const empty=()=>({mode,bets:0,stake:0,returned:0,pl:0,pending:0,latest:null});
    const total=empty(), groups=new Map();let undated=0;
    const add=(g,b)=>{g.bets++;g.stake+=b.stake;g.returned+=b.returned;g.pl+=b.pl;g.pending+=Number(b.pending);if(b.date&&(!g.latest||b.date>g.latest))g.latest=b.date;};
    for(const b of bets) {
        add(total,b);if(!b.date)undated++;
        let name,key;
        if(view==='tournament'){name=b.tournament;key=JSON.stringify([b.year,b.tour,name]);}
        else if(view==='player'){name=b.player;key=name;}
        else {
            if(!b.date)continue;
            name=view==='monthly'?`${b.year}-${String(b.date.getMonth()+1).padStart(2,'0')}`:view==='weekly'?`${b.year}-W${String(isoWeek(b.date)).padStart(2,'0')}`:b.year;key=name;
        }
        if(!groups.has(key))groups.set(key,{...empty(),name,year:b.year,tour:b.tour});
        add(groups.get(key),b);
    }
    const result=[...groups.values()].sort((a,b)=>(a.latest?.getTime()||0)-(b.latest?.getTime()||0)||a.name.localeCompare(b.name));
    let running=0;for(const g of result){running+=g.pl;g.running=running;}
    return {total,groups:result,undated};
}
