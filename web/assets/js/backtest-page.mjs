import {aggregate, eligibility, groupName, scoreEvent, selectSnapshot} from './backtest-core.mjs';
const $ = id => document.getElementById(id);
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const format = n => Number.isFinite(n) ? n.toFixed(4) : '—';
const percent = n => Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : '—';
function table(headers, rows, caption) {
    return `<table><caption>${escape(caption)}</caption><thead><tr>${headers.map(h => `<th scope="col">${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
async function json(path) {
    const response = await fetch(path, {cache: 'no-cache', signal: AbortSignal.timeout(15000)});
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.json();
}
let loaded = [];
function render() {
    const metric = $('metric').value, dimension = $('dimension').value;
    const selected = loaded.filter(({event}) => ($('year').value === 'all' || String(event.year) === $('year').value) && ($('tour').value === 'all' || event.tour === $('tour').value));
    const records = selected.map(item => {
        const event = selectSnapshot(item.event, $('snapshot').value);
        const final = event.selected_snapshot === 'final';
        const reason = eligibility(event, item.summary) || item.error || (final ? item.finalError : item.initialError);
        const summary = final ? {...item.summary, field_size:item.finalSummary?.field_size ?? null} : item.summary;
        return reason ? {...item, event, reason} : scoreEvent(event, summary, final ? item.finalPredictions : item.predictions, item.results, metric);
    });
    const included = records.filter(row => !row.reason), total = aggregate(records);
    $('status').textContent = `${included.length} of ${records.length} selected archive events included. ${records.length - included.length} excluded; see coverage below.`;
    const tiles = [['Scored events', included.length], ['Player appearances', total?.players ?? 0], ['Excluded events', records.length - included.length], ['Calibration gap', total ? `${(total.gap * 100).toFixed(1)} pp` : '—']];
    $('tiles').innerHTML = tiles.map(([label, value]) => `<div class="tile">${escape(label)}<strong>${escape(value)}</strong></div>`).join('');
    const groups = new Map();
    included.forEach(row => {const key = groupName(row.event, row.summary, dimension); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(row);});
    const groupRows = [...groups].sort(([a], [b]) => a.localeCompare(b)).map(([label, events]) => {
        const score = aggregate(events);
        return [label, score.events, score.players, format(score.brier), format(score.baseline), format(score.delta), score.interval ? score.interval.map(format).join(' to ') : 'Unavailable', format(score.logLoss), `${(score.gap * 100).toFixed(1)} pp`, score.events < 10 ? 'Limited evidence (<10 events)' : 'Descriptive sample'];
    });
    $('groups').innerHTML = groupRows.length ? table(['Group', 'Events', 'Appearances', 'Brier', 'Baseline Brier', 'Difference', '95% difference interval', 'Log loss', 'Calibration gap', 'Evidence'], groupRows, 'Event-weighted scores; lower is better.') : '<p>No eligible events for these filters. Check the exclusions below.</p>';
    $('calibration').innerHTML = total ? table(['Probability band', 'Appearances', 'Mean predicted', 'Observed rate'], total.bins.filter(b => b.n).map(b => [b.label, b.n, percent(b.predicted), percent(b.observed)]), 'Event-weighted rates; counts are unweighted appearances.') : '<p>No calibration data available.</p>';
    $('audit').innerHTML = table(['Event', 'Tour', 'Start', 'Snapshot', 'Prediction time (UTC)', 'Status / reason', 'Scored', 'Unmatched predictions', 'Unmatched results', 'Missing probability / outcome'], records.map(r => [r.event.event_name, r.event.tour, r.event.start_date, r.event.selected_snapshot === 'final' ? 'Final pre-event' : r.event.selected_snapshot === 'missing-final' ? 'Unavailable' : 'Initial', r.event.selected_snapshot === 'final' ? r.event.final_snapshot.prediction_generated_utc : r.event.initial_snapshot_created_utc ?? 'Unknown', r.reason ?? 'Included', r.reason ? '—' : r.rows.length, r.unmatched ?? '—', r.unmatchedResults ?? '—', r.missing ?? '—']), 'Excluded events do not contribute to any metric. Preferred mode uses the initial archive only when no final snapshot was saved.');
}
async function main() {
    for (const id of ['year', 'tour', 'metric', 'dimension', 'snapshot']) $(id).disabled = true;
    const index = await json('archive/index.json');
    if (!Array.isArray(index)) throw new Error('Archive index is not a list');
    const seen = new Set();
    const events = index.filter(event => {const key = `${event.tour}/${event.year}/${event.event_id}`; if (seen.has(key)) return false; seen.add(key); return true;});
    [...new Set(events.map(e => String(e.year)))].sort().reverse().forEach(year => $('year').add(new Option(year, year)));
    let cursor = 0, finished = 0;
    await Promise.all(Array.from({length: Math.min(6, events.length)}, async () => {
        while (cursor < events.length) {
            const event = events[cursor++];
            let item = {event};
            try {
                // Inspect provenance before requesting files for old/unverified events.
                const reason = eligibility(event, {status:'completed'}) && eligibility(selectSnapshot(event), {status:'completed'});
                if (reason) item.error = reason;
                else {
                    if (!/^\d{4}$/.test(String(event.year)) || !/^[a-zA-Z0-9_-]+$/.test(event.slug)) throw new Error('Invalid archive path');
                    const base = `archive/${event.year}/${event.slug}`;
                    item.summary = await json(`${base}/tournament_summary.json`);
                    if (!eligibility(event, item.summary) || !eligibility(selectSnapshot(event), item.summary)) {
                        item.results = await json(`${base}/results.json`);
                        try { item.predictions = await json(`${base}/leaderboard.json`); }
                        catch (error) {item.initialError = `Initial data unavailable: ${error.message}`;}
                        if (event.final_snapshot) {
                            try {
                                const meta = await json(`${base}/final/snapshot.json`);
                                for (const key of ['snapshot_type','event_id','tour','year','start_date','timezone','prediction_generated_utc','snapshot_created_utc','cutoff_utc','capture_window_open_utc']) {
                                    if (meta[key] !== event.final_snapshot[key]) throw new Error('Final snapshot metadata does not match archive index');
                                }
                                [item.finalPredictions, item.finalSummary] = await Promise.all([json(`${base}/final/leaderboard.json`), json(`${base}/final/tournament_summary.json`)]);
                            } catch (error) {item.finalError = `Final data unavailable: ${error.message}`;}
                        }
                    }
                }
            } catch (error) {item.error = `Data unavailable: ${error.message}`;}
            loaded.push(item);
            $('status').textContent = `Checking archives… ${++finished}/${events.length}`;
        }
    }));
    loaded.sort((a, b) => String(b.event.start_date).localeCompare(String(a.event.start_date)) || a.event.event_name.localeCompare(b.event.event_name));
    for (const id of ['year', 'tour', 'metric', 'dimension', 'snapshot']) {$(id).disabled = false; $(id).addEventListener('change', render);}
    render();
}
main().catch(error => {$('status').textContent = `Unable to load backtests: ${error.message}. Please reload to retry.`;});
