// Pure scoring functions shared by the static page and regression tests.
export const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
export function number(value) {
    return value === null || value === undefined || String(value).trim() === '' || typeof value === 'boolean'
        ? null : Number.isFinite(Number(value)) ? Number(value) : null;
}
export function probability(row, metric) {
    const keys = {win: ['p_win_%', 'p_win', 'pWin'], top10: ['p_top10_%', 'p_top10', 'pTop10'], cut: ['p_mc_%', 'p_mc', 'pMakeCut']}[metric];
    const value = number(keys.map(key => row[key]).find(value => value !== undefined && value !== null));
    return value !== null && value >= 0 && value <= 100 ? value / 100 : null;
}
export function nameKeys(value) {
    let name = String(value ?? '');
    if (name.includes(',')) { const [last, ...first] = name.split(','); name = `${first.join(' ')} ${last}`; }
    const tokens = name.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase().match(/[a-z0-9]+/g)?.filter(t => !['jr', 'sr', 'ii', 'iii', 'iv', 'v'].includes(t)) ?? [];
    if (!tokens.length) return [];
    const canonical = [...tokens];
    if (canonical[0] === 'matt') canonical[0] = 'matthew';
    return [...new Set([tokens.join(''), canonical.join(''), ...(tokens.length > 2 ? [tokens[0] + tokens.at(-1), canonical[0] + canonical.at(-1)] : [])])];
}
const playerName = row => row.player_name ?? row.player ?? row.Player ?? row.Name;
export function outcome(row, metric) {
    const text = String(row.finish_text ?? '').toUpperCase().trim();
    const finish = number(row.finish_pos) ?? (/^T?\d+$/.test(text) ? Number(text.replace('T', '')) : null);
    if (metric === 'cut') {
        if ([true, 1, 'true', 'True', '1'].includes(row.made_cut)) return 1;
        if ([false, 0, 'false', 'False', '0'].includes(row.made_cut)) return 0;
        if (['CUT', 'MC'].includes(text)) return 0;
        return finish !== null && finish > 0 ? 1 : null;
    }
    if (finish !== null && finish > 0) return Number(finish <= (metric === 'win' ? 1 : 10));
    return ['CUT', 'MC', 'WD', 'DQ', 'W/D'].includes(text) ? 0 : null;
}
export function eligibility(event, summary) {
    if (event.reconstruction || event.reconstructed || event.prediction_source === 'reconstructed') return 'Reconstructed predictions';
    if (event.prediction_snapshot !== 'initial') return 'No verified initial snapshot';
    const start = Date.parse(`${event.start_date}T00:00:00Z`), created = Date.parse(event.initial_snapshot_created_utc);
    if (!Number.isFinite(start) || !Number.isFinite(created) || created >= start) return 'Snapshot not verified before start date';
    if (!['completed', 'finished'].includes(String(summary?.status).toLowerCase())) return 'Not completed / summary unavailable';
    if (/presidents cup|ryder cup|match play|matchplay/i.test(event.event_name)) return 'Unsupported event format';
    return null;
}
export function groupName(event, summary, dimension) {
    if (dimension === 'tour') return event.tour === 'pga' ? 'PGA Tour' : event.tour === 'euro' ? 'DP World Tour' : 'Unknown tour';
    if (dimension === 'type') return /^(the )?masters( tournament)?$|^pga championship$|^u\.?s\.? open$|^(the )?open championship$/i.test(event.event_name.trim()) ? 'Major' : 'Regular event';
    const size = number(summary?.field_size);
    return size === null || size <= 0 ? 'Unknown field size' : size < 80 ? 'Small field (<80)' : size < 120 ? 'Medium field (80–119)' : 'Full field (120+)';
}
export function scoreEvent(event, summary, predictions, results, metric) {
    const record = {event, summary, reason: eligibility(event, summary), rows: [], unmatched: 0, missing: 0};
    if (record.reason) return record;
    const players = results?.players;
    if (!Array.isArray(predictions) || !predictions.length || !Array.isArray(players) || !players.length) return {...record, reason: 'Predictions or results unavailable'};
    for (const key of ['event_id', 'tour', 'year']) {
        if (results.event?.[key] !== undefined && String(results.event[key]) !== String(event[key])) return {...record, reason: 'Results event identity mismatch'};
    }
    const indexed = new Map();
    players.forEach((row, i) => nameKeys(playerName(row)).forEach(key => {
        if (!indexed.has(key)) indexed.set(key, new Set());
        indexed.get(key).add(i);
    }));
    const used = new Set();
    for (const pred of predictions) {
        let match;
        for (const key of nameKeys(playerName(pred))) {
            const candidates = indexed.get(key);
            if (candidates?.size === 1) { match = [...candidates][0]; break; }
        }
        if (match === undefined) { record.unmatched++; continue; }
        if (used.has(match)) return {...record, rows: [], reason: 'Duplicate player match'};
        used.add(match);
        const p = probability(pred, metric), y = outcome(players[match], metric);
        if (p === null || y === null) { record.missing++; continue; }
        record.rows.push({p, y, resultIndex: match});
    }
    record.resultCount = players.length;
    record.predictionCount = predictions.length;
    record.unmatchedResults = players.length - used.size;
    const known = players.map(row => outcome(row, metric)).filter(y => y !== null);
    const positives = known.filter(y => y === 1).length;
    if (metric === 'win' && positives !== 1) record.reason = 'Missing or ambiguous winner';
    else if (!known.length || !positives) record.reason = 'Required outcomes missing';
    else if (metric === 'cut' && !known.includes(0)) record.reason = 'No observed cut (no-cut / incomplete data)';
    else if (metric !== 'cut' && record.rows.filter(row => row.y === 1).length !== positives) record.reason = 'Positive outcome missing from scored predictions';
    else if (record.rows.length / players.length < 0.95 || record.rows.length / predictions.length < 0.95) record.reason = 'Less than 95% prediction / result coverage';
    if (record.reason) return record;
    // Retrospective null model: all players get the full known-results outcome rate.
    const baseline = positives / known.length;
    const loss = (p, y) => -(y * Math.log(Math.max(1e-6, Math.min(1 - 1e-6, p))) + (1 - y) * Math.log(Math.max(1e-6, Math.min(1 - 1e-6, 1 - p))));
    record.brier = mean(record.rows.map(({p, y}) => (p - y) ** 2));
    record.logLoss = mean(record.rows.map(({p, y}) => loss(p, y)));
    record.baseline = mean(record.rows.map(({y}) => (baseline - y) ** 2));
    record.delta = record.brier - record.baseline;
    return record;
}
export function interval(values) {
    if (values.length < 2) return null;
    let seed = 3917;
    const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
    const draws = Array.from({length: 1000}, () => mean(values.map(() => values[Math.floor(random() * values.length)]))).sort((a, b) => a - b);
    return [draws[24], draws[974]];
}
export function aggregate(records) {
    const events = records.filter(row => !row.reason);
    if (!events.length) return null;
    const rows = events.flatMap(event => event.rows.map(row => ({...row, weight: 1 / event.rows.length / events.length})));
    const bins = Array.from({length: 10}, (_, index) => {
        const sample = rows.filter(row => Math.min(9, Math.floor(row.p * 10)) === index);
        const weight = sample.reduce((sum, row) => sum + row.weight, 0);
        return {label: `${index * 10}–${(index + 1) * 10}%`, n: sample.length, weight,
            predicted: weight ? sample.reduce((sum, row) => sum + row.p * row.weight, 0) / weight : null,
            observed: weight ? sample.reduce((sum, row) => sum + row.y * row.weight, 0) / weight : null};
    });
    return {events: events.length, players: rows.length, brier: mean(events.map(e => e.brier)), baseline: mean(events.map(e => e.baseline)), logLoss: mean(events.map(e => e.logLoss)), delta: mean(events.map(e => e.delta)), interval: interval(events.map(e => e.delta)), gap: bins.reduce((sum, bin) => sum + (bin.n ? bin.weight * Math.abs(bin.predicted - bin.observed) : 0), 0), bins};
}
