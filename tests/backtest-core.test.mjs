import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {aggregate, eligibility, groupName, interval, nameKeys, number, outcome, probability, scoreEvent, selectSnapshot} from '../web/assets/js/backtest-core.mjs';

const event = {event_id:'1', year:'2026', tour:'pga', event_name:'Test Open', prediction_snapshot:'initial', start_date:'2026-06-04', initial_snapshot_created_utc:'2026-06-01T12:00:00Z'};
const summary = {status:'completed', field_size:2};
const predictions = [{player_name:'Smith, Matt', 'p_win_%':75}, {player_name:'Alex Brown', 'p_win_%':25}];
const results = {players:[{player:'Matthew Smith', finish_pos:1}, {player:'Alex Brown', finish_pos:2}]};

test('final preferred uses saved evening run, initial remains selectable, absent final falls back', () => {
    const final = {snapshot_type:'final', event_id:'1', tour:'pga', year:'2026', start_date:'2026-06-04', timezone:'America/Denver', capture_window_open_utc:'2026-06-04T00:00:00Z', cutoff_utc:'2026-06-04T06:00:00Z', prediction_generated_utc:'2026-06-04T02:00:00Z', snapshot_created_utc:'2026-06-04T02:10:00Z'};
    const withFinal = {...event, final_snapshot:final};
    assert.equal(selectSnapshot(withFinal).selected_snapshot, 'final');
    assert.equal(eligibility(selectSnapshot(withFinal), summary), null);
    assert.equal(selectSnapshot(withFinal, 'initial').selected_snapshot, 'initial');
    assert.equal(selectSnapshot(event).selected_snapshot, 'initial');
    assert.match(eligibility(selectSnapshot(event, 'final'), summary), /No saved final/);
    for (const patch of [{snapshot_created_utc:final.cutoff_utc}, {tour:'euro'}, {timezone:''}, {cutoff_utc:'2026-06-05T06:00:00Z'}]) {
        assert.ok(eligibility(selectSnapshot({...event, final_snapshot:{...final, ...patch}}), summary));
    }
});

test('null and blank probabilities are not zero; percentages are bounded', () => {
    for (const value of [null, undefined, '', ' ', false, 'NaN']) assert.equal(number(value), null);
    assert.equal(probability({'p_win_%':0}, 'win'), 0);
    assert.equal(probability({'p_win_%':100}, 'win'), 1);
    for (const value of [-1, 101, '', null]) assert.equal(probability({'p_win_%':value}, 'win'), null);
});
test('name normalization handles aliases, accents, suffixes and first-last fallback', () => {
    assert.ok(nameKeys('Fitzpatrick, Matt').includes('matthewfitzpatrick'));
    assert.ok(nameKeys('Eugenio Lopez-Chacarra').includes('eugeniochacarra'));
    assert.ok(nameKeys('José Smith Jr.').includes('josesmith'));
});
test('provenance must be initial, pre-date and not reconstructed', () => {
    assert.equal(eligibility(event, summary), null);
    for (const patch of [{prediction_snapshot:null}, {reconstruction:{}}, {initial_snapshot_created_utc:'2026-06-04T01:00:00Z'}, {start_date:null}]) assert.ok(eligibility({...event, ...patch}, summary));
    assert.ok(eligibility(event, {status:'upcoming'}));
});
test('scores match hand calculations and use exact zero/one Brier probabilities', () => {
    const score = scoreEvent(event, summary, predictions, results, 'win');
    assert.equal(score.reason, null);
    assert.equal(score.brier, 0.0625);
    assert.equal(score.baseline, 0.25);
    assert.equal(score.delta, -0.1875);
    assert.ok(Math.abs(score.logLoss + Math.log(.75)) < 1e-12);
    const perfect = predictions.map((p, i) => ({...p, 'p_win_%': i ? 0 : 100}));
    assert.equal(scoreEvent(event, summary, perfect, results, 'win').brier, 0);
});
test('winner missing, duplicates and mismatched event identity are excluded', () => {
    assert.match(scoreEvent(event, summary, predictions.slice(1), results, 'win').reason, /Positive outcome/);
    assert.match(scoreEvent(event, summary, [...predictions, predictions[0]], results, 'win').reason, /Duplicate/);
    assert.match(scoreEvent(event, summary, predictions, {...results, event:{tour:'euro'}}, 'win').reason, /identity/);
    assert.match(scoreEvent(event, summary, predictions, {players:[{player:'Alex Brown', finish_pos:2}]}, 'win').reason, /winner/);
});
test('ambiguous fallback names do not silently match', () => {
    const ambiguous = {players:[{player:'Alex A Brown', finish_pos:1}, {player:'Alex B Brown', finish_pos:2}]};
    const score = scoreEvent(event, summary, predictions, ambiguous, 'win');
    assert.ok(score.reason);
    assert.equal(score.unmatched, 2);
});
test('ties count as top10, unknown WD cut outcome is not treated as a miss', () => {
    assert.equal(outcome({finish_text:'T10'}, 'top10'), 1);
    assert.equal(outcome({finish_text:'T11'}, 'top10'), 0);
    assert.equal(outcome({finish_text:'WD'}, 'cut'), null);
    assert.equal(outcome({finish_text:'WD', made_cut:true}, 'cut'), 1);
    assert.equal(outcome({finish_text:'MC'}, 'cut'), 0);
    assert.equal(outcome({finish_text:'WD'}, 'win'), 0);
    assert.match(scoreEvent(event, summary, predictions.map(p => ({...p, p_mc:100})), results, 'cut').reason, /No observed cut/);
});
test('coverage threshold and baseline use full results, not just matched players', () => {
    const players = Array.from({length:21}, (_, i) => ({player:`Player ${i}`, finish_pos:i + 1}));
    const preds = players.slice(0, 20).map(row => ({player_name:row.player, p_win:5}));
    const score = scoreEvent(event, summary, preds, {players}, 'win');
    assert.equal(score.reason, null);
    assert.equal(score.unmatchedResults, 1);
    const expected = ((1 - 1 / 21) ** 2 + 19 * (1 / 21) ** 2) / 20;
    assert.ok(Math.abs(score.baseline - expected) < 1e-12);
    assert.match(scoreEvent(event, summary, preds.slice(0, 19), {players}, 'win').reason, /95%/);
    const missingWinnerProbability = preds.map((p, i) => ({...p, p_win:i === 0 ? null : p.p_win}));
    assert.match(scoreEvent(event, summary, missingWinnerProbability, {players}, 'win').reason, /Positive outcome/);
});
test('field group boundaries and majors exclude similarly named regular events', () => {
    assert.equal(groupName(event, {field_size:79}, 'field'), 'Small field (<80)');
    assert.equal(groupName(event, {field_size:80}, 'field'), 'Medium field (80–119)');
    assert.equal(groupName(event, {field_size:120}, 'field'), 'Full field (120+)');
    assert.equal(groupName(event, {}, 'field'), 'Unknown field size');
    for (const name of ['Masters Tournament', 'PGA Championship', 'U.S. Open', 'The Open Championship']) assert.equal(groupName({...event, event_name:name}, summary, 'type'), 'Major');
    assert.equal(groupName({...event, event_name:'BMW PGA Championship'}, summary, 'type'), 'Regular event');
});
test('aggregation weights events equally, bootstrap is deterministic', () => {
    const small = {reason:null, rows:[{p:.5,y:1}], brier:.25, baseline:.25, delta:0, logLoss:1};
    const large = {reason:null, rows:Array.from({length:100}, () => ({p:.1,y:0})), brier:.01, baseline:.02, delta:-.01, logLoss:.1};
    const score = aggregate([small, large]);
    assert.equal(score.events, 2);
    assert.equal(score.players, 101);
    assert.equal(score.brier, .13);
    assert.ok(Math.abs(score.gap - .3) < 1e-10);
    assert.deepEqual(interval([0, -.01]), interval([0, -.01]));
    assert.equal(interval([0]), null);
    assert.equal(aggregate([{reason:'Excluded'}]), null);
});
test('French Open archive retains the winner and name aliases', async () => {
    const load = async path => JSON.parse(await readFile(new URL(`../web/${path}`, import.meta.url), 'utf8'));
    const index = await load('archive/index.json');
    const french = index.find(e => e.year === '2026' && e.slug === 'fedex_open_de_france');
    assert.ok(french);
    const base = `archive/${french.year}/${french.slug}`;
    const score = scoreEvent(french, await load(`${base}/tournament_summary.json`), await load(`${base}/leaderboard.json`), await load(`${base}/results.json`), 'win');
    assert.equal(score.reason, null);
    assert.equal(score.rows.filter(r => r.y === 1).length, 1);
});
test('deduplicated 3M scores while Corales retains independent coverage exclusions', async () => {
    const load = async path => JSON.parse(await readFile(new URL(`../web/${path}`, import.meta.url), 'utf8'));
    const index = await load('archive/index.json');
    for (const slug of ['3m_open', 'corales_puntacana_championship']) {
        const event = index.find(e => e.year === '2026' && e.slug === slug);
        const base = `archive/${event.year}/${slug}`;
        const summary = await load(`${base}/tournament_summary.json`), predictions = await load(`${base}/leaderboard.json`), results = await load(`${base}/results.json`);
        assert.equal(new Set(predictions.map(p => p.player_name)).size, predictions.length);
        for (const metric of ['win', 'top10', 'cut']) {
            const score = scoreEvent(event, summary, predictions, results, metric);
            if (slug === '3m_open') assert.equal(score.reason, null);
            else assert.match(score.reason, /95%|Positive outcome/);
        }
    }
});
