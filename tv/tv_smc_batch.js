/**
 * Batch-fetch the LuxAlgo SMC indicator (daily + weekly) for many tickers. RUNS ONLY IN DOCKER.
 * Resumable (skips tickers already written), gentle (one reused WS client, delays), trims to event columns.
 *
 * Usage: docker compose run --rm tv tv_smc_batch.js <symbols-file-or-comma-list> [maxbars]
 *   docker compose run --rm tv tv_smc_batch.js /data/tv_symbols_pilot.txt 2500
 * Bare tickers are resolved to the US primary listing via searchMarket. Output: /data/tradingview/smc/<TICKER>.json
 *   { ticker, symbol, daily:[{t, <event cols>}...], weekly:[...] }
 * Event cols kept: BOS/CHoCH (internal+swing), OB breakouts, Equal H/L, FVG. Recolored candles dropped.
 */
const fs = require('fs');
const path = require('path');
const TV = require('@mathieuc/tradingview');

const arg = process.argv[2] || '/data/tv_symbols_pilot.txt';
const MAXBARS = Number(process.argv[3] || 2500);
const SMC = process.argv[4] || 'PUB;6daafb2cabe6419d98ae25229d2327f8';   // any indicator id (default LuxAlgo SMC)
const outDir = '/data/tradingview/' + (process.argv[5] || 'smc');          // output subdir
fs.mkdirSync(outDir, { recursive: true });

let tickers = [];
if (fs.existsSync(arg)) tickers = fs.readFileSync(arg, 'utf8').split(/\s+/).filter(Boolean);
else tickers = arg.split(',').map((s) => s.trim()).filter(Boolean);

const US = /^(NASDAQ|NYSE|AMEX|NYSE ARCA|BATS|OTC|CBOE)/i;
const client = new TV.Client({ token: process.env.TV_SESSION, signature: process.env.TV_SIGNATURE });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function resolve(ticker) {
  try {
    const r = await TV.searchMarket(ticker);
    const eq = r.find((x) => /stock/i.test(x.type || '') && US.test(x.exchange || x.id || '')) ||
               r.find((x) => /stock/i.test(x.type || '')) || r[0];
    return eq ? eq.id : null;
  } catch (e) {
    return null;
  }
}

function fetchStudy(fullsym, tf, indic) {
  return new Promise((resolve) => {
    const chart = new client.Session.Chart();
    let done = false;
    const finish = (data) => {
      if (done) return;
      done = true;
      try { chart.delete(); } catch (e) {}
      resolve(data);
    };
    try { chart.setMarket(fullsym, { timeframe: tf, range: MAXBARS }); } catch (e) { return finish(null); }
    chart.onError(() => finish(null));
    const study = new chart.Study(indic);
    study.onError(() => finish(null));
    study.onUpdate(() => {
      if (!study.periods || !study.periods.length) return;
      const rows = study.periods
        .slice()
        .sort((a, b) => a.time - b.time)
        .map((p) => {
          const o = { t: p.$time };
          for (const k in p) if (k !== '$time' && !k.startsWith('plotcandle')) o[k] = p[k];
          return o;
        });
      finish(rows);
    });
    setTimeout(() => finish(null), 15000);
  });
}

(async () => {
  const indic = await TV.getIndicator(SMC);
  let ok = 0, skip = 0, fail = 0;
  for (const tk of tickers) {
    const outPath = path.join(outDir, tk.replace(/[^A-Za-z0-9]/g, '_') + '.json');
    if (fs.existsSync(outPath)) { skip++; continue; }
    const sym = tk.includes(':') ? tk : await resolve(tk);
    if (!sym) { fail++; console.error('noresolve', tk); await sleep(150); continue; }
    const daily = await fetchStudy(sym, 'D', indic);
    await sleep(200);
    const weekly = await fetchStudy(sym, 'W', indic);
    if (!daily && !weekly) { fail++; console.error('nodata', tk, sym); }
    else { fs.writeFileSync(outPath, JSON.stringify({ ticker: tk, symbol: sym, daily, weekly })); ok++; }
    if ((ok + fail) % 20 === 0) console.error(`progress: ${ok} ok / ${fail} fail / ${skip} skip`);
    await sleep(250);
  }
  console.error(`DONE: ${ok} ok / ${fail} fail / ${skip} skip (of ${tickers.length})`);
  client.end();
  process.exit(0);
})();
