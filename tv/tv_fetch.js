/**
 * Dockerized TradingView data + indicator fetcher.
 * RUNS ONLY IN DOCKER (see docker-compose `tv` service). Never invoke with a host node.
 *
 * Usage (via compose):
 *   docker compose run --rm tv tv_fetch.js <SYMBOL> [TIMEFRAME] [NBARS] [INDICATOR_ID]
 * Examples:
 *   docker compose run --rm tv tv_fetch.js NASDAQ:AAPL D 300
 *   docker compose run --rm tv tv_fetch.js AMEX:SPY W 500
 *   docker compose run --rm tv tv_fetch.js NASDAQ:MU D 300 STD;Relative_Strength_Index
 *
 * Auth: anonymous works for delayed/most data. For real-time / premium, set TV_SESSION (+ TV_SIGNATURE)
 * in .env — a logged-in TradingView `sessionid` cookie. Never hardcode it.
 *
 * Output: writes /data/tradingview/<SYMBOL>_<TF>.json (shared repo ./.data), which the Python engine can read.
 * NOTE: this uses TradingView's unofficial WebSocket API — for PERSONAL research use; mind TradingView's ToS.
 */
const fs = require('fs');
const path = require('path');
const TradingView = require('@mathieuc/tradingview');

const [, , symbol = 'NASDAQ:AAPL', timeframe = 'D', nbars = '300', indicatorId = ''] = process.argv;
const token = process.env.TV_SESSION || undefined;
const signature = process.env.TV_SIGNATURE || undefined;

const client = new TradingView.Client(token ? { token, signature } : {});
const chart = new client.Session.Chart();
chart.setMarket(symbol, { timeframe, range: Number(nbars) });

let wroteData = false;
let study = null;
let studyDone = !indicatorId;

function finishMaybe() {
  if (wroteData && studyDone) {
    client.end();
    process.exit(0);
  }
}

chart.onError((...err) => {
  console.error('CHART ERROR:', ...err);
  client.end();
  process.exit(1);
});

chart.onSymbolLoaded(() => {
  console.error(`loaded: ${chart.infos.description} (${chart.infos.currency_id}, ${chart.infos.type})`);
});

chart.onUpdate(() => {
  if (wroteData || !chart.periods.length) return;
  wroteData = true;
  const bars = chart.periods
    .slice()
    .sort((a, b) => a.time - b.time)
    .map((p) => ({
      time: p.time,
      date: new Date(p.time * 1000).toISOString().slice(0, 10),
      open: p.open,
      high: p.max,
      low: p.min,
      close: p.close,
      volume: p.volume,
    }));
  const outDir = '/data/tradingview';
  fs.mkdirSync(outDir, { recursive: true });
  const safe = symbol.replace(/[^A-Za-z0-9]/g, '_');
  const outPath = path.join(outDir, `${safe}_${timeframe}.json`);
  fs.writeFileSync(
    outPath,
    JSON.stringify({ symbol, timeframe, description: chart.infos.description, n: bars.length, bars }, null, 2),
  );
  console.error(`wrote ${bars.length} OHLCV bars -> ${outPath}`);
  finishMaybe();
});

// Optional: attach a built-in / public indicator and dump its series alongside the OHLCV.
if (indicatorId) {
  TradingView.getIndicator(indicatorId)
    .then((indic) => {
      study = new chart.Study(indic);
      study.onError((...e) => {
        console.error('STUDY ERROR:', ...e);
        studyDone = true;
        finishMaybe();
      });
      study.onUpdate(() => {
        if (studyDone || !study.periods.length) return;
        studyDone = true;
        const outDir = '/data/tradingview';
        fs.mkdirSync(outDir, { recursive: true });
        const safe = symbol.replace(/[^A-Za-z0-9]/g, '_');
        const idSafe = indicatorId.replace(/[^A-Za-z0-9]/g, '_');
        const outPath = path.join(outDir, `${safe}_${timeframe}_${idSafe}.json`);
        fs.writeFileSync(outPath, JSON.stringify(study.periods.slice().sort((a, b) => a.time - b.time), null, 2));
        console.error(`wrote indicator ${indicatorId} series -> ${outPath}`);
        finishMaybe();
      });
    })
    .catch((e) => {
      console.error('getIndicator failed:', e && e.message ? e.message : e);
      studyDone = true;
      finishMaybe();
    });
}

setTimeout(() => {
  console.error('timeout (30s) — is the symbol valid? for premium data set TV_SESSION');
  client.end();
  process.exit(2);
}, 30000);
