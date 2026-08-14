import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

const API = process.env.REACT_APP_API_URL || '/api';

// Shared fetch that turns backend failures into readable messages instead of a cryptic
// "Unexpected token < in JSON" (what a 502/504 HTML page throws on r.json()) or a dead page.
// Callers: `apiFetch('/scan?...')` -> resolves to parsed JSON, or rejects with a friendly Error.
async function apiFetch(path, opts) {
  const url = path.startsWith('http') ? path : `${API}${path}`;
  let r;
  try {
    r = await fetch(url, opts);
  } catch (e) {
    throw new Error('Backend unreachable — the server may be down or restarting.');
  }
  if (!r.ok) {
    if ([502, 503, 504].includes(r.status))
      throw new Error(`Backend unavailable (${r.status}) — it may be restarting or busy. Retry shortly.`);
    let detail = '';
    try { const j = await r.json(); detail = j.error || j.detail || ''; } catch (_) {}
    throw new Error(`Request failed (${r.status})${detail ? ': ' + detail : ''}`);
  }
  return r.json();
}

// Reusable inline error with an optional Retry — use in place of a swallowed catch / dead "Loading…".
function ErrorBanner({ message, onRetry, onDismiss }) {
  return (
    <div className="error-banner" role="alert">
      <span className="error-banner-icon">&#9888;</span>
      <span className="error-banner-msg">{message || 'Something went wrong.'}</span>
      {onRetry && <button className="error-banner-btn" onClick={onRetry}>Retry</button>}
      {onDismiss && <button className="error-banner-x" onClick={onDismiss} aria-label="Dismiss">&times;</button>}
    </div>
  );
}

// ---- Reusable table sorting ---------------------------------------------------
// useSortedRows(rows, initialKey, initialDir, accessors)
//   - requestSort(key): toggles dir if same key, else selects key with a sensible
//     default direction ('desc' for numbers, 'asc' for strings).
//   - Sorting detects the value type per key from the first non-null value; numbers
//     compare numerically, strings via case-insensitive localeCompare. null/undefined/
//     NaN ALWAYS sort to the bottom regardless of direction. Stable for equal keys.
//   - accessors[key](row) supplies the sort value for computed/derived columns.
function sortValue(row, key, accessors) {
  if (accessors && accessors[key]) return accessors[key](row);
  return row ? row[key] : undefined;
}
function isNil(v) {
  return v === null || v === undefined || (typeof v === 'number' && isNaN(v));
}
function useSortedRows(rows, initialKey = null, initialDir = 'desc', accessors = null) {
  const [sortKey, setSortKey] = useState(initialKey);
  const [sortDir, setSortDir] = useState(initialDir);
  const requestSort = useCallback((key) => {
    setSortKey(prevKey => {
      if (prevKey === key) {
        setSortDir(d => (d === 'asc' ? 'desc' : 'asc'));
        return prevKey;
      }
      // New column: infer default direction from the first non-null value's type.
      let firstVal;
      if (Array.isArray(rows)) {
        for (let i = 0; i < rows.length; i++) {
          const v = sortValue(rows[i], key, accessors);
          if (!isNil(v)) { firstVal = v; break; }
        }
      }
      setSortDir(typeof firstVal === 'string' ? 'asc' : 'desc');
      return key;
    });
  }, [rows, accessors]);

  const sortedRows = React.useMemo(() => {
    if (!Array.isArray(rows) || !sortKey) return rows;
    // Detect type from first non-null value.
    let sample;
    for (let i = 0; i < rows.length; i++) {
      const v = sortValue(rows[i], sortKey, accessors);
      if (!isNil(v)) { sample = v; break; }
    }
    const isString = typeof sample === 'string';
    const dirMul = sortDir === 'asc' ? 1 : -1;
    // Decorate with original index for a stable sort.
    const decorated = rows.map((row, i) => ({ row, i, v: sortValue(row, sortKey, accessors) }));
    decorated.sort((a, b) => {
      const an = isNil(a.v), bn = isNil(b.v);
      if (an && bn) return a.i - b.i;      // both null -> keep order
      if (an) return 1;                     // nulls always to the bottom
      if (bn) return -1;
      let cmp;
      if (isString) cmp = String(a.v).localeCompare(String(b.v), undefined, { sensitivity: 'base' });
      else cmp = a.v < b.v ? -1 : a.v > b.v ? 1 : 0;
      if (cmp !== 0) return cmp * dirMul;
      return a.i - b.i;                     // stable for equal keys
    });
    return decorated.map(d => d.row);
  }, [rows, sortKey, sortDir, accessors]);

  return { rows: sortedRows, sortKey, sortDir, requestSort };
}

// Reusable sortable header cell. `sort` is the object returned by useSortedRows.
function SortTh({ label, colKey, sort, className, title, align, style, children }) {
  const active = sort && sort.sortKey === colKey;
  const arrow = active ? (sort.sortDir === 'asc' ? ' ▲' : ' ▼') : '';
  const cls = 'sortable-th' + (active ? ' sorted' : '') + (className ? ' ' + className : '');
  const st = { ...(align ? { textAlign: align } : {}), ...(style || {}) };
  return (
    <th
      className={cls}
      title={title}
      style={st}
      onClick={() => sort && sort.requestSort(colKey)}
    >
      {children != null ? children : label}
      <span className="sort-arrow">{arrow || '⇅'}</span>
    </th>
  );
}

// Small shared "Updated: <timestamp>" chip. Renders nothing for a missing/invalid value.
function LastUpdatedChip({ value }) {
  if (!value) return null;
  const d = new Date(value);
  if (isNaN(d.getTime())) return null;
  return <span className="last-updated-chip">Updated: {d.toLocaleString()}</span>;
}

// Reusable inline "the edge lives in the tail" mini-bar strip. Renders one small equal-width
// segment per bucket, left→right: a tiny label on top, a vertical bar whose HEIGHT ∝ |value|
// (normalized to the strip's max |value|, min ~3px / max ~26px), colored green (value>0) /
// red (value<0) / dim (~0 or null), and the value printed below. Skips null/absent buckets
// gracefully (renders a dim placeholder so alignment holds). In-file use only.
//   buckets: [{ label, value, n }]   unit: string appended in the tooltip (default '%')
function TailStrip({ buckets, unit = '%' }) {
  const list = Array.isArray(buckets) ? buckets : [];
  const isNum = (v) => v != null && !isNaN(Number(v));
  const vals = list.filter(b => b && isNum(b.value)).map(b => Math.abs(Number(b.value)));
  if (!vals.length) return null;
  const maxAbs = Math.max(...vals, 1e-9);
  const MIN_H = 3, MAX_H = 26;
  const fmt = (v) => `${v > 0 ? '+' : ''}${Number(v).toFixed(1)}`;
  const colorOf = (v) => (!isNum(v) || Math.abs(Number(v)) < 1e-9) ? '#8a8f98' : (Number(v) > 0 ? '#2ec46b' : '#e0555f');
  const segStyle = { flex: '1 1 0', minWidth: 34, textAlign: 'center' };
  const labStyle = { fontSize: 9, color: '#9aa0a8', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' };
  const barBox = { height: MAX_H, display: 'flex', alignItems: 'flex-end', justifyContent: 'center' };
  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'flex-end', margin: '4px 0' }}>
      {list.map((b, i) => {
        const has = b && isNum(b.value);
        const v = has ? Number(b.value) : null;
        const col = colorOf(v);
        const h = has ? Math.max(MIN_H, Math.round((Math.abs(v) / maxAbs) * MAX_H)) : MIN_H;
        return (
          <div key={i} style={segStyle}
               title={b ? `${b.label}: ${has ? v : '–'}${unit} (n=${b && b.n != null ? b.n : '–'})` : ''}>
            <div style={labStyle}>{b ? b.label : ''}</div>
            <div style={barBox}>
              <div style={{ width: 10, height: h, borderRadius: 1, background: has ? col : '#555', opacity: has ? 1 : 0.4 }} />
            </div>
            <div style={{ fontSize: 10, fontFamily: 'ui-monospace, Menlo, monospace', fontWeight: 600, color: col }}>
              {has ? fmt(v) : '–'}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---- "Firing now" indicator (shared by both signal-listing pages) -----------
// FiringCell: a compact count badge for a signal's live fires over the last ~3 bars. When
// count>0 it's a clickable green sm-badge (🔥 N) that opens the FiringPane; stopPropagation
// keeps a table-row's own onClick (expand/toggle) from firing. count 0/falsy -> a dim, inert –.
function FiringCell({ count, signalKey, onOpen }) {
  if (!count) return <span className="dim">–</span>;
  return (
    <span
      className="sm-badge good"
      style={{ cursor: 'pointer' }}
      title={`${count} names fired this signal in the last 3 bars — click for the list`}
      onClick={(e) => { e.stopPropagation(); onOpen && onOpen(signalKey); }}
    >
      🔥 {count}
    </span>
  );
}

// FiringPane: shared overlay listing the names currently firing a given signal. Reuses the
// existing .addiv-chart-overlay backdrop (closes on click) + .addiv-chart-box panel. Hooks run
// unconditionally (fetch on signalKey change; sortable table) and it early-returns null when no
// signalKey is set, so it's rendered once per page and only paints when opened.
function FiringPane({ signalKey, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!signalKey) return;
    let live = true;
    setLoading(true); setError(null); setData(null);
    apiFetch('/signal-firing?signal=' + encodeURIComponent(signalKey))
      .then(d => { if (live) { setData(d); setLoading(false); } })
      .catch(e => { if (live) { setError(e.message || 'Failed to load firing list.'); setLoading(false); } });
    return () => { live = false; };
  }, [signalKey]);

  const firing = (data && data.firing) || [];
  const sort = useSortedRows(firing, 'days_ago', 'asc', {
    sectors: r => (r.sectors || []).slice(0, 2).join(', '),
  });

  if (!signalKey) return null;

  const recentBars = (data && data.recent_bars) || 3;
  return (
    <div className="addiv-chart-overlay" onClick={onClose}>
      <div className="addiv-chart-box" style={{ width: 'min(560px, 94vw)', maxHeight: '80vh' }} onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12, marginBottom: 10 }}>
          <div>
            <div style={{ fontWeight: 700 }}>{(data && data.signal_name) || signalKey}</div>
            <div className="dim" style={{ fontSize: 11 }}>
              {(data && data.n_firing) || 0} firing (last {recentBars}d)
            </div>
          </div>
          <button className="error-banner-x" onClick={onClose} aria-label="Close" title="Close">&times;</button>
        </div>
        {loading ? <div className="dim" style={{ padding: 8 }}>Loading…</div> :
         error ? <ErrorBanner message={error} /> :
         firing.length ? (
          <table className="studies-table">
            <thead>
              <tr>
                <SortTh label="Ticker" colKey="ticker" sort={sort} />
                <SortTh label="Days ago" colKey="days_ago" sort={sort} align="right" />
                <SortTh label="Last close" colKey="last_close" sort={sort} align="right" />
                <SortTh label="Sector(s)" colKey="sectors" sort={sort} />
              </tr>
            </thead>
            <tbody>
              {sort.rows.map(r => (
                <tr key={r.ticker} className="study-row">
                  <td>{r.ticker}</td>
                  <td style={{ textAlign: 'right' }}>{r.days_ago === 0 ? 'today' : `${r.days_ago}d`}</td>
                  <td style={{ textAlign: 'right' }}>{r.last_close != null ? r.last_close : '–'}</td>
                  <td className="dim">{(r.sectors || []).slice(0, 2).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
         ) : <div className="dim" style={{ padding: 8 }}>Not scanned yet or none firing.</div>}
      </div>
    </div>
  );
}

// ---- Server-side pagination + infinite scroll --------------------------------
// usePagedList(path, params): fetches page 1 of `path?paginate=1&<params>` and appends further pages
// (by offset) via loadMore(). Any change to `params` (ordering / dir / search / category / regime / …)
// discards the loaded pages and re-fetches from offset 0 — this is how server-side sort replaces the
// old client-side sort on the large tables. Returns { rows, meta, loading, error, hasMore, loadMore }.
//   rows  — accumulated result rows across the loaded pages
//   meta  — the non-row envelope keys (total, total_studies, categories, last_updated, …)
function usePagedList(path, params) {
  const [rows, setRows] = useState([]);
  const [meta, setMeta] = useState({});
  const [nextOffset, setNextOffset] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Serialize params so the effect keys off value, not object identity (which would loop forever).
  const key = JSON.stringify(params || {});

  const buildUrl = useCallback((offset) => {
    const qs = new URLSearchParams({ paginate: '1', offset: String(offset) });
    const p = JSON.parse(key);
    Object.entries(p).forEach(([k, v]) => { if (v != null && v !== '') qs.set(k, v); });
    return `${path}?${qs.toString()}`;
  }, [path, key]);

  useEffect(() => {
    let live = true;
    setLoading(true); setError(null);
    apiFetch(buildUrl(0))
      .then(d => {
        if (!live) return;
        // eslint-disable-next-line no-unused-vars
        const { results, next_offset, ...rest } = d;
        setRows(results || []); setMeta(rest);
        setNextOffset(next_offset == null ? null : next_offset); setLoading(false);
      })
      .catch(e => { if (live) { setError(e.message || 'Failed to load.'); setLoading(false); } });
    return () => { live = false; };
  }, [buildUrl]);

  const loadMore = useCallback(() => {
    if (nextOffset == null || loading) return;
    setLoading(true);
    apiFetch(buildUrl(nextOffset))
      .then(d => {
        setRows(prev => [...prev, ...(d.results || [])]);
        setNextOffset(d.next_offset == null ? null : d.next_offset); setLoading(false);
      })
      .catch(e => { setError(e.message || 'Failed to load more.'); setLoading(false); });
  }, [nextOffset, loading, buildUrl]);

  return { rows, meta, loading, error, hasMore: nextOffset != null, loadMore };
}

// IntersectionObserver sentinel — invokes onVisible() when scrolled near view. Place at a list's end.
function ScrollSentinel({ onVisible, disabled }) {
  const ref = useRef(null);
  useEffect(() => {
    if (disabled) return undefined;
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') return undefined;
    const obs = new IntersectionObserver(
      entries => { if (entries[0].isIntersecting) onVisible(); },
      { rootMargin: '400px' });
    obs.observe(el);
    return () => obs.disconnect();
  }, [onVisible, disabled]);
  return <div ref={ref} className="scroll-sentinel" aria-hidden="true" style={{ height: 1 }} />;
}

// Site-wide plain-English glossary. Wrap any jargon in <Term k="cagr">CAGR</Term> to get a hover ?.
const GLOSSARY = {
  cagr: ['CAGR', 'Compound Annual Growth Rate — the smoothed yearly % an investment would need to grow at to go from start value to end value over the period. Strips out the lumpiness of individual years.'],
  sortino: ['Sortino ratio', 'Return per unit of DOWNSIDE risk. Like Sharpe, but only penalizes losing volatility — upside swings don’t count against you. Higher is better. Here it runs on a 10-day window.'],
  sharpe: ['Sharpe ratio', 'Return per unit of TOTAL volatility (both up and down). Higher = more return for the risk taken. Above 1 is good, above 2 is excellent.'],
  omega: ['Omega ratio', 'Probability-weighted ratio of gains to losses above/below a threshold. Omega > 1 means the odds are tilted toward gains. Used here on a 10-day window as a bullish gate.'],
  rsi: ['RSI', 'Relative Strength Index (0–100) — momentum oscillator. Below 30 = oversold (potential bounce), above 70 = overbought. Crossing above its own moving average is a bullish trigger.'],
  sma: ['SMA', 'Simple Moving Average — the average price over the last N days. Price or RSI crossing above/below its SMA is a common trend signal.'],
  macd: ['MACD', 'Moving Average Convergence Divergence — momentum indicator from two EMAs. Removed from the signal rules here; kept as an info-only column.'],
  beta: ['Beta', 'How much a stock moves relative to the market (SPY). Beta 1.5 = tends to move 1.5× the market. Used to strip out market-driven moves when isolating a stock’s own reaction.'],
  alpha: ['Alpha', 'Return above what the market move alone would explain (beta-adjusted). Positive alpha = outperformance not attributable to just riding the market.'],
  correlation: ['Correlation', 'How tightly two series move together, −1 to +1. +1 = lockstep, 0 = unrelated, −1 = opposite. Here: a stock vs SPY/QQQ over 60 days.'],
  drawdown: ['Drawdown', 'The % drop from a prior peak. ‘30% drawdown’ means the price is 30% below its highest point. Deep drawdowns can precede mean-reversion bounces.'],
  winrate: ['Win rate', 'Share of trades that ended profitable. 60% win rate = 6 of 10 trades made money. High win rate with small average return can still be good.'],
  avgreturn: ['Avg return', 'Average % gain/loss per trade across all trades in the backtest, from entry to exit.'],
  avghold: ['Avg hold', 'Average number of trading days a trade is held from entry signal to exit.'],
  adline: ['A/D line', 'Accumulation/Distribution line — a running total of volume weighted by where price closed in its range. Rising while price is flat/falling = quiet buying (accumulation divergence).'],
  pead: ['PEAD / drift', 'Post-Event Announcement Drift — the tendency for a stock to keep drifting in the direction of a surprise (or reverse) for days/weeks after the news, rather than fully pricing it instantly.'],
  abnreturn: ['Abnormal return', 'A stock’s move minus the move its beta says the market alone would have caused. Isolates the part of the move driven by the stock’s own news.'],
  dayabn: ['Day move (abnormal)', 'The beta-adjusted % move over the news’s reaction session (prior close → reaction close), so it captures the overnight and pre-market gap, not just the regular session.'],
  dayeffect: ['Moved that day', 'The news produced a ≥2-sigma abnormal move (vs the stock’s own recent volatility, floored at 3%). Separates news that actually moved the stock from noise.'],
  sigma: ['Sigma (σ)', 'Standard deviation — a measure of typical volatility. A ‘2-sigma move’ is twice the usual daily swing, i.e. statistically unusual.'],
  vix: ['VIX', 'The market’s ‘fear index’ — expected 30-day S&P 500 volatility. High VIX = fearful, choppy markets; low VIX = calm.'],
  regime: ['Market regime', 'The prevailing market state (e.g. bull/bear, high/low vol, risk-on/off). Signals often behave very differently across regimes.'],
  pe: ['P/E ratio', 'Price / Earnings — how many dollars you pay per dollar of annual earnings. Low P/E = cheap/value, high P/E = expensive/growth expectations.'],
  forwardpe: ['Forward P/E', 'P/E using next year’s EXPECTED earnings instead of trailing actuals. Lower forward P/E suggests earnings are expected to grow.'],
  eps: ['EPS', 'Earnings Per Share — company profit divided by shares outstanding. Negative EPS = unprofitable.'],
  revgrowth: ['Revenue growth', 'Year-over-year % change in sales. High growth (≥40%) flags hyper-growth names.'],
  margin: ['Profit margin', 'Profit as a % of revenue. Negative margin = losing money on operations; thin (0–10%) vs fat margins signal very different business quality.'],
  float: ['Float', 'Shares actually available to trade (excludes locked-up insider/institutional shares). Low float = thin supply, can move violently.'],
  marketcap: ['Market cap', 'Total value of all shares (price × shares). Micro-cap (<$500M) names are more volatile and less liquid than large-caps.'],
  obv: ['OBV', 'On-Balance Volume — a running total that adds volume on up days and subtracts it on down days. OBV rising while price lags can signal hidden buying.'],
  gap: ['Gap', 'When a stock opens far from the prior close (an overnight jump). ‘Gap down’ = opens well below yesterday’s close, often on bad news.'],
  h52: ['52-week high/low', 'The highest/lowest price over the past year. New 52-week highs signal strength/breakout; new lows signal capitulation.'],
  signal: ['Signal', 'A rule that fires an ENTRY (e.g. ‘RSI crosses above its average while oversold’). The study tests each signal paired with each exit.'],
  exit: ['Exit', 'The rule that closes a trade (e.g. ‘hold 6 months’ or ‘until RSI > 60 crosses down’). The same signal can perform very differently by exit.'],
  sentiment: ['Sentiment', 'The news provider’s polarity score (−1 to +1). Noisy here — which is why we lean on the actual same-day price move instead.'],
  horizon: ['Horizon', 'The digestion window that matters for a news TYPE — how long the market takes to fully price it. M&A resolves in a day; analyst calls over a week; product news ~1 month; earnings/guidance drift over ~3 months.'],
  fwdret: ['Results since', 'Raw total return of the stock from the news’s reaction-session close forward — 1 month (~21 trading days), 3 months (~63), 1 year (~252). Shows what actually happened after the news. Blank when the news is too recent to have the full window yet.'],
  cluster: ['News cluster', 'A burst of headlines on one ticker in a short window (each within a few days of the last). A dense cluster — especially one heavy on PR/opinion “junk” with a one-sided bullish tilt — is the footprint of a promotion or “propping” campaign.'],
  propscore: ['Prop score', 'A 0–100 heuristic for how much a cluster looks like artificial propping: combines headline volume, bullish skew, and junk/PR share, gated by an actual upward push. High score + a negative “results since” = pump-then-fade. Descriptive, not a trade signal.'],
  netmove: ['Net move', 'Sum of the per-headline β-adjusted day moves across the cluster — the net abnormal push the stock got while the news was firing.'],
  smartmoney: ['Smart money', 'Institutional & insider conviction signals from SEC filings: 13D (an ACTIVIST >5% stake with intent to influence), 13G (a PASSIVE institutional >5% stake), and insider OPEN-MARKET BUYS (officers/directors buying with their own money, Form 4). Click the badges to see every individual filing.'],
  ticker: ['Ticker', 'The stock’s exchange symbol. Click it (where underlined) to open the price chart, news, or divergence detail for that name.'],
  rank: ['Rank', 'Row position after sorting — 1 is the top-ranked item for whichever column the table is currently sorted by.'],
  price: ['Price', 'The latest close price for the stock, in dollars.'],
  lastclose: ['Last', 'The most recent daily closing price, in dollars — the reference point for “now” in this table.'],
  weight: ['Weight %', 'The stock’s weight inside the sector ETF — how much of the fund this holding represents. Bigger weights move the ETF more.'],
  spread: ['RSI spread', 'RSI(10) minus its own SMA(10). Positive = RSI is above its average (bullish momentum); the size shows how stretched. A cross from negative to positive is the entry trigger.'],
  cross: ['Crossover', 'Whether RSI(10) has crossed ABOVE its SMA(10) within the last 3 trading days, while RSI was still below 50 and Omega > 1 — the fresh mean-reversion turn the strategy buys.'],
  spy: ['SPY', 'The S&P 500 ETF — the market benchmark. Sector and stock risk metrics (Sortino, Omega, capture, beta) are all measured relative to SPY.'],
  risk: ['Risk rating', 'A composite caution flag the playbook attaches to a candidate — penalizing falling knives, low-quality micro/penny/unprofitable names, and stretched post-news pops. Higher risk = handle with more skepticism.'],
  entry: ['Entry', 'Where the trade was opened — the entry price and/or the date the entry signal fired.'],
  entrydate: ['Entry date', 'The date the entry signal fired and the (paper) position was opened.'],
  exitprice: ['Exit price', 'The price at which the trade closed (or the current price if it’s still open) — paired with entry to compute the trade’s return.'],
  exitdate: ['Exit date', 'The date the exit rule closed the trade.'],
  qty: ['Quantity', 'Number of shares in the (paper) position.'],
  side: ['Side', 'Direction of the trade — long (betting the price rises) or short (betting it falls). This system is long-biased mean-reversion.'],
  status: ['Status', 'Whether the (paper) trade is still open or has been closed by its exit rule.'],
  stance: ['Stance', 'The news-horizon read: FADE = a robustness-validated “good news fades” setup worth betting against; WATCH = informational only, not validated.'],
  strategy: ['Strategy', 'The named signal→exit combination (or rotation rule) being backtested in this row.'],
  stock: ['Stock', 'The individual company the row refers to (as opposed to a sector ETF).'],
  timeframe: ['Timeframe', 'The candle resolution the study runs on — Daily (D) unless noted Weekly (W). All risk metrics use 10-period rolling windows.'],
  trigger: ['Trigger', 'The entry rule that fires the trade — e.g. a deep-oversold RSI cross, a capitulation signal, or an A/D-divergence setup.'],
  window: ['Window', 'The span the row summarizes — for a news cluster, the first-to-last headline dates and how many days it covers; for a horizon row, days elapsed vs days remaining.'],
  mode: ['Hold mode', 'What a rotation study actually holds in each winning sector: the sector ETF, the top-momentum stock, or the highest-beta stock — mixing rotation with stock-picking.'],
  ntrades: ['Trades (n)', 'Number of historical trades the backtest fired for this signal→exit. More trades = more statistically reliable; tiny samples are noise.'],
  fired: ['Fired', 'How recently the signal last fired — “today” or the number of trading days ago. Fresh fires are the actionable ones.'],
  fires60d: ['Fires in 60d', 'How many times the signal fired in the last 60 bars. A high count while still near the lows flags a “falling knife” — repeated new lows, not a bounce.'],
  pctofflow: ['% off low', 'How far above its recent low the stock currently sits. Near 0% means it’s still right at the lows (no bounce yet); higher means it has already started recovering.'],
  peak: ['Peak return', 'The best point the trade reached during the hold (peak profit before exit) — shows how much upside was on the table vs what the exit actually captured.'],
  pnl: ['P&L', 'Profit and loss on the (paper) trade — the gain or loss from entry to exit/now, in % or dollars.'],
  median: ['Median', 'The middle value of the distribution — half the outcomes are above, half below. More robust to outliers than the average.'],
  lift: ['Lift / edge', 'How much better (or worse) the conditioned result is versus the base rate — the extra return attributable to the signal or fundamental bucket.'],
  items: ['Items', 'Number of headlines in the cluster — the size of the news burst for that ticker in the window.'],
  junk: ['Junk headlines', 'How many headlines in the cluster are opinion / clickbait / award-PR / roundups (“nothing happened”). A high junk share is a hallmark of a promotion.'],
  moved: ['Moved', 'Whether — or how many — headlines produced a ≥2-sigma abnormal price move that day. Separates real events from noise.'],
  headline: ['Headline', 'The news title. Click through (where linked) to the original source article.'],
  samplehead: ['Sample headlines', 'A few representative titles from the cluster, so you can eyeball what the burst was about at a glance.'],
  invvol: ['Inverse-vol weight', 'Position sizing that gives calmer (lower-volatility) holdings a bigger share and jumpier ones a smaller share, equalizing each position’s risk contribution.'],
  hit50: ['Hit +50%', 'Share of trades that reached a +50% gain at some point during the hold — a tail-upside hit rate, not the average.'],
  histdrift: ['Historical drift', 'The average post-news price drift historically observed for this news type / setup — the empirical basis for the fade or continuation call.'],
  histedge: ['Historical edge', 'The backtested average return of the signal at its best exit — the track record behind why it’s being surfaced now.'],
  buys: ['Insider buys', 'Dollar value of insider OPEN-MARKET purchases (Form 4, code P) — officers/directors buying with their own money. A classic conviction signal.'],
  sells: ['Insider sells', 'Dollar value of insider OPEN-MARKET sales (Form 4, code S). Insiders sell for many reasons, so sells are far weaker signal than buys.'],
  filed: ['Filed date', 'The date the filing became public on SEC EDGAR — the point-in-time when the information was disclosable.'],
  dir: ['Direction', 'Which way the move went — ▲ up (bullish) or ▼ down (bearish), from the sign of the abnormal price move.'],
  newstype: ['News type', 'The local-LLM signed read of the headline: its category plus a direction (bullish/bearish) and impact (minor/moderate/major).'],
  category: ['Category', 'The event type of the headline — earnings, guidance, analyst, M&A, partnership, contract, product, clinical, legal, offering, dividend, buyback, insider, mgmt, or macro. A ✦ marks ones refined by the local LLM.'],
  capband: ['Cap band', 'Market-cap bucket (micro / small / mid / large). Many edges — especially the news fade — concentrate in the smaller bands.'],
  bestexit: ['Best exit', 'The exit rule that produced this signal’s strongest historical result — the pairing the numbers in the row are based on.'],
  endcol: ['End', 'The end date of the window this row summarizes.'],
  notes: ['Notes', 'Free-text notes you’ve logged for this trade or decision.'],
  held: ['Held', 'What the rotation strategy actually held in each winning sector for this run (ETF, top-momentum stock, or highest-beta stock).'],
  bestfund: ['Best fundamental bucket', 'The fundamental slice (e.g. micro-cap, cheap P/E, negative EPS) where this signal’s return was strongest — where the edge concentrates. Snapshot-based, directional.'],
  window_ret: ['Window', 'How far into the news’s digestion horizon we are — trading days elapsed vs days left before the expected move is fully priced.'],
  ret1w: ['1-week return', 'The stock’s price change over the last 5 trading days (~1 week).'],
  ret1m: ['1-month return', 'The stock’s price change over the last ~21 trading days (~1 month).'],
  ret90d: ['90-day return', 'The trade’s return measured 90 days after entry — a fixed medium-term checkpoint used to compare studies on equal footing.'],
  alphacol: ['Alpha vs SPY', 'The trade’s return minus SPY’s return over the same window — the excess return not explained by simply being in the market.'],
  retcol: ['Return', 'The trade’s total return from entry to exit (or to now if still open), in %.'],
  pct52w: ['% of 52-week range', 'Where the price sits within its 1-year high–low band. Near 0% = at the lows (capitulation), near 100% = at the highs (breakout).'],
  accumdiv: ['Accumulation divergence', 'Whether the Accumulation/Distribution line is rising while price is flat or falling — quiet institutional buying that price hasn’t reflected yet.'],
  ntx: ['# transactions', 'How many individual insider transactions are aggregated into this row.'],
  best: ['Best', 'The best-performing item (sector, signal, or exit) for this grouping — the top result the row highlights.'],
  dimcol: ['Dimension', 'A fundamental slice being intersected — e.g. market-cap band, P/E band, EPS sign — to test whether amplifiers stack.'],
  study: ['Study', 'A named signal→exit strategy being backtested. The engine runs every signal against every exit and tags each trade with the regime at entry.'],
  sector: ['Sector', 'The sector/industry group the row belongs to (one of 93 sector ETFs, or the sectors a stock is a holding of).'],
  etf: ['ETF', 'The exchange-traded fund proxy for the sector — what the sector-level study actually trades.'],
  lookback: ['Lookback', 'How many months of trailing momentum the rotation ranks sectors on before deciding what to hold.'],
  holdmonths: ['Hold months', 'How long the rotation holds each pick before re-ranking and rotating.'],
  topn: ['Top N', 'How many top-ranked sectors the rotation holds at once.'],
  totalret: ['Total return', 'Cumulative return of the strategy over the full backtest period.'],
  annualret: ['Annualized return', 'The strategy’s return expressed as an equivalent per-year rate.'],
  datecol: ['Date', 'The date this row refers to — the trade, rebalance, or headline date depending on the table.'],
  bullbear: ['Bullish / bearish', 'Count of headlines in the cluster the model read as bullish vs bearish. A one-sided, all-bullish split is a promotion tell.'],
  avgdip: ['Avg dip (MAE)', 'Average maximum adverse excursion — the typical worst intraday drawdown a trade goes through after entry before it exits. A shallow avg dip means entries tend to work right away; a deep one means you usually have to sit underwater before it goes your way.'],
  cleanpct: ['Clean entry %', 'Share of trades whose worst dip never breached ~2% below entry — i.e. near-“perfect” entries that barely drew down. High = you tend to nail the timing; low = expect to ride out a drawdown first.'],
  sig: ['Significance (t-stat)', 'One-sample t-stat of the per-trade returns vs 0, computed over EFFECTIVE trades — fires within a trading week are collapsed to one independent observation, so overlapping near-duplicate trades don’t inflate the sample. |t| ≥ 2 ≈ a real edge; small |t| means the average return is statistically indistinguishable from zero no matter how large the raw trade count. Hover shows the effective (deduped) trade count. Coarse overlap proxy, not a full Newey-West correction.'],
  maxpeak: ['Max trade peak (single best)', 'The single highest intraday peak ANY ONE trade in this study reached — an order statistic (n=1), not an expectation. It is the luckiest historical outcome, NOT what a typical trade returns. Use “Avg Peak” for the expectation.'],
  p90ret: ['P90 90-day return', 'The 90th-percentile per-trade 90-day return — only ~1 trade in 10 did this well or better. A top-decile outcome, NOT an average. Compare against “Avg 90d” for the expectation.'],
};

// column-key → glossary-key for the dynamic sortable tables (Term renders label plainly if unmapped)
const TH_KEY = {
  id: 'rank', name: 'study', study_name: 'study', category: 'category', cat: 'category',
  exit_name: 'exit', total_trades: 'ntrades', trades: 'ntrades', stock_trades: 'ntrades',
  t_stat: 'sig', eff_trades: 'ntrades',
  num_trades: 'ntrades', avg_return: 'avgreturn', win_rate: 'winrate', avg_hold: 'avghold',
  avg_mae: 'avgdip', clean_pct: 'cleanpct', hist_avg_mae: 'avgdip', hist_clean_pct: 'cleanpct',
  peak_day: 'peak', peak_avg: 'peak', peak_ret: 'peak', ret_90d: 'ret90d', best_peak_ret: 'maxpeak',
  best_ret_90d: 'p90ret', entry_date: 'entrydate', sector: 'sector', etf: 'etf', entry_price: 'entry',
  exit_price: 'exitprice', exit_date: 'exitdate', return_pct: 'retcol', spy_ret: 'spy', alpha: 'alphacol',
  hold_days: 'avghold', max_drawdown: 'drawdown', stock_avg_return: 'avgreturn', etf_avg_return: 'avgreturn',
  alpha_vs_etf: 'alphacol', stock_win_rate: 'winrate', stock_max_drawdown: 'drawdown', hold_mode: 'mode',
  lookback_months: 'lookback', hold_months: 'holdmonths', top_n: 'topn', total_return: 'totalret',
  annual_return: 'annualret', spy_total: 'spy',
};

// title-based event categories (NewsItem.cat_auto) → human labels for the News Category column/filter
const CAT_LABELS = {
  earnings: 'Earnings report', guidance: 'Guidance', analyst: 'Analyst rating', ma: 'M&A',
  partnership: 'Partnership', contract: 'Contract / order', product: 'Product / launch',
  clinical: 'Clinical / FDA', legal: 'Legal', offering: 'Offering / dilution', dividend: 'Dividend',
  buyback: 'Buyback', insider: 'Insider / stake', mgmt: 'Management', macro: 'Macro', other: 'Other',
};

function Term({ k, children }) {
  const g = GLOSSARY[k];
  if (!g) return <>{children}</>;
  const [label, def] = g;
  return (
    <span className="term">{children || label}<span className="term-q" tabIndex={0} role="button" aria-label={label + ' definition'} onClick={e => e.stopPropagation()}>?<span className="term-tip"><b>{label}</b> — {def}</span></span></span>
  );
}

function Arrow({ trend, goodDir = 'up' }) {
  if (!trend || trend === 'flat') return <span className="arrow flat">-</span>;
  const isUp = trend === 'up';
  const isGood = (isUp && goodDir === 'up') || (!isUp && goodDir === 'down');
  return <span className={`arrow ${isGood ? 'good' : 'bad'}`}>{isUp ? '\u25B2' : '\u25BC'}</span>;
}

function Signal({ signal }) {
  const cls = {
    'ROTATE IN': 'signal-rotate',
    'BULLISH': 'signal-bullish',
    'RSI ONLY': 'signal-partial',
    'OMEGA ONLY': 'signal-partial',
    'BEARISH': 'signal-bearish',
  }[signal] || 'signal-bearish';
  // "ROTATE IN" is a fresh up-turn in the sector's OWN price — an absolute-trend alert, NOT a
  // beat-SPY rotation call (backtested worst beat-SPY signal). Display it as TREND TURN; the real
  // rotation call lives on the Rotation Call tab (regime-leaders ∩ value-pick ∩ oversold entry).
  const label = signal === 'ROTATE IN' ? 'TREND TURN' : signal;
  return <span className={`signal ${cls}`} title={signal === 'ROTATE IN'
    ? 'Absolute trend alert: the sector’s own price just turned up. NOT a beat-SPY signal — see the Rotation Call tab.' : undefined}>{label}</span>;
}

function Val({ val, spyVal, fmt = 3, higherBetter = true }) {
  if (val == null) return <span className="dim">-</span>;
  const better = spyVal != null ? (higherBetter ? val > spyVal : val < spyVal) : null;
  const cls = better === true ? 'good' : better === false ? 'bad' : '';
  return <span className={cls}>{val.toFixed(fmt)}</span>;
}

function Cross({ crossover, daysAgo }) {
  if (!crossover) return <span className="dim">-</span>;
  const label = daysAgo && daysAgo > 0 ? `${daysAgo}d` : 'now';
  return <span className="cross-buy">{label}</span>;
}

function SectorTable({ data, onSectorClick, onEtfClick, sortCol, sortDir, onSort }) {
  if (!data || !data.sectors) return <div className="loading">Loading...</div>;

  const sorted = [...data.sectors].sort((a, b) => {
    let av = a[sortCol], bv = b[sortCol];
    if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    // For crossover/gap: null = no event, push to bottom
    const nullBottom = sortDir === 'desc' ? -Infinity : Infinity;
    if (av == null) av = nullBottom;
    if (bv == null) bv = nullBottom;
    return sortDir === 'asc' ? av - bv : bv - av;
  });

  return (
    <div className="table-wrapper">
      <div className="spy-bar">
        SPY: Sortino <b>{data.spy_sortino?.toFixed(3)}</b> | Omega <b>{data.spy_omega?.toFixed(3)}</b> | CVaR <b>{data.spy_cvar?.toFixed(4)}</b> | Ulcer <b>{data.spy_ulcer?.toFixed(3)}</b>
      </div>
      <div className="signal-legend">
        <span className="signal-legend-title">Signal states:</span>
        <span className="signal signal-rotate">TREND TURN</span>
        <span className="signal-legend-def">BULLISH <b>+</b> RSI(10) just crossed above its SMA in the last 3 days — a <b>fresh up-turn in the sector's own price</b>. An <b>absolute-trend alert, not a beat-SPY rotation call</b> (backtests worst on beating SPY); for the real rotation call see the <b>Rotation Call</b> tab.</span>
        <span className="signal signal-bullish">BULLISH</span>
        <span className="signal-legend-def">RSI(10) &gt; SMA(10) <b>and</b> Omega(10) &gt; 1 — already trending up</span>
        <span className="signal signal-partial">RSI / OMEGA ONLY</span>
        <span className="signal-legend-def">only one of the two BULLISH conditions met</span>
        <span className="signal signal-bearish">BEARISH</span>
        <span className="signal-legend-def">neither condition met</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            {['sector','etf','pe','rsi','rsi_sma','sortino','omega','cvar','ulcer','up_capture','down_capture','beta','corr_spy','corr_qqq','beta_qqq','signal','crossover_days_ago','gap_pct'].map(key => {
              const labels = {sector:'Sector',etf:'ETF',pe:'P/E',rsi:'RSI',rsi_sma:'SMA',sortino:'Sortino',omega:'Omega',cvar:'CVaR',ulcer:'Ulcer',up_capture:'Up%',down_capture:'Dn%',beta:'Beta',corr_spy:'cSPY',corr_qqq:'cQQQ',beta_qqq:'βQQQ',signal:'Signal',crossover_days_ago:'Cr',gap_pct:'Gap'};
              const tips = {
                pe: 'Forward P/E of the sector ETF (falls back to trailing P/E) from the fundamentals feed. Lower = cheaper relative to expected earnings.',
                rsi: 'RSI(10): Relative Strength Index over 10 days. Above 50 = bullish momentum, below 50 = bearish.',
                rsi_sma: 'SMA(10) of RSI: 10-period simple moving average of the RSI. RSI crossing above this = buy signal.',
                sortino: 'Sortino Ratio (10d): Return per unit of downside risk. Higher = better risk-adjusted returns. Green = beats SPY.',
                omega: 'Omega Ratio (10d): Total gains / total losses. Above 1 = more gains than losses. Green = beats SPY.',
                cvar: 'CVaR (10d): Average loss on the worst 5% of days. Less negative = less tail risk. Green = better than SPY.',
                ulcer: 'Ulcer Index (10d): Measures depth and duration of drawdowns. Lower = less painful drops. Green = less than SPY.',
                up_capture: 'Upside Capture vs SPY (10d): On days SPY goes up, how much does this capture? Above 100 = captures more upside.',
                down_capture: 'Downside Capture vs SPY (10d): On days SPY drops, how much does this drop? Below 100 = drops less than SPY.',
                beta: 'Beta vs SPY (10d): Sensitivity to SPY moves. Below 1 = less volatile, above 1.5 = amplifies SPY moves.',
                corr_spy: 'Correlation to SPY (10d): How closely this ETF moves with SPY. Near 1 = moves with the market, near 0 = independent.',
                corr_qqq: 'Correlation to QQQ (10d): How closely this ETF moves with the Nasdaq-100. Near 1 = tracks tech/growth.',
                beta_qqq: 'Beta vs QQQ (10d): Sensitivity to QQQ moves. Above 1 = amplifies Nasdaq-100 moves.',
                rsi_sort: 'RSI of Sortino (10d): RSI(10) computed on the Sortino ratio. Shows momentum of risk-adjusted returns. Cross above SMA = improving.',
                signal: 'Signal ladder. BULLISH = RSI(10) > its SMA(10) AND Omega(10) > 1 (already trending up). TREND TURN (internally "ROTATE IN") = BULLISH AND RSI crossed above its SMA within the last 3 days, with RSI < 50 and Omega > 1 at the cross — a fresh up-turn in the sector\'s OWN price. It is an absolute-trend alert, NOT a beat-SPY rotation signal (it backtests as the WORST beat-SPY signal); the real rotation call is on the Rotation Call tab. RSI ONLY / OMEGA ONLY = just one of the two BULLISH conditions met. BEARISH = neither.',
                crossover_days_ago: 'Crossover: RSI crossed above its SMA within last 3 days, while RSI was below 50 and Omega > 1.',
                gap_pct: 'Gap Up: Open price was above previous day\'s High by at least 0.5%, within last 3 days.',
              };
              return (
                <th key={key} onClick={() => onSort(key)} className="sortable">
                  {labels[key]}{tips[key] && <span className="tip" title={tips[key]}>?</span>} {sortCol === key ? (sortDir === 'asc' ? '\u25B2' : '\u25BC') : ''}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={r.etf} className={r.bullish ? 'row-bullish' : r.signal === 'BEARISH' ? 'row-bearish' : 'row-partial'}>
              <td className="dim">{i + 1}</td>
              <td className="sector-name" onClick={() => onSectorClick(r.sector)}>{r.sector}</td>
              <td className="etf clickable" onClick={() => onEtfClick(r.etf)}>{r.etf}</td>
              <td className="dim">{r.pe != null ? r.pe.toFixed(1) : '–'}</td>
              <td className={r.rsi_above_sma ? 'good' : 'bad'}>{r.rsi?.toFixed(0)}</td>
              <td className="dim">{r.rsi_sma?.toFixed(0)}</td>
              <td><Val val={r.sortino} spyVal={data.spy_sortino} /><Arrow trend={r.sortino_trend} /></td>
              <td><Val val={r.omega} spyVal={data.spy_omega} /><Arrow trend={r.omega_trend} /></td>
              <td><Val val={r.cvar} spyVal={data.spy_cvar} fmt={4} /></td>
              <td><Val val={r.ulcer} spyVal={data.spy_ulcer} fmt={2} higherBetter={false} /><Arrow trend={r.ulcer_trend} goodDir="down" /></td>
              <td className={r.up_capture > 100 ? 'good' : 'bad'}>{r.up_capture?.toFixed(0) ?? '-'}</td>
              <td><span className={r.down_capture < 100 ? 'good' : 'bad'}>{r.down_capture?.toFixed(0) ?? '-'}</span><Arrow trend={r.down_capture_trend} goodDir="down" /></td>
              <td className={r.beta < 0 ? 'neutral' : r.beta < 1 ? 'good' : r.beta > 1.5 ? 'bad' : 'neutral'}>{r.beta?.toFixed(2) ?? '-'}</td>
              <td className={r.corr_spy == null ? 'dim' : r.corr_spy >= 0.7 ? 'good' : r.corr_spy >= 0.4 ? 'neutral' : 'bad'}>{r.corr_spy?.toFixed(2) ?? '-'}</td>
              <td className={r.corr_qqq == null ? 'dim' : r.corr_qqq >= 0.7 ? 'good' : r.corr_qqq >= 0.4 ? 'neutral' : 'bad'}>{r.corr_qqq?.toFixed(2) ?? '-'}</td>
              <td className={r.beta_qqq == null ? 'dim' : r.beta_qqq < 0 ? 'neutral' : r.beta_qqq < 1 ? 'good' : r.beta_qqq > 1.5 ? 'bad' : 'neutral'}>{r.beta_qqq?.toFixed(2) ?? '-'}</td>
              <td><Signal signal={r.signal} /></td>
              <td><Cross crossover={r.rsi_crossover} daysAgo={r.crossover_days_ago} /></td>
              <td>{r.gap ? <span className={r.gap_dir === 'up' ? 'gap-badge gap-up' : 'gap-badge gap-down'}>{r.gap_pct > 0 ? '+' : ''}{r.gap_pct}% {r.gap_days_ago === 0 ? 'today' : r.gap_days_ago + 'd'}</span> : <span className="dim">-</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function drawMetricPanel(canvas, candles, key, label, color, pad, cw, opts = {}) {
  if (!canvas) return;
  const H = 100;
  const W = canvas.parentElement.clientWidth;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  const pT = 8, pB = 20;
  const vals = candles.map(c => c[key]).filter(v => v != null);
  if (vals.length < 3) {
    ctx.fillStyle = '#7d8590'; ctx.font = '11px sans-serif';
    ctx.fillText(`${label}: no data`, pad.l + 5, H / 2);
    return;
  }

  const zeroLine = opts.zeroLine !== false;
  const vMin = Math.min(...vals, zeroLine ? 0 : Infinity);
  const vMax = Math.max(...vals, zeroLine ? 0 : -Infinity);
  const range = (vMax - vMin) || 1;
  const scY = (v) => pT + (1 - (v - vMin) / range) * (H - pT - pB);

  // Grid
  if (zeroLine) {
    const zy = scY(0);
    ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, zy); ctx.lineTo(W - pad.r, zy); ctx.stroke();
    ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText('0', pad.l - 5, zy + 4);
  }

  // Bars
  if (opts.bars !== false) {
    const zy = zeroLine ? scY(0) : H - pB;
    candles.forEach((c, i) => {
      if (c[key] == null) return;
      const x = pad.l + i * cw + cw / 2;
      const y = scY(c[key]);
      const bw = Math.max(1, cw * 0.6);
      ctx.fillStyle = c[key] > 0 ? '#3fb95050' : '#f8514950';
      if (c[key] > 0) ctx.fillRect(x - bw / 2, y, bw, zy - y);
      else ctx.fillRect(x - bw / 2, zy, bw, y - zy);
    });
  }

  // Line
  ctx.strokeStyle = color; ctx.lineWidth = 1.5;
  ctx.beginPath();
  let started = false;
  candles.forEach((c, i) => {
    if (c[key] == null) return;
    const x = pad.l + i * cw + cw / 2;
    const y = scY(c[key]);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Horizontal ref line
  if (opts.refLine != null) {
    const ry = scY(opts.refLine);
    ctx.strokeStyle = '#d2992280'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad.l, ry); ctx.lineTo(W - pad.r, ry); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#d29922'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText(String(opts.refLine), pad.l - 5, ry + 4);
  }

  // Y labels
  ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
  ctx.fillText(opts.fmt ? opts.fmt(vMax) : vMax.toFixed(1), pad.l - 5, pT + 8);
  ctx.fillText(opts.fmt ? opts.fmt(vMin) : vMin.toFixed(1), pad.l - 5, H - pB - 2);

  // Label at top
  ctx.fillStyle = color; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText(label, pad.l + 5, 14);
}

function ChartView({ ticker, onClose, embedded = false, sectorEtf = null }) {
  const priceCanvas = useRef(null);
  const rsiCanvas = useRef(null);
  const metricRefs = useRef({});
  const [chartData, setChartData] = useState(null);
  const saved = loadSettings();
  const [panels, setPanels] = useState(saved.panels);
  const [overlays, setOverlays] = useState({ ...saved.overlays, gaps: false });
  const [timeframe, setTimeframe] = useState('1d');

  useEffect(() => {
    const p = new URLSearchParams();
    if (sectorEtf) p.set('sector_etf', sectorEtf);
    if (timeframe !== '1d') p.set('interval', timeframe);
    const qs = p.toString() ? '?' + p.toString() : '';
    fetch(`${API}/chart/${ticker}${qs}`)
      .then(r => r.json())
      .then(d => setChartData(d))
      .catch(() => {});
  }, [ticker, sectorEtf, timeframe]);

  const PANELS = [
    { key: 'sortino', label: 'Sortino (own)', color: '#a371f7', refLine: null },
    { key: 'omega', label: 'Omega (own)', color: '#f0883e', refLine: 1 },
    { key: 'cvar', label: 'CVaR (own)', color: '#f85149', refLine: null },
    { key: 'ulcer', label: 'Ulcer Index (own)', color: '#d29922', zeroLine: false, bars: false },
    { key: 'beta', label: 'Beta vs SPY', color: '#79c0ff', zeroLine: false, bars: false, refLine: 1 },
    { key: 'rsi_sort', label: 'RSI of Sortino', color: '#e879f9', zeroLine: false, bars: false },
    { key: 'capture', label: 'Up/Dn Capture vs SPY', color: '#3fb950' },
  ];
  const togglePanel = (key) => setPanels(p => ({ ...p, [key]: !p[key] }));
  const toggleOverlay = (key) => setOverlays(p => ({ ...p, [key]: !p[key] }));

  const drawChart = useCallback(() => {
    if (!chartData || !priceCanvas.current || !rsiCanvas.current) return;
    const candles = chartData.candles.slice(-120);
    const W = priceCanvas.current.parentElement.clientWidth;
    const priceH = 320;
    const rsiH = 140;
    priceCanvas.current.width = W;
    priceCanvas.current.height = priceH;
    rsiCanvas.current.width = W;
    rsiCanvas.current.height = rsiH;

    const stockPrices = [...candles.map(c => c.high), ...candles.map(c => c.low)];
    const pMin = Math.min(...stockPrices) * 0.998;
    const pMax = Math.max(...stockPrices) * 1.002;
    const hasRightAxis = overlays.spy || (overlays.sector && chartData.sector_etf);
    const padR = hasRightAxis ? 65 : 20;
    const pad = { l: 60, r: padR, t: 20, b: 25 };
    const cw = (W - pad.l - pad.r) / candles.length;
    const scaleY = (v) => pad.t + (1 - (v - pMin) / (pMax - pMin)) * (priceH - pad.t - pad.b);

    // SPY separate scale
    const spyPrices = candles.map(c => c.spy_price).filter(v => v != null);
    const spyMin = spyPrices.length ? Math.min(...spyPrices) * 0.998 : 0;
    const spyMax = spyPrices.length ? Math.max(...spyPrices) * 1.002 : 1;
    const scaleSPY = (v) => pad.t + (1 - (v - spyMin) / ((spyMax - spyMin) || 1)) * (priceH - pad.t - pad.b);

    // Sector separate scale
    const sectPrices = candles.map(c => c.sect_price).filter(v => v != null);
    const sectMin = sectPrices.length ? Math.min(...sectPrices) * 0.998 : 0;
    const sectMax = sectPrices.length ? Math.max(...sectPrices) * 1.002 : 1;
    const scaleSect = (v) => pad.t + (1 - (v - sectMin) / ((sectMax - sectMin) || 1)) * (priceH - pad.t - pad.b);

    const ctx = priceCanvas.current.getContext('2d');
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, W, priceH);

    ctx.strokeStyle = '#1e2430'; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = pad.t + i * (priceH - pad.t - pad.b) / 4;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      const price = pMax - i * (pMax - pMin) / 4;
      ctx.fillStyle = '#7d8590'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
      ctx.fillText(price.toFixed(2), pad.l - 5, y + 4);
    }

    candles.forEach((c, i) => {
      const x = pad.l + i * cw + cw / 2;
      const green = c.close >= c.open;
      ctx.fillStyle = green ? '#3fb950' : '#f85149';
      ctx.strokeStyle = green ? '#3fb950' : '#f85149';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, scaleY(c.high)); ctx.lineTo(x, scaleY(c.low)); ctx.stroke();
      const bodyTop = scaleY(Math.max(c.open, c.close));
      const bodyBot = scaleY(Math.min(c.open, c.close));
      ctx.fillRect(x - cw * 0.35, bodyTop, cw * 0.7, Math.max(1, bodyBot - bodyTop));
    });

    // SPY overlay line (own scale + right axis)
    if (overlays.spy && spyPrices.length > 2) {
      ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.beginPath(); let s1 = false;
      candles.forEach((c, i) => { if (c.spy_price == null) return; const x = pad.l + i * cw + cw / 2, y = scaleSPY(c.spy_price); if (!s1) { ctx.moveTo(x, y); s1 = true; } else ctx.lineTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#58a6ff'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left'; ctx.fillText('SPY', pad.l + 5, pad.t + 14);
      // Right axis for SPY
      ctx.fillStyle = '#58a6ff'; ctx.font = '10px monospace'; ctx.textAlign = 'left';
      for (let i = 0; i < 5; i++) {
        const y = pad.t + i * (priceH - pad.t - pad.b) / 4;
        const price = spyMax - i * (spyMax - spyMin) / 4;
        ctx.fillText(price.toFixed(0), W - pad.r + 5, y + 4);
      }
    }
    // Sector ETF overlay line (own scale)
    if (overlays.sector && chartData.sector_etf && sectPrices.length > 2) {
      ctx.strokeStyle = '#d29922'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.beginPath(); let s2 = false;
      candles.forEach((c, i) => { if (c.sect_price == null) return; const x = pad.l + i * cw + cw / 2, y = scaleSect(c.sect_price); if (!s2) { ctx.moveTo(x, y); s2 = true; } else ctx.lineTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#d29922'; ctx.font = '10px sans-serif'; ctx.textAlign = 'left'; ctx.fillText(chartData.sector_etf, pad.l + 35, pad.t + 14);
    }

    // Gap markers
    if (overlays.gaps) {
      ctx.font = 'bold 9px sans-serif'; ctx.textAlign = 'center';
      candles.forEach((c, i) => {
        if (c.gap == null) return;
        const x = pad.l + i * cw + cw / 2;
        const isUp = c.gap > 0;
        const y = isUp ? scaleY(c.high) - 12 : scaleY(c.low) + 14;
        ctx.fillStyle = isUp ? '#3fb950' : '#f85149';
        const label = (isUp ? '+' : '') + c.gap.toFixed(1) + '%';
        // Background pill
        const tw = ctx.measureText(label).width + 6;
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = isUp ? '#1a3a1a' : '#3a1a1a';
        ctx.fillRect(x - tw / 2, y - 8, tw, 12);
        ctx.globalAlpha = 1;
        ctx.fillStyle = isUp ? '#3fb950' : '#f85149';
        ctx.fillText(label, x, y + 1);
      });
    }

    ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(candles.length / 6));
    candles.forEach((c, i) => { if (i % step === 0) ctx.fillText(c.date.slice(5), pad.l + i * cw + cw / 2, priceH - 5); });

    // RSI
    const rctx = rsiCanvas.current.getContext('2d');
    rctx.fillStyle = '#0d1117'; rctx.fillRect(0, 0, W, rsiH);
    const scRSI = (v) => 10 + (1 - v / 100) * (rsiH - 35);
    [30, 50, 70].forEach(lv => {
      const y = scRSI(lv);
      rctx.strokeStyle = lv === 50 ? '#30363d' : '#1e2430'; rctx.lineWidth = 1;
      rctx.setLineDash(lv === 50 ? [] : [4, 4]);
      rctx.beginPath(); rctx.moveTo(pad.l, y); rctx.lineTo(W - pad.r, y); rctx.stroke();
      rctx.setLineDash([]);
      rctx.fillStyle = '#7d8590'; rctx.font = '10px monospace'; rctx.textAlign = 'right';
      rctx.fillText(lv, pad.l - 5, y + 4);
    });
    [['rsi', '#58a6ff'], ['rsi_sma', '#d29922']].forEach(([k, col]) => {
      rctx.strokeStyle = col; rctx.lineWidth = 1.5; rctx.beginPath(); let s = false;
      candles.forEach((c, i) => { if (c[k] == null) return; const x = pad.l + i * cw + cw / 2, y = scRSI(c[k]); if (!s) { rctx.moveTo(x, y); s = true; } else rctx.lineTo(x, y); });
      rctx.stroke();
    });
    rctx.font = '11px sans-serif'; rctx.textAlign = 'left';
    rctx.fillStyle = '#58a6ff'; rctx.fillText('RSI(10)', pad.l + 5, 14);
    rctx.fillStyle = '#d29922'; rctx.fillText('SMA(10)', pad.l + 65, 14);

    // Metric panels
    PANELS.forEach(p => {
      const ref = metricRefs.current[p.key];
      if (!panels[p.key] || !ref) return;

      if (p.key === 'rsi_sort') {
        // Special: RSI of Sortino with SMA overlay + 30/50/70 lines
        const H = 120; ref.width = W; ref.height = H;
        const mctx = ref.getContext('2d');
        mctx.fillStyle = '#0d1117'; mctx.fillRect(0, 0, W, H);
        const scRS = (v) => 10 + (1 - v / 100) * (H - 35);
        [30, 50, 70].forEach(lv => {
          const y = scRS(lv);
          mctx.strokeStyle = lv === 50 ? '#30363d' : '#1e2430'; mctx.lineWidth = 1;
          mctx.setLineDash(lv === 50 ? [] : [4, 4]);
          mctx.beginPath(); mctx.moveTo(pad.l, y); mctx.lineTo(W - pad.r, y); mctx.stroke();
          mctx.setLineDash([]);
          mctx.fillStyle = '#7d8590'; mctx.font = '10px monospace'; mctx.textAlign = 'right';
          mctx.fillText(lv, pad.l - 5, y + 4);
        });
        [['rsi_sort', '#e879f9'], ['rsi_sort_sma', '#9333ea']].forEach(([k, col]) => {
          mctx.strokeStyle = col; mctx.lineWidth = 1.5; mctx.beginPath(); let s = false;
          candles.forEach((c, i) => { if (c[k] == null) return; const x = pad.l + i * cw + cw / 2, y = scRS(c[k]); if (!s) { mctx.moveTo(x, y); s = true; } else mctx.lineTo(x, y); });
          mctx.stroke();
        });
        mctx.font = '11px sans-serif'; mctx.textAlign = 'left';
        mctx.fillStyle = '#e879f9'; mctx.fillText('RSI of Sortino', pad.l + 5, 14);
        mctx.fillStyle = '#9333ea'; mctx.fillText('SMA', pad.l + 110, 14);
      } else if (p.key === 'capture') {
        // Special: draw two lines (up + dn capture)
        const H = 100; ref.width = W; ref.height = H;
        const mctx = ref.getContext('2d');
        mctx.fillStyle = '#0d1117'; mctx.fillRect(0, 0, W, H);
        const uVals = candles.map(c => c.up_capture).filter(v => v != null);
        const dVals = candles.map(c => c.dn_capture).filter(v => v != null);
        const all = [...uVals, ...dVals, 100];
        if (all.length < 3) return;
        const mn = Math.min(...all), mx = Math.max(...all), rng = (mx - mn) || 1;
        const sc = (v) => 8 + (1 - (v - mn) / rng) * (H - 33);
        // 100% ref line
        const ry = sc(100);
        mctx.strokeStyle = '#30363d'; mctx.lineWidth = 1;
        mctx.beginPath(); mctx.moveTo(pad.l, ry); mctx.lineTo(W - pad.r, ry); mctx.stroke();
        mctx.fillStyle = '#7d8590'; mctx.font = '10px monospace'; mctx.textAlign = 'right';
        mctx.fillText('100', pad.l - 5, ry + 4);
        [['up_capture', '#3fb950'], ['dn_capture', '#f85149']].forEach(([k, col]) => {
          mctx.strokeStyle = col; mctx.lineWidth = 1.5; mctx.beginPath(); let s = false;
          candles.forEach((c, i) => { if (c[k] == null) return; const x = pad.l + i * cw + cw / 2, y = sc(c[k]); if (!s) { mctx.moveTo(x, y); s = true; } else mctx.lineTo(x, y); });
          mctx.stroke();
        });
        // Capture Spread line (Up% - Dn%) — shows the edge
        const spreadVals = candles.filter(c => c.up_capture != null && c.dn_capture != null);
        if (spreadVals.length > 2) {
          const spAll = spreadVals.map(c => c.up_capture - c.dn_capture);
          const spMn = Math.min(...spAll, 0), spMx = Math.max(...spAll, 0), spRng = (spMx - spMn) || 1;
          // Draw spread as filled area on a separate mini section at bottom
          mctx.strokeStyle = '#58a6ff'; mctx.lineWidth = 2; mctx.setLineDash([3, 3]);
          mctx.beginPath(); let s2 = false;
          candles.forEach((c, i) => {
            if (c.up_capture == null || c.dn_capture == null) return;
            const spread = c.up_capture - c.dn_capture;
            const x = pad.l + i * cw + cw / 2;
            const y = sc(spread + 100); // offset to fit in same scale roughly
            if (!s2) { mctx.moveTo(x, y); s2 = true; } else mctx.lineTo(x, y);
          });
          mctx.stroke(); mctx.setLineDash([]);
        }
        mctx.font = '11px sans-serif'; mctx.textAlign = 'left';
        mctx.fillStyle = '#3fb950'; mctx.fillText('Up%', pad.l + 5, 14);
        mctx.fillStyle = '#f85149'; mctx.fillText('Dn%', pad.l + 40, 14);
        mctx.fillStyle = '#58a6ff'; mctx.fillText('Spread (Up-Dn)', pad.l + 75, 14);
      } else {
        drawMetricPanel(ref, candles, p.key, p.label, p.color, pad, cw, {
          zeroLine: p.zeroLine, bars: p.bars, refLine: p.refLine,
        });
        // Draw SPY + sector comparison lines on sortino and omega
        if ((p.key === 'sortino' || p.key === 'omega') && ref) {
          const H = 100;
          const mctx = ref.getContext('2d');
          const vals = candles.map(c => c[p.key]).filter(v => v != null);
          const spyKey = 'spy_' + p.key;
          const sectKey = 'sect_' + p.key;
          const spyVals = candles.map(c => c[spyKey]).filter(v => v != null);
          const sectVals = candles.map(c => c[sectKey]).filter(v => v != null);
          const allV = [...vals, ...spyVals, ...sectVals, 0];
          if (allV.length < 3) return;
          const mn = Math.min(...allV), mx = Math.max(...allV), rng = (mx - mn) || 1;
          const sc2 = (v) => 8 + (1 - (v - mn) / rng) * (H - 33);
          if (overlays.spy_sortino && spyVals.length > 2) {
            mctx.strokeStyle = '#58a6ff80'; mctx.lineWidth = 1; mctx.setLineDash([3, 3]); mctx.beginPath(); let s = false;
            candles.forEach((c, i) => { if (c[spyKey] == null) return; const x = pad.l + i * cw + cw / 2, y = sc2(c[spyKey]); if (!s) { mctx.moveTo(x, y); s = true; } else mctx.lineTo(x, y); });
            mctx.stroke(); mctx.setLineDash([]);
            mctx.fillStyle = '#58a6ff'; mctx.font = '10px sans-serif'; mctx.fillText('SPY', W - pad.r - 25, 14);
          }
          if (overlays.sector && sectVals.length > 2 && chartData.sector_etf) {
            mctx.strokeStyle = '#d2992280'; mctx.lineWidth = 1; mctx.setLineDash([3, 3]); mctx.beginPath(); let s = false;
            candles.forEach((c, i) => { if (c[sectKey] == null) return; const x = pad.l + i * cw + cw / 2, y = sc2(c[sectKey]); if (!s) { mctx.moveTo(x, y); s = true; } else mctx.lineTo(x, y); });
            mctx.stroke(); mctx.setLineDash([]);
            mctx.fillStyle = '#d29922'; mctx.font = '10px sans-serif'; mctx.fillText(chartData.sector_etf, W - pad.r - 60, 14);
          }
        }
      }
    });
  }, [chartData, panels, overlays]);

  useEffect(() => { drawChart(); }, [drawChart]);

  if (!chartData) return <div className="loading">Loading chart for {ticker}...</div>;

  const last = chartData.candles[chartData.candles.length - 1];

  return (
    <div className={embedded ? "chart-embedded" : "chart-modal"}>
      <div className="chart-header">
        <h3>
          {ticker} — ${last?.close}
          <select className="tf-select" value={timeframe} onChange={e => setTimeframe(e.target.value)}>
            <option value="1d">Daily</option>
            <option value="1wk">Weekly</option>
            <option value="1mo">Monthly</option>
          </select>
        </h3>
        <div className="panel-toggles">
          {PANELS.map(p => {
            const tips = {
              sortino: 'Return per unit of downside risk over 10 days. Higher = better.',
              omega: 'Total gains / total losses over 10 days. Above 1 = more gain than pain.',
              cvar: 'Average loss on worst 5% of days. Less negative = less tail risk.',
              ulcer: 'Depth and duration of drawdowns. Lower = less painful.',
              beta: 'Sensitivity to SPY. Below 1 = less volatile than SPY.',
              rsi_sort: 'RSI(10) computed on the Sortino ratio. Crossover = risk-adjusted momentum turning.',
              capture: 'Upside vs Downside capture relative to SPY. Spread = the edge.',
            };
            return (
              <label key={p.key} className="toggle-label" title={tips[p.key]}>
                <input type="checkbox" checked={panels[p.key]} onChange={() => togglePanel(p.key)} />
                <span style={{ color: p.color }}>{p.label}</span>
              </label>
            );
          })}
        </div>
        <div className="panel-toggles">
          <label className="toggle-label" title="Show SPY normalized price on candle chart">
            <input type="checkbox" checked={overlays.spy} onChange={() => toggleOverlay('spy')} />
            <span style={{color:'#58a6ff'}}>SPY Price</span>
          </label>
          {chartData?.sector_etf && <label className="toggle-label" title="Show sector ETF normalized price on candle chart">
            <input type="checkbox" checked={overlays.sector} onChange={() => toggleOverlay('sector')} />
            <span style={{color:'#d29922'}}>{chartData.sector_etf} Price</span>
          </label>}
          <label className="toggle-label" title="Show SPY Sortino/Omega on metric panels">
            <input type="checkbox" checked={overlays.spy_sortino} onChange={() => toggleOverlay('spy_sortino')} />
            <span style={{color:'#58a6ff80'}}>SPY Metrics</span>
          </label>
          <label className="toggle-label" title="Show gap up/down percentages on the price chart">
            <input type="checkbox" checked={overlays.gaps} onChange={() => toggleOverlay('gaps')} />
            <span style={{color:'#d29922'}}>Gaps</span>
          </label>
        </div>
        {!embedded && <button className="close-btn" onClick={onClose}>Close</button>}
      </div>
      {chartData && (() => {
        const c = chartData.candles;
        const l = c[c.length - 1] || {};
        const rsiAbove = l.rsi > l.rsi_sma;
        const rsAbove = l.rsi_sort > l.rsi_sort_sma;
        return (
          <div className="indicator-bar">
            <span>RSI: <b className={rsiAbove ? 'good' : 'bad'}>{l.rsi?.toFixed(0) || '-'}</b>/<span className="dim">{l.rsi_sma?.toFixed(0) || '-'}</span></span>
            <span>Sortino: <b className={l.sortino > 0 ? 'good' : 'bad'}>{l.sortino?.toFixed(2) || '-'}</b></span>
            <span>Omega: <b className={l.omega > 1 ? 'good' : 'bad'}>{l.omega?.toFixed(2) || '-'}</b></span>
            <span>Beta: <b className={l.beta < 1 ? 'good' : l.beta > 1.5 ? 'bad' : 'neutral'}>{l.beta?.toFixed(2) || '-'}</b></span>
            {l.rsi_sort != null && <span>RS: <b className={rsAbove ? 'good' : 'bad'}>{l.rsi_sort?.toFixed(0)}</b>/<span className="dim">{l.rsi_sort_sma?.toFixed(0)}</span></span>}
            {l.ulcer != null && <span>Ulcer: <b>{l.ulcer?.toFixed(1)}</b></span>}
            {l.up_capture != null && <span>Up%: <b className={l.up_capture > 100 ? 'good' : 'bad'}>{l.up_capture?.toFixed(0)}</b></span>}
            {l.dn_capture != null && <span>Dn%: <b className={l.dn_capture < 100 ? 'good' : 'bad'}>{l.dn_capture?.toFixed(0)}</b></span>}
          </div>
        );
      })()}
      <div className="chart-container">
        <canvas ref={priceCanvas} />
        <canvas ref={rsiCanvas} />
        {PANELS.map(p => panels[p.key] ? <canvas key={p.key} ref={el => metricRefs.current[p.key] = el} /> : null)}
      </div>
    </div>
  );
}

function StockTable({ sector, data, onBack, onTickerClick }) {
  const sort = useSortedRows(data && data.stocks, null, 'desc', {
    cross: r => r.crossover_days_ago,
    gap: r => r.gap_pct,
  });
  if (!data) return <div className="loading">Loading stocks for {sector}...</div>;
  if (!data.stocks || data.stocks.length === 0) return null;
  return (
    <div className="table-wrapper">
      <button className="back-btn" onClick={onBack}>Back to Sectors</button>
      <h2>{data.sector} ({data.etf})</h2>
      <ChartView ticker={data.etf} onClose={() => {}} embedded sectorEtf={null} />
      <table>
        <thead>
          <tr>
            <th><Term k="rank">#</Term></th><SortTh colKey="ticker" sort={sort}><Term k="ticker">Ticker</Term></SortTh><SortTh colKey="weight" sort={sort}><Term k="weight">Wt%</Term></SortTh><SortTh colKey="price" sort={sort}><Term k="price">Price</Term></SortTh><SortTh colKey="return_1w" sort={sort}><Term k="ret1w">1W</Term></SortTh><SortTh colKey="return_1m" sort={sort}><Term k="ret1m">1M</Term></SortTh>
            <SortTh colKey="rsi" sort={sort}><Term k="rsi">RSI</Term></SortTh><SortTh colKey="rsi_sma" sort={sort}><Term k="sma">SMA</Term></SortTh><SortTh colKey="rsi_spread" sort={sort}><Term k="spread">Spread</Term></SortTh><SortTh colKey="sortino" sort={sort}><Term k="sortino">Sortino</Term></SortTh><SortTh colKey="beta" sort={sort}><Term k="beta">Beta</Term></SortTh><SortTh colKey="cross" sort={sort}><Term k="cross">Cross</Term></SortTh><SortTh colKey="gap" sort={sort}><Term k="gap">Gap</Term></SortTh><SortTh colKey="signal" sort={sort}><Term k="signal">Signal</Term></SortTh>
          </tr>
        </thead>
        <tbody>
          {sort.rows.map((s, i) => (
            <tr key={s.ticker} className={s.rsi_above_sma ? 'row-bullish' : 'row-bearish'}>
              <td className="dim">{i + 1}</td>
              <td className="ticker clickable" onClick={() => onTickerClick(s.ticker, data.etf)}>{s.ticker}</td>
              <td className="dim">{s.weight != null ? s.weight.toFixed(1) + '%' : '-'}</td>
              <td>${s.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
              <td className={s.return_1w > 0 ? 'good' : 'bad'}>{s.return_1w > 0 ? '+' : ''}{s.return_1w}%</td>
              <td className={s.return_1m > 0 ? 'good' : 'bad'}>{s.return_1m > 0 ? '+' : ''}{s.return_1m}%</td>
              <td className={s.rsi_above_sma ? 'good' : 'bad'}>{s.rsi?.toFixed(1)}</td>
              <td className="dim">{s.rsi_sma?.toFixed(1)}</td>
              <td className={s.rsi_spread > 0 ? 'good' : 'bad'}>{s.rsi_spread > 0 ? '+' : ''}{s.rsi_spread.toFixed(1)}</td>
              <td className={s.sortino > 0 ? 'good' : 'bad'}>{s.sortino != null ? s.sortino.toFixed(3) : '-'}</td>
              <td className={s.beta != null ? (s.beta < 0 ? 'neutral' : s.beta < 1 ? 'good' : s.beta > 1.5 ? 'bad' : 'neutral') : 'dim'}>{s.beta != null ? s.beta.toFixed(2) : '-'}</td>
              <td><Cross crossover={s.rsi_crossover} daysAgo={s.crossover_days_ago} /></td>
              <td>{s.gap ? <span className={s.gap_dir === 'up' ? 'gap-badge gap-up' : 'gap-badge gap-down'}>{s.gap_pct > 0 ? '+' : ''}{s.gap_pct}% {s.gap_days_ago === 0 ? 'today' : s.gap_days_ago + 'd'}</span> : <span className="dim">-</span>}</td>
              <td><Signal signal={s.signal} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const DEFAULT_SETTINGS = {
  panels: { sortino: true, omega: true, cvar: false, ulcer: false, beta: false, rsi_sort: false, capture: false },
  overlays: { spy: true, sector: true, spy_sortino: true },
};

function loadSettings() {
  try {
    const saved = localStorage.getItem('chart_settings');
    if (saved) return JSON.parse(saved);
  } catch {}
  return DEFAULT_SETTINGS;
}

function saveSettings(s) {
  localStorage.setItem('chart_settings', JSON.stringify(s));
}

function SettingsPage() {
  const [settings, setSettings] = useState(loadSettings);

  const togglePanel = (key) => {
    setSettings(prev => {
      const next = { ...prev, panels: { ...prev.panels, [key]: !prev.panels[key] } };
      saveSettings(next);
      return next;
    });
  };

  const toggleOverlay = (key) => {
    setSettings(prev => {
      const next = { ...prev, overlays: { ...prev.overlays, [key]: !prev.overlays[key] } };
      saveSettings(next);
      return next;
    });
  };

  const panelList = [
    { key: 'sortino', label: 'Sortino (own)', color: '#a371f7', desc: 'Return per unit of downside risk over 10 days.' },
    { key: 'omega', label: 'Omega (own)', color: '#f0883e', desc: 'Total gains / total losses. Above 1 = more gain than pain.' },
    { key: 'cvar', label: 'CVaR (own)', color: '#f85149', desc: 'Average loss on worst 5% of days.' },
    { key: 'ulcer', label: 'Ulcer Index (own)', color: '#d29922', desc: 'Depth and duration of drawdowns. Lower = better.' },
    { key: 'beta', label: 'Beta vs SPY', color: '#79c0ff', desc: 'Sensitivity to SPY. Below 1 = less volatile.' },
    { key: 'rsi_sort', label: 'RSI of Sortino', color: '#e879f9', desc: 'RSI(10) on the Sortino ratio. Momentum of risk-adjusted returns.' },
    { key: 'capture', label: 'Up/Dn Capture vs SPY', color: '#3fb950', desc: 'Upside and downside capture relative to SPY.' },
  ];

  const overlayList = [
    { key: 'spy', label: 'SPY Price', color: '#58a6ff', desc: 'Normalized SPY price overlay on candle chart.' },
    { key: 'sector', label: 'Sector ETF Price', color: '#d29922', desc: 'Normalized sector ETF price overlay (stocks only).' },
    { key: 'spy_sortino', label: 'SPY Metrics', color: '#58a6ff', desc: 'SPY Sortino/Omega lines on metric panels.' },
  ];

  return (
    <div className="settings-page">
      <h1>Chart Settings</h1>
      <p className="subtitle">Choose which indicators are enabled by default when opening a chart.</p>

      <div className="settings-section">
        <h2>Indicator Panels</h2>
        <div className="settings-grid">
          {panelList.map(p => (
            <div key={p.key} className={`settings-card ${settings.panels[p.key] ? 'enabled' : ''}`} onClick={() => togglePanel(p.key)}>
              <div className="settings-card-header">
                <input type="checkbox" checked={settings.panels[p.key]} onChange={() => togglePanel(p.key)} />
                <span className="settings-card-label" style={{ color: p.color }}>{p.label}</span>
              </div>
              <p className="settings-card-desc">{p.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="settings-section">
        <h2>Price Overlays</h2>
        <div className="settings-grid">
          {overlayList.map(p => (
            <div key={p.key} className={`settings-card ${settings.overlays[p.key] ? 'enabled' : ''}`} onClick={() => toggleOverlay(p.key)}>
              <div className="settings-card-header">
                <input type="checkbox" checked={settings.overlays[p.key]} onChange={() => toggleOverlay(p.key)} />
                <span className="settings-card-label" style={{ color: p.color }}>{p.label}</span>
              </div>
              <p className="settings-card-desc">{p.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function drawIndicatorPanel(canvas, candles, key, label, color, refLines, cw, pad) {
  if (!canvas) return;
  const vals = candles.map(c => c[key]).filter(v => v != null);
  if (vals.length === 0) return;
  const H = 100;
  const W = canvas.parentElement.clientWidth;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);

  const vMin = Math.min(...vals, ...(refLines || []));
  const vMax = Math.max(...vals, ...(refLines || []));
  const range = vMax - vMin || 1;
  const scY = v => 8 + (1 - (v - vMin) / range) * (H - 20);

  // Grid
  ctx.strokeStyle = '#1e2430'; ctx.lineWidth = 1;
  (refLines || []).forEach(rv => {
    const y = scY(rv);
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#7d8590'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
    ctx.fillText(String(rv), pad.l - 4, y + 3);
  });

  // Line
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash([]);
  ctx.beginPath();
  let started = false;
  candles.forEach((c, i) => {
    if (c[key] == null) return;
    const x = pad.l + i * cw + cw / 2;
    if (!started) { ctx.moveTo(x, scY(c[key])); started = true; }
    else ctx.lineTo(x, scY(c[key]));
  });
  ctx.stroke();

  // Label
  ctx.fillStyle = color; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText(label, pad.l + 4, 14);

  // Current value
  const last = candles.filter(c => c[key] != null).slice(-1)[0];
  if (last) {
    ctx.fillStyle = '#e6edf3'; ctx.font = '11px monospace'; ctx.textAlign = 'left';
    ctx.fillText(last[key].toFixed(2), pad.l + 4 + ctx.measureText(label).width + 10, 14);
  }
}

function drawDualLinePanel(canvas, candles, key1, key2, label1, label2, color1, color2, refLines, cw, pad) {
  if (!canvas) return;
  const vals = [...candles.map(c => c[key1]).filter(v => v != null), ...candles.map(c => c[key2]).filter(v => v != null)];
  if (vals.length === 0) return;
  const H = 100;
  const W = canvas.parentElement.clientWidth;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);

  const vMin = Math.min(...vals, ...(refLines || []));
  const vMax = Math.max(...vals, ...(refLines || []));
  const range = vMax - vMin || 1;
  const scY = v => 8 + (1 - (v - vMin) / range) * (H - 20);

  (refLines || []).forEach(rv => {
    const y = scY(rv);
    ctx.strokeStyle = '#1e2430'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#7d8590'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
    ctx.fillText(String(rv), pad.l - 4, y + 3);
  });

  [{ k: key1, c: color1 }, { k: key2, c: color2 }].forEach(({ k, c }) => {
    ctx.strokeStyle = c; ctx.lineWidth = 1.5; ctx.setLineDash([]); ctx.beginPath();
    let started = false;
    candles.forEach((ci, i) => {
      if (ci[k] == null) return;
      const x = pad.l + i * cw + cw / 2;
      if (!started) { ctx.moveTo(x, scY(ci[k])); started = true; } else ctx.lineTo(x, scY(ci[k]));
    });
    ctx.stroke();
  });

  ctx.fillStyle = color1; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'left';
  ctx.fillText(label1, pad.l + 4, 14);
  ctx.fillStyle = color2;
  ctx.fillText(label2, pad.l + 4 + ctx.measureText(label1).width + 14, 14);
}

function StudyChartView({ study, onBack }) {
  const [sectorTrades, setSectorTrades] = useState(null);
  const [chartData, setChartData] = useState(null);
  const sector = study.chartSector;
  const priceCanvas = useRef(null);
  const rsiCanvas = useRef(null);
  const sortinoCanvas = useRef(null);
  const omegaCanvas = useRef(null);
  const rsiSortCanvas = useRef(null);
  const cvarCanvas = useRef(null);
  const ulcerCanvas = useRef(null);
  const betaCanvas = useRef(null);
  const captureCanvas = useRef(null);

  // Load trades for this study (filtered to this sector)
  useEffect(() => {
    fetch(`${API}/studies/${study.id}/trades?signal=${encodeURIComponent(study.signal)}&exit=${encodeURIComponent(study.exit)}&sector=${encodeURIComponent(sector)}`)
      .then(r => r.json())
      .then(d => setSectorTrades(d))
      .catch(() => {});
  }, [study.id, study.signal, study.exit, sector]);

  // Load chart data with indicators
  useEffect(() => {
    if (!sector) return;
    fetch(`${API}/chart/${sector}`)
      .then(r => r.json())
      .then(d => setChartData(d))
      .catch(() => {});
  }, [sector]);

  // Draw price chart with buy/sell markers
  const drawChart = useCallback(() => {
    if (!chartData || !priceCanvas.current) return;
    const candles = chartData.candles;
    const W = priceCanvas.current.parentElement.clientWidth;
    const H = 400;
    priceCanvas.current.width = W;
    priceCanvas.current.height = H;
    const pad = { l: 60, r: 20, t: 30, b: 25 };
    const cw = (W - pad.l - pad.r) / candles.length;

    const prices = [...candles.map(c => c.high), ...candles.map(c => c.low)];
    const pMin = Math.min(...prices) * 0.995;
    const pMax = Math.max(...prices) * 1.005;
    const scY = (v) => pad.t + (1 - (v - pMin) / (pMax - pMin)) * (H - pad.t - pad.b);

    const ctx = priceCanvas.current.getContext('2d');
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#1e2430'; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = pad.t + i * (H - pad.t - pad.b) / 4;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      const price = pMax - i * (pMax - pMin) / 4;
      ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
      ctx.fillText(price.toFixed(2), pad.l - 5, y + 4);
    }

    // Candles
    candles.forEach((c, i) => {
      const x = pad.l + i * cw + cw / 2;
      const green = c.close >= c.open;
      ctx.fillStyle = green ? '#3fb950' : '#f85149';
      ctx.strokeStyle = green ? '#3fb950' : '#f85149';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, scY(c.high)); ctx.lineTo(x, scY(c.low)); ctx.stroke();
      const bt = scY(Math.max(c.open, c.close));
      const bb = scY(Math.min(c.open, c.close));
      ctx.fillRect(x - cw * 0.35, bt, cw * 0.7, Math.max(1, bb - bt));
    });

    // Date labels
    ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(candles.length / 8));
    candles.forEach((c, i) => { if (i % step === 0) ctx.fillText(c.date.slice(2, 7), pad.l + i * cw + cw / 2, H - 5); });

    // Build date->index map
    const dateMap = {};
    candles.forEach((c, i) => { dateMap[c.date] = i; });

    // Draw buy/sell markers (non-overlapping only)
    const allTrades = (sectorTrades?.trades || []).filter(t => (t.etf || t.sector) === sector);
    allTrades.sort((a, b) => (a.entry_date || '').localeCompare(b.entry_date || ''));
    const trades = [];
    let lastExit = null;
    allTrades.forEach(t => {
      const eIdx = dateMap[t.entry_date];
      const xIdx = dateMap[t.exit_date];
      if (eIdx == null) return;
      if (lastExit != null && eIdx <= lastExit) return; // skip overlapping
      trades.push(t);
      if (xIdx != null) lastExit = xIdx;
    });
    trades.forEach(t => {
      const entryIdx = dateMap[t.entry_date];
      const exitIdx = dateMap[t.exit_date];
      const ret = t.return ?? t.return_pct ?? 0;

      if (entryIdx != null) {
        const x = pad.l + entryIdx * cw + cw / 2;
        const y = scY(candles[entryIdx].low) + 12;
        ctx.fillStyle = '#3fb950';
        ctx.beginPath();
        ctx.moveTo(x, y); ctx.lineTo(x - 5, y + 10); ctx.lineTo(x + 5, y + 10);
        ctx.closePath(); ctx.fill();
      }

      if (exitIdx != null) {
        const x = pad.l + exitIdx * cw + cw / 2;
        const y = scY(candles[exitIdx].high) - 12;
        ctx.fillStyle = ret > 0 ? '#3fb950' : '#f85149';
        ctx.beginPath();
        ctx.moveTo(x, y); ctx.lineTo(x - 5, y - 10); ctx.lineTo(x + 5, y - 10);
        ctx.closePath(); ctx.fill();

        if (entryIdx != null) {
          const ex = pad.l + entryIdx * cw + cw / 2;
          const ey = scY(candles[entryIdx].close);
          ctx.strokeStyle = ret > 0 ? '#3fb95040' : '#f8514940';
          ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
          ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(x, scY(candles[exitIdx].close)); ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    });

    // Title
    ctx.fillStyle = '#e6edf3'; ctx.font = 'bold 14px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(`${sector} — ${study.name}`, pad.l + 5, 18);
    ctx.fillStyle = '#7d8590'; ctx.font = '12px sans-serif';
    ctx.fillText(`${trades.length} trades`, pad.l + 5 + ctx.measureText(`${sector} — ${study.name}`).width + 15, 18);

    // Draw indicator panels
    drawDualLinePanel(rsiCanvas.current, candles, 'rsi', 'rsi_sma', 'RSI(10)', 'SMA(10)', '#58a6ff', '#d29922', [30, 50, 70], cw, pad);
    drawIndicatorPanel(sortinoCanvas.current, candles, 'sortino', 'Sortino(10)', '#3fb950', [0], cw, pad);
    drawIndicatorPanel(omegaCanvas.current, candles, 'omega', 'Omega(10)', '#d29922', [1], cw, pad);
    drawDualLinePanel(rsiSortCanvas.current, candles, 'rsi_sort', 'rsi_sort_sma', 'RSI of Sortino', 'SMA', '#a371f7', '#7d8590', [30, 50, 70], cw, pad);
    drawIndicatorPanel(cvarCanvas.current, candles, 'cvar', 'CVaR(10)', '#f85149', [], cw, pad);
    drawIndicatorPanel(ulcerCanvas.current, candles, 'ulcer', 'Ulcer Index(10)', '#f0883e', [], cw, pad);
    drawIndicatorPanel(betaCanvas.current, candles, 'beta', 'Beta(10)', '#58a6ff', [1], cw, pad);
    drawDualLinePanel(captureCanvas.current, candles, 'up_capture', 'dn_capture', 'Up%', 'Dn%', '#3fb950', '#f85149', [100], cw, pad);

  }, [chartData, sectorTrades, sector, study]);

  useEffect(() => { drawChart(); }, [drawChart]);

  const trades = (sectorTrades?.trades || []).filter(t => (t.etf || t.sector) === sector);
  const wins = trades.filter(t => (t.return ?? t.return_pct ?? 0) > 0).length;
  const tradesSort = useSortedRows(trades, null, 'desc', {
    ret: t => (t.return ?? t.return_pct ?? 0),
  });

  return (
    <div className="study-chart-page">
      <button className="back-btn" onClick={() => { navigate('/studies'); onBack(); }}>Back to Indicator Studies</button>
      <h2>{sector} — {study.signal_name} <span className="dim" style={{fontSize:12,fontWeight:400}}>(D)</span> &rarr; {study.exit_name}</h2>
      <p className="dim">{trades.length} trades on {sector} | {wins} wins ({trades.length > 0 ? (wins/trades.length*100).toFixed(0) : 0}% WR) | Overall: {study.avg_return > 0 ? '+' : ''}{study.avg_return?.toFixed(3)}% avg</p>

      <div className="chart-container" style={{marginTop: 12}}>
        {chartData ? <canvas ref={priceCanvas} /> : <div className="loading">Loading chart...</div>}
      </div>
      {chartData && <>
        <div className="chart-container"><canvas ref={rsiCanvas} /></div>
        <div className="chart-container"><canvas ref={sortinoCanvas} /></div>
        <div className="chart-container"><canvas ref={omegaCanvas} /></div>
        <div className="chart-container"><canvas ref={rsiSortCanvas} /></div>
        <div className="chart-container"><canvas ref={cvarCanvas} /></div>
        <div className="chart-container"><canvas ref={ulcerCanvas} /></div>
        <div className="chart-container"><canvas ref={betaCanvas} /></div>
        <div className="chart-container"><canvas ref={captureCanvas} /></div>
      </>}

      {trades.length > 0 && (
        <div style={{marginTop: 16}}>
          <h3 style={{fontSize:14,marginBottom:8}}>Trades on {sector}</h3>
          <div className="trades-table-wrap">
            <table className="trades-table">
              <thead><tr><SortTh colKey="entry_date" sort={tradesSort}><Term k="entrydate">Entry</Term></SortTh><SortTh colKey="exit_date" sort={tradesSort}><Term k="exitdate">Exit</Term></SortTh><SortTh colKey="entry_price" sort={tradesSort}><Term k="entry">Entry $</Term></SortTh><SortTh colKey="exit_price" sort={tradesSort}><Term k="exitprice">Exit $</Term></SortTh><SortTh colKey="ret" sort={tradesSort}><Term k="retcol">Return</Term></SortTh><SortTh colKey="spy_ret" sort={tradesSort}><Term k="spy">SPY</Term></SortTh><SortTh colKey="alpha" sort={tradesSort}><Term k="alphacol">Alpha</Term></SortTh><SortTh colKey="hold_days" sort={tradesSort}><Term k="avghold">Hold</Term></SortTh><SortTh colKey="max_drawdown" sort={tradesSort}><Term k="drawdown">DD</Term></SortTh><SortTh colKey="peak_day" sort={tradesSort}><Term k="peak">Peak</Term></SortTh><SortTh colKey="peak_ret" sort={tradesSort}><Term k="peak">Peak Ret</Term></SortTh><SortTh colKey="ret_90d" sort={tradesSort}><Term k="ret90d">90d Ret</Term></SortTh></tr></thead>
              <tbody>
                {tradesSort.rows.map((t, i) => {
                  const ret = t.return ?? t.return_pct ?? 0;
                  return (
                    <tr key={i} className={ret > 0 ? 'row-bullish' : 'row-bearish'}>
                      <td>{t.entry_date}</td>
                      <td>{t.exit_date}</td>
                      <td>${t.entry_price?.toFixed(2)}</td>
                      <td>${t.exit_price?.toFixed(2)}</td>
                      <td className={ret > 0 ? 'good' : 'bad'}>{ret > 0 ? '+' : ''}{ret.toFixed(2)}%</td>
                      <td className={t.spy_ret > 0 ? 'good' : t.spy_ret < 0 ? 'bad' : 'dim'}>{t.spy_ret != null ? `${t.spy_ret > 0 ? '+' : ''}${t.spy_ret.toFixed(1)}%` : '-'}</td>
                      <td className={t.alpha > 0 ? 'good' : t.alpha < 0 ? 'bad' : 'dim'} style={{fontWeight:600}}>{t.alpha != null ? `${t.alpha > 0 ? '+' : ''}${t.alpha.toFixed(1)}%` : '-'}</td>
                      <td className="dim">{t.hold_days}d</td>
                      <td className="bad">{t.max_drawdown != null ? `${t.max_drawdown.toFixed(1)}%` : '-'}</td>
                      <td className="dim">{t.peak_day != null ? `d${t.peak_day}` : '-'}</td>
                      <td className={t.peak_ret > 0 ? 'good' : 'dim'}>{t.peak_ret != null ? `+${t.peak_ret.toFixed(1)}%` : '-'}</td>
                      <td className={t.ret_90d > 0 ? 'good' : t.ret_90d < 0 ? 'bad' : 'dim'}>{t.ret_90d != null ? `${t.ret_90d > 0 ? '+' : ''}${t.ret_90d.toFixed(1)}%` : '-'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// Exit ladder for a single signal group: a sortable sub-table of that signal's exits.
// Own hooks (sort + per-exit trades) so rules-of-hooks stay legal inside the parent's row map.
// `onChart(row)` opens the StudyChartView (parent owns navigation + studyDetail state).
function StudyExitLadder({ rows, onChart }) {
  const sort = useSortedRows(rows, 'avg_return', 'desc');
  const [expandedExit, setExpandedExit] = useState(null);
  const [trades, setTrades] = useState({});
  const [tradesLoading, setTradesLoading] = useState(null);

  const LADDER_COLS = 10;
  const toggleExit = (s) => {
    const next = expandedExit === s.exit ? null : s.exit;
    setExpandedExit(next);
    if (next && s.id != null && !trades[s.id]) {
      setTradesLoading(s.id);
      fetch(`${API}/studies/${s.id}/trades?signal=${encodeURIComponent(s.signal)}&exit=${encodeURIComponent(s.exit)}`)
        .then(r => r.json())
        .then(d => { setTrades(prev => ({ ...prev, [s.id]: d })); setTradesLoading(null); })
        .catch(() => setTradesLoading(null));
    }
  };

  return (
    <table className="exit-ladder">
      <thead>
        <tr>
          <th style={{ width: 18 }}></th>
          <SortTh label="Exit" colKey="exit_name" sort={sort}><Term k="exit">Exit</Term></SortTh>
          <SortTh label="Avg Ret" colKey="avg_return" sort={sort}><Term k="avgreturn">Avg Ret</Term></SortTh>
          <SortTh label="Win%" colKey="win_rate" sort={sort}><Term k="winrate">Win%</Term></SortTh>
          <SortTh label="Trades" colKey="total_trades" sort={sort}><Term k="ntrades">Trades</Term></SortTh>
          <SortTh label="Sig t" colKey="t_stat" sort={sort}><Term k="sig">Sig t</Term></SortTh>
          <SortTh label="Hold" colKey="avg_hold" sort={sort}><Term k="avghold">Hold</Term></SortTh>
          <SortTh label="Avg Dip" colKey="avg_mae" sort={sort}><Term k="avgdip">Avg Dip</Term></SortTh>
          <SortTh label="Clean%" colKey="clean_pct" sort={sort}><Term k="cleanpct">Clean%</Term></SortTh>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {sort.rows.map(s => (
          <React.Fragment key={s.exit}>
            <tr className={s.avg_return > 0 ? 'row-bullish' : 'row-bearish'} style={{ cursor: 'pointer' }} onClick={() => toggleExit(s)}>
              <td className="dim">{expandedExit === s.exit ? '▾' : '▸'}</td>
              <td>{s.exit_name}</td>
              <td className={s.avg_return > 0 ? 'good' : 'bad'}>{s.avg_return > 0 ? '+' : ''}{s.avg_return.toFixed(3)}%</td>
              <td className={s.win_rate > 55 ? 'good' : s.win_rate < 45 ? 'bad' : ''}>{s.win_rate.toFixed(1)}%</td>
              <td>{(s.total_trades ?? 0).toLocaleString()}</td>
              <td className={s.t_stat == null ? 'dim' : Math.abs(s.t_stat) >= 2 ? 'good' : Math.abs(s.t_stat) < 1 ? 'bad' : ''}>{s.t_stat != null ? s.t_stat.toFixed(1) : '-'}</td>
              <td className="dim">{s.avg_hold != null ? `${s.avg_hold.toFixed(0)}d` : '-'}</td>
              <td className={s.avg_mae == null ? 'dim' : s.avg_mae >= -3 ? 'good' : s.avg_mae >= -8 ? '' : 'bad'}>{s.avg_mae != null ? `${s.avg_mae.toFixed(1)}%` : '-'}</td>
              <td className={s.clean_pct == null ? 'dim' : s.clean_pct >= 40 ? 'good' : s.clean_pct < 20 ? 'bad' : ''}>{s.clean_pct != null ? `${s.clean_pct.toFixed(0)}%` : '-'}</td>
              <td><button className="chart-link" onClick={e => { e.stopPropagation(); onChart(s); }}>chart</button></td>
            </tr>
            {expandedExit === s.exit && (
              <tr className="study-detail-row">
                <td colSpan={LADDER_COLS}>
                  <div className="study-detail">
                    <div className="study-detail-section">
                      <h4>Best Sectors</h4>
                      {s.best_sectors?.map(b => (
                        <div key={b.sector} className="study-sector-row good">{b.sector}: {b.avg_return > 0 ? '+' : ''}{b.avg_return.toFixed(3)}% ({b.trades} trades, {b.win_rate.toFixed(0)}% wr)</div>
                      ))}
                    </div>
                    <div className="study-detail-section">
                      <h4>Worst Sectors</h4>
                      {s.worst_sectors?.map(b => (
                        <div key={b.sector} className="study-sector-row bad">{b.sector}: {b.avg_return > 0 ? '+' : ''}{b.avg_return.toFixed(3)}% ({b.trades} trades, {b.win_rate.toFixed(0)}% wr)</div>
                      ))}
                    </div>
                    <div className="study-detail-section" style={{ minWidth: '100%' }}>
                      <h4>Strategy</h4>
                      <div style={{ fontSize: 13, marginBottom: 4 }}>
                        <b>Buy</b> when <span className="good">{s.signal_name}</span> triggers on the daily chart.
                        <b> Sell</b> when <span className="bad">{s.exit_name}</span>.
                        {s.avg_hold ? ` Average hold: ${s.avg_hold.toFixed(0)} days.` : ''}
                        {s.ret_90d != null ? ` If held 90 days instead: ${s.ret_90d > 0 ? '+' : ''}${s.ret_90d.toFixed(2)}%.` : ''}
                        {s.peak_day ? ` Optimal exit: day ${s.peak_day} (${s.peak_avg > 0 ? '+' : ''}${s.peak_avg?.toFixed(2)}%).` : ''}
                      </div>
                      <div className="dim">{s.sector_count || '?'} sectors | {s.total_trades?.toLocaleString()} trades</div>
                    </div>
                    {s.by_regime && Object.keys(s.by_regime).length > 0 && (
                      <div className="study-detail-section">
                        <h4>By Rate Regime</h4>
                        {['LOW','MEDIUM','HIGH'].map(r => s.by_regime[r] && (
                          <div key={r} className="study-sector-row">
                            <span className={r === 'LOW' ? 'good' : r === 'HIGH' ? 'bad' : 'neutral'}>{r}</span>
                            : {s.by_regime[r].avg_return > 0 ? '+' : ''}{s.by_regime[r].avg_return.toFixed(3)}% ({s.by_regime[r].trades} trades, {s.by_regime[r].win_rate}% wr)
                          </div>
                        ))}
                        {(() => {
                          const b = ['LOW','MEDIUM','HIGH'].filter(r => s.by_regime[r]).map(r => ({ label: r, value: s.by_regime[r].avg_return, n: s.by_regime[r].trades }));
                          return b.length >= 2 ? (<div style={{ maxWidth: 260, marginTop: 4 }}>
                            <div className="subtitle darkpool-muted" style={{ margin: '2px 0' }}>By rate regime</div>
                            <TailStrip buckets={b} />
                          </div>) : null;
                        })()}
                      </div>
                    )}
                    {s.by_curve && Object.keys(s.by_curve).length > 0 && (
                      <div className="study-detail-section">
                        <h4>By Yield Curve</h4>
                        {['NORMAL','INVERTED'].map(c => s.by_curve[c] && (
                          <div key={c} className="study-sector-row">
                            <span className={c === 'NORMAL' ? 'good' : 'bad'}>{c}</span>
                            : {s.by_curve[c].avg_return > 0 ? '+' : ''}{s.by_curve[c].avg_return.toFixed(3)}% ({s.by_curve[c].trades} trades, {s.by_curve[c].win_rate}% wr)
                          </div>
                        ))}
                      </div>
                    )}
                    {s.by_vix && Object.keys(s.by_vix).length > 0 && (
                      <div className="study-detail-section">
                        <h4>By VIX Level</h4>
                        {['LOW_VIX','MED_VIX','HIGH_VIX'].map(v => s.by_vix[v] && (
                          <div key={v} className="study-sector-row">
                            <span className={v === 'LOW_VIX' ? 'good' : v === 'HIGH_VIX' ? 'bad' : 'neutral'}>{v.replace('_',' ')}</span>
                            : {s.by_vix[v].avg_return > 0 ? '+' : ''}{s.by_vix[v].avg_return.toFixed(3)}% ({s.by_vix[v].trades} trades, {s.by_vix[v].win_rate}% wr)
                          </div>
                        ))}
                      </div>
                    )}
                    {s.by_spy_trend && Object.keys(s.by_spy_trend).length > 0 && (
                      <div className="study-detail-section">
                        <h4>By SPY Trend</h4>
                        {['BULL','BEAR'].map(t => s.by_spy_trend[t] && (
                          <div key={t} className="study-sector-row">
                            <span className={t === 'BULL' ? 'good' : 'bad'}>{t}</span>
                            : {s.by_spy_trend[t].avg_return > 0 ? '+' : ''}{s.by_spy_trend[t].avg_return.toFixed(3)}% ({s.by_spy_trend[t].trades} trades, {s.by_spy_trend[t].win_rate}% wr)
                          </div>
                        ))}
                        {(() => {
                          const b = ['BULL','BEAR'].filter(t => s.by_spy_trend[t]).map(t => ({ label: t, value: s.by_spy_trend[t].avg_return, n: s.by_spy_trend[t].trades }));
                          return b.length >= 2 ? (<div style={{ maxWidth: 200, marginTop: 4 }}>
                            <div className="subtitle darkpool-muted" style={{ margin: '2px 0' }}>By SPY trend</div>
                            <TailStrip buckets={b} />
                          </div>) : null;
                        })()}
                      </div>
                    )}
                    {s.by_season && Object.keys(s.by_season).length > 0 && (
                      <div className="study-detail-section">
                        <h4>By Season</h4>
                        {['NOV_APR','MAY_OCT'].map(t => s.by_season[t] && (
                          <div key={t} className="study-sector-row">
                            <span className={t === 'NOV_APR' ? 'good' : 'neutral'}>{t.replace('_','-')}</span>
                            : {s.by_season[t].avg_return > 0 ? '+' : ''}{s.by_season[t].avg_return.toFixed(3)}% ({s.by_season[t].trades} trades, {s.by_season[t].win_rate}% wr)
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="study-trades">
                    <h4>All Trades</h4>
                    {tradesLoading === s.id ? <div className="dim">Loading trades...</div> :
                     s.id != null && trades[s.id] ? (() => {
                      const allT = trades[s.id].trades || [];
                      const sectors = [...new Set(allT.map(t => t.sector || t.etf))].sort();
                      const _ts = (k, d) => [trades[s.id]?.[k] ?? d, (v) => setTrades(prev => ({ ...prev, [s.id]: { ...prev[s.id], [k]: v } }))];
                      const [tradeSectorFilter, setTradeSectorFilter] = _ts('_sectorFilter', 'all');
                      const [tradeSort, setTradeSort] = _ts('_sort', 'entry_date');
                      const [tradeSortDir, setTradeSortDir] = _ts('_sortDir', 'desc');
                      const [tradeRegimeFilter, setTradeRegimeFilter] = _ts('_regimeFilter', null);
                      let filtered = tradeSectorFilter === 'all' ? [...allT] : allT.filter(t => (t.sector || t.etf) === tradeSectorFilter);
                      if (tradeRegimeFilter) {
                        const [rk, rv] = tradeRegimeFilter.split(':');
                        filtered = filtered.filter(t => t[rk] === rv);
                      }
                      filtered.sort((a, b) => {
                        let av = a[tradeSort] ?? a['return_pct'] ?? 0, bv = b[tradeSort] ?? b['return_pct'] ?? 0;
                        if (typeof av === 'string') return tradeSortDir === 'desc' ? bv.localeCompare(av) : av.localeCompare(bv);
                        return tradeSortDir === 'desc' ? bv - av : av - bv;
                      });
                      const thClick = (col) => { if (tradeSort === col) setTradeSortDir(tradeSortDir === 'desc' ? 'asc' : 'desc'); else { setTradeSort(col); setTradeSortDir('desc'); } };
                      const thArrow = (col) => tradeSort === col ? (tradeSortDir === 'desc' ? ' ▼' : ' ▲') : '';
                      return (
                      <div>
                        <div className="filters" style={{ marginBottom: 8 }}>
                          <button className={tradeSectorFilter === 'all' ? 'active' : ''} onClick={() => setTradeSectorFilter('all')}>All ({allT.length})</button>
                          {sectors.slice(0, 15).map(sec => {
                            const cnt = allT.filter(t => (t.sector || t.etf) === sec).length;
                            return <button key={sec} className={tradeSectorFilter === sec ? 'active' : ''} onClick={() => setTradeSectorFilter(sec)}>{sec} ({cnt})</button>;
                          })}
                        </div>
                        <div className="filters" style={{ marginBottom: 8, flexWrap: 'wrap' }}>
                          <span className="dim" style={{ fontSize: 11 }}>Regime:</span>
                          {[
                            ['regime:LOW','Rates Low'],['regime:MEDIUM','Rates Med'],['regime:HIGH','Rates High'],
                            null,
                            ['curve:NORMAL','Normal'],['curve:INVERTED','Inverted'],
                            null,
                            ['vix:LOW_VIX','VIX Low'],['vix:MED_VIX','VIX Med'],['vix:HIGH_VIX','VIX High'],
                            null,
                            ['spy:BULL','Bull'],['spy:BEAR','Bear'],
                            null,
                            ['season:NOV_APR','Nov-Apr'],['season:MAY_OCT','May-Oct'],
                          ].map((item, i) => item === null ? <span key={i} className="filter-sep">|</span> :
                            <button key={item[0]} className={tradeRegimeFilter === item[0] ? 'active' : ''} onClick={() => setTradeRegimeFilter(tradeRegimeFilter === item[0] ? null : item[0])} style={{ fontSize: 11, padding: '3px 8px' }}>{item[1]}</button>
                          )}
                          {tradeRegimeFilter && <button onClick={() => setTradeRegimeFilter(null)} style={{ fontSize: 11, padding: '3px 8px', color: 'var(--red)' }}>Clear</button>}
                        </div>
                        <div className="trades-table-wrap">
                        <table className="trades-table">
                          <thead><tr>
                            {[['entry_date','Date'],['sector','Sector'],['etf','ETF'],['entry_price','Entry'],['exit_price','Exit'],['exit_date','Exit Date'],['return_pct','Return'],['spy_ret','SPY'],['alpha','Alpha'],['hold_days','Hold'],['max_drawdown','DD'],['peak_day','Peak'],['peak_ret','Peak Ret'],['ret_90d','90d']].map(([k,l]) => (
                              <th key={k} className="sortable" onClick={() => thClick(k)}><Term k={TH_KEY[k]}>{l}</Term>{thArrow(k)}</th>
                            ))}
                            <th></th>
                          </tr></thead>
                          <tbody>
                            {filtered.map((t, ti) => {
                              const ret = t.return ?? t.return_pct ?? 0;
                              return (
                              <tr key={ti} className={ret > 0 ? 'row-bullish' : 'row-bearish'}>
                                <td>{t.entry_date}</td>
                                <td>{t.sector}</td>
                                <td className="etf">{t.etf}</td>
                                <td>${t.entry_price?.toFixed(2)}</td>
                                <td>${t.exit_price?.toFixed(2)}</td>
                                <td>{t.exit_date}</td>
                                <td className={t.ongoing ? 'neutral' : ret > 0 ? 'good' : 'bad'}>{t.ongoing ? 'OPEN ' : ''}{ret > 0 ? '+' : ''}{ret.toFixed(2)}%</td>
                                <td className={t.spy_ret > 0 ? 'good' : t.spy_ret < 0 ? 'bad' : 'dim'}>{t.spy_ret != null ? `${t.spy_ret > 0 ? '+' : ''}${t.spy_ret.toFixed(1)}%` : '-'}</td>
                                <td className={t.alpha > 0 ? 'good' : t.alpha < 0 ? 'bad' : 'dim'} style={{ fontWeight: 600 }}>{t.alpha != null ? `${t.alpha > 0 ? '+' : ''}${t.alpha.toFixed(1)}%` : '-'}</td>
                                <td className="dim">{t.hold_days}d</td>
                                <td className="bad">{t.max_drawdown != null ? `${t.max_drawdown.toFixed(1)}%` : '-'}</td>
                                <td className="dim">{t.peak_day != null ? `d${t.peak_day}` : '-'}</td>
                                <td className={t.peak_ret > 0 ? 'good' : 'dim'}>{t.peak_ret != null ? `+${t.peak_ret.toFixed(1)}%` : '-'}</td>
                                <td className={t.ret_90d > 0 ? 'good' : t.ret_90d < 0 ? 'bad' : 'dim'}>{t.ret_90d != null ? `${t.ret_90d > 0 ? '+' : ''}${t.ret_90d.toFixed(1)}%` : '-'}</td>
                                <td><button className="chart-link" onClick={() => { const sec = t.etf || t.sector; onChart(s, sec); }}>chart</button></td>
                              </tr>
                              );
                            })}
                          </tbody>
                        </table>
                        {(() => {
                          const closed = filtered.filter(t => !t.ongoing);
                          const open = filtered.length - closed.length;
                          const wins = closed.filter(t => (t.return_pct ?? t.return ?? 0) > 0).length;
                          const avgRet = closed.length > 0 ? closed.reduce((s2, t) => s2 + (t.return_pct ?? t.return ?? 0), 0) / closed.length : 0;
                          return <div className="dim" style={{ padding: 8 }}>
                            {closed.length} closed trades | {wins} wins ({closed.length > 0 ? (wins/closed.length*100).toFixed(1) : 0}% WR) | Avg: {avgRet > 0 ? '+' : ''}{avgRet.toFixed(2)}%
                            {open > 0 && <span className="neutral"> | {open} ongoing</span>}
                          </div>;
                        })()}
                        </div>
                      </div>
                      );
                    })() : <div className="dim">Expand to load trades.</div>}
                  </div>
                </td>
              </tr>
            )}
          </React.Fragment>
        ))}
      </tbody>
    </table>
  );
}

function StudiesPage() {
  const [search, setSearch] = useState('');
  const [dsearch, setDsearch] = useState('');          // debounced -> server query
  const [catFilter, setCatFilter] = useState('all');
  const [regimeFilter, setRegimeFilter] = useState(null);
  const [regime, setRegime] = useState(null);

  const [grouped, setGrouped] = useState([]);
  const [meta, setMeta] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [expandedSignal, setExpandedSignal] = useState(null);
  const [firingSignal, setFiringSignal] = useState(null);   // open FiringPane for this signal key
  const [exitRows, setExitRows] = useState({});        // {signal: [exit rows]}
  const [exitLoading, setExitLoading] = useState(null);
  const [exitErr, setExitErr] = useState({});          // {signal: message}
  const [studyDetail, setStudyDetail] = useState(null);

  // Debounce the search box so a keystroke doesn't re-query the server on every character.
  useEffect(() => { const t = setTimeout(() => setDsearch(search), 300); return () => clearTimeout(t); }, [search]);

  // Grouped fetch: ONE row per signal (~375 rows, no pagination). Refetches when the
  // debounced search / category / regime filter change (keyed off the serialized params).
  const paramsKey = React.useMemo(() => JSON.stringify({
    search: dsearch,
    category: catFilter === 'all' ? '' : catFilter,
    regime: regimeFilter || '',
  }), [dsearch, catFilter, regimeFilter]);

  useEffect(() => {
    let live = true;
    setLoading(true); setError(null);
    const p = JSON.parse(paramsKey);
    const qs = new URLSearchParams({ group: 'signal' });
    Object.entries(p).forEach(([k, v]) => { if (v) qs.set(k, v); });
    apiFetch(`/studies?${qs.toString()}`)
      .then(d => { if (!live) return; setGrouped(d.grouped || []); setMeta(d); setLoading(false); })
      .catch(e => { if (live) { setError(e.message || 'Failed to load.'); setLoading(false); } });
    return () => { live = false; };
  }, [paramsKey]);

  useEffect(() => { fetch(`${API}/regime`).then(r => r.json()).then(setRegime).catch(() => {}); }, []);

  // Deep link to a specific study chart (#/study/<signal>/<exit>/<sector>): fetch just that study.
  useEffect(() => {
    const init = parseHash();
    if (init.view === 'study_chart' && init.signal && init.exit) {
      apiFetch(`/studies?paginate=1&signal=${encodeURIComponent(init.signal)}&exit=${encodeURIComponent(init.exit)}`)
        .then(d => { const m = (d.results || [])[0]; if (m) setStudyDetail({ ...m, chartSector: init.sector }); })
        .catch(() => {});
    }
  }, []);

  // Open the chart for a given exit row (optionally scoped to a sector). Parent owns navigation.
  const openChart = useCallback((row, sector) => {
    const sec = sector || row.best_sectors?.[0]?.sector || '';
    navigate(sec ? `/study/${encodeURIComponent(row.signal)}/${encodeURIComponent(row.exit)}/${encodeURIComponent(sec)}`
                 : `/study/${encodeURIComponent(row.signal)}/${encodeURIComponent(row.exit)}`);
    setStudyDetail({ ...row, chartSector: sec });
  }, []);

  // Lazily load a signal's exit ladder on first expand; cache per signal.
  const toggleSignal = useCallback((g) => {
    setExpandedSignal(prev => {
      const next = prev === g.signal ? null : g.signal;
      if (next && !exitRows[g.signal] && exitLoading !== g.signal) {
        setExitLoading(g.signal);
        setExitErr(prev2 => { const n = { ...prev2 }; delete n[g.signal]; return n; });
        apiFetch(`/studies?paginate=1&page_size=100&signal=${encodeURIComponent(g.signal)}&ordering=avg_return&dir=desc`)
          .then(d => { setExitRows(prev2 => ({ ...prev2, [g.signal]: d.results || [] })); setExitLoading(null); })
          .catch(e => { setExitErr(prev2 => ({ ...prev2, [g.signal]: e.message || 'Failed to load exits.' })); setExitLoading(null); });
      }
      return next;
    });
  }, [exitRows, exitLoading]);

  // Sortable grouped table. Nested best.* columns use accessors.
  const accessors = React.useMemo(() => ({
    best_exit: g => g.best?.exit_name,
    best_avg: g => g.best?.avg_return,
    best_wr: g => g.best?.win_rate,
    best_t: g => g.best?.t_stat,
    range: g => g.max_ret,
  }), []);
  const sort = useSortedRows(grouped, 'best_avg', 'desc', accessors);

  if (studyDetail) {
    return <StudyChartView study={studyDetail} onBack={() => setStudyDetail(null)} />;
  }
  if (error && !grouped.length) return <div className="studies-page"><ErrorBanner message={error} /></div>;
  if (loading && !grouped.length) return <div className="loading">Loading studies...</div>;

  const categories = meta.categories || [];
  const totalSignals = meta.total_signals ?? grouped.length;
  const GROUP_COLS = 11;

  return (
    <div className="studies-page">
      <h1>Indicator Studies <span className="dim">({totalSignals.toLocaleString()} signals)</span>
        <LastUpdatedChip value={meta.last_updated} />
      </h1>
      <p className="subtitle">One row per signal across all sector ETFs (5y daily backtest). Expand a signal to see its exit ladder; sort either level by any column.</p>

      {regime && (
        <div className="regime-bar">
          <span>Rates: <b className={regime.regime === 'LOW' ? 'good' : regime.regime === 'HIGH' ? 'bad' : 'neutral'}>{regime.regime}</b> ({regime.rate_3m}%)</span>
          <span>Curve: <b className={regime.curve === 'NORMAL' ? 'good' : 'bad'}>{regime.curve}</b></span>
          <span>VIX: <b className={regime.vix_regime === 'LOW_VIX' ? 'good' : regime.vix_regime === 'HIGH_VIX' ? 'bad' : 'neutral'}>{regime.vix}</b></span>
          <span>SPY: <b className={regime.spy_trend === 'BULL' ? 'good' : 'bad'}>{regime.spy_trend}</b></span>
          <span>Season: <b>{regime.season?.replace('_','-')}</b></span>
        </div>
      )}

      <div className="studies-controls">
        <input className="studies-search" type="text" placeholder="Search studies..." value={search} onChange={e => setSearch(e.target.value)} />
        <div className="filters">
          <button className={catFilter === 'all' ? 'active' : ''} onClick={() => setCatFilter('all')}>All</button>
          {categories.map(c => (
            <button key={c} className={catFilter === c ? 'active' : ''} onClick={() => setCatFilter(c)}>{c}</button>
          ))}
        </div>
        <div className="filters" style={{ flexWrap: 'wrap' }}>
          <span className="dim" style={{ fontSize: 11 }}>Regime:</span>
          {[
            ['by_regime:LOW','Rates: Low'],['by_regime:MEDIUM','Rates: Med'],['by_regime:HIGH','Rates: High'],
            null,
            ['by_curve:NORMAL','Curve: Normal'],['by_curve:INVERTED','Curve: Inverted'],
            null,
            ['by_vix:LOW_VIX','VIX: Low'],['by_vix:MED_VIX','VIX: Med'],['by_vix:HIGH_VIX','VIX: High'],
            null,
            ['by_spy_trend:BULL','SPY: Bull'],['by_spy_trend:BEAR','SPY: Bear'],
            null,
            ['by_season:NOV_APR','Nov-Apr'],['by_season:MAY_OCT','May-Oct'],
          ].map((item, i) => item === null ? <span key={i} className="filter-sep">|</span> :
            <button key={item[0]} className={regimeFilter === item[0] ? 'active' : ''} onClick={() => setRegimeFilter(regimeFilter === item[0] ? null : item[0])} style={{ fontSize: 11, padding: '3px 8px' }}>{item[1]}</button>
          )}
          {regimeFilter && <button onClick={() => setRegimeFilter(null)} style={{ fontSize: 11, padding: '3px 8px', color: 'var(--red)' }}>Clear</button>}
        </div>
      </div>

      <table>
        <thead>
          <tr>
            <th style={{ width: 18 }}></th>
            <SortTh label="Signal" colKey="signal_name" sort={sort}><Term k="signal">Signal</Term></SortTh>
            <SortTh label="Cat" colKey="category" sort={sort}><Term k="category">Cat</Term></SortTh>
            <SortTh label="# Exits" colKey="n_exits" sort={sort}><Term k="exit"># Exits</Term></SortTh>
            <SortTh label="Profitable" colKey="profitable" sort={sort} title="Exits with a positive average return">Profitable</SortTh>
            <SortTh label="Best exit" colKey="best_exit" sort={sort}><Term k="exit">Best exit</Term></SortTh>
            <SortTh label="Best Avg Ret" colKey="best_avg" sort={sort}><Term k="avgreturn">Best Avg Ret</Term></SortTh>
            <SortTh label="Win%" colKey="best_wr" sort={sort}><Term k="winrate">Win%</Term></SortTh>
            <SortTh label="Sig t" colKey="best_t" sort={sort}><Term k="sig">Sig t</Term></SortTh>
            <SortTh label="Range" colKey="range" sort={sort} title="Min..max average return across this signal's exits">Range</SortTh>
            <SortTh label="Firing 3d" colKey="n_firing" sort={sort} title="Names firing this signal in the last 3 bars">Firing 3d</SortTh>
          </tr>
        </thead>
        <tbody>
          {sort.rows.map(g => {
            const best = g.best || {};
            const isOpen = expandedSignal === g.signal;
            return (
            <React.Fragment key={g.signal}>
              <tr className={best.avg_return > 0 ? 'row-bullish' : 'row-bearish'} style={{ cursor: 'pointer' }} onClick={() => toggleSignal(g)}>
                <td className="dim">{isOpen ? '▾' : '▸'}</td>
                <td>{g.signal_name} <span className="dim" style={{ fontSize: 10 }}>(D)</span></td>
                <td><span className="study-cat">{g.category}</span></td>
                <td>{g.n_exits}</td>
                <td className={g.profitable > 0 ? 'good' : 'dim'}>{g.profitable}/{g.n_exits}</td>
                <td className="dim">{best.exit_name || '-'}</td>
                <td className={best.avg_return > 0 ? 'good' : 'bad'}>{best.avg_return != null ? `${best.avg_return > 0 ? '+' : ''}${best.avg_return.toFixed(3)}%` : '-'}</td>
                <td className={best.win_rate > 55 ? 'good' : best.win_rate < 45 ? 'bad' : ''}>{best.win_rate != null ? `${best.win_rate.toFixed(1)}%` : '-'}</td>
                <td className={best.t_stat == null ? 'dim' : Math.abs(best.t_stat) >= 2 ? 'good' : Math.abs(best.t_stat) < 1 ? 'bad' : ''}>{best.t_stat != null ? best.t_stat.toFixed(1) : '-'}</td>
                <td className="dim">{g.min_ret != null && g.max_ret != null ? `${g.min_ret > 0 ? '+' : ''}${g.min_ret.toFixed(1)}..${g.max_ret > 0 ? '+' : ''}${g.max_ret.toFixed(1)}%` : '-'}</td>
                <td><FiringCell count={g.n_firing} signalKey={g.signal} onOpen={setFiringSignal} /></td>
              </tr>
              {isOpen && (
                <tr className="study-detail-row">
                  <td colSpan={GROUP_COLS}>
                    {exitLoading === g.signal ? <div className="dim" style={{ padding: 8 }}>Loading exits...</div> :
                     exitErr[g.signal] ? <ErrorBanner message={exitErr[g.signal]} /> :
                     exitRows[g.signal] ? (
                       exitRows[g.signal].length ?
                         <StudyExitLadder rows={exitRows[g.signal]} onChart={openChart} /> :
                         <div className="dim" style={{ padding: 8 }}>No exits for this signal.</div>
                     ) : <div className="dim" style={{ padding: 8 }}>Expand to load exits.</div>}
                  </td>
                </tr>
              )}
            </React.Fragment>
            );
          })}
        </tbody>
      </table>
      {!grouped.length && <div className="dim" style={{ padding: 12, textAlign: 'center' }}>No studies match.</div>}
      <FiringPane signalKey={firingSignal} onClose={() => setFiringSignal(null)} />
    </div>
  );
}

function StockDrilldownPage() {
  const [data, setData] = useState(null);
  const [sortBy, setSortBy] = useState('stock_avg_return');
  const [sortDir, setSortDir] = useState('desc');
  const [expanded, setExpanded] = useState(null);
  const { decomp } = useSectionBacktest();
  const pickOnly = decomp?.decomposition?.lowpb?.arm1_pick_only;

  useEffect(() => {
    fetch(`${API}/stock-drilldown`).then(r => r.json()).then(d => setData(d)).catch(() => {});
  }, []);

  if (!data) return <div className="loading">Loading stock drilldowns...</div>;

  let sorted = [...data.drilldowns];
  sorted.sort((a, b) => {
    const av = a[sortBy] ?? 0, bv = b[sortBy] ?? 0;
    if (typeof av === 'string') return sortDir === 'desc' ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
    return sortDir === 'desc' ? bv - av : av - bv;
  });

  const thClick = (col) => { if (sortBy === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc'); else { setSortBy(col); setSortDir('desc'); } };
  const thArrow = (col) => sortBy === col ? (sortDir === 'desc' ? ' \u25BC' : ' \u25B2') : '';
  const beating = sorted.filter(s => s.alpha_vs_etf > 0).length;

  return (
    <div className="studies-page">
      <h1>Stock Drilldown <span className="dim">({data.total})</span>
        <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle">Top 10% indicator studies: buy highest-beta stock in sector instead of ETF. {beating}/{data.total} beat the ETF.</p>

      <BacktestPanel
        title="Cheapest-P/B stock pick vs SPY"
        subtitle="Whole-universe backtest: each month buy the single cheapest price-to-book stock (the value pick), vs SPY buy-hold."
        curve={pickOnly?.curve} summary={pickOnly?.summary} />

      <table>
        <thead>
          <tr>
            {[['study_name','Strategy'],['category','Cat'],['stock_trades','Trades'],['stock_avg_return','Stock Avg'],['etf_avg_return','ETF Avg'],['alpha_vs_etf','Alpha vs ETF'],['stock_win_rate','Win%'],['stock_max_drawdown','Max Loss']].map(([k,l]) => (
              <th key={k} className="sortable" onClick={() => thClick(k)}><Term k={TH_KEY[k]}>{l}</Term>{thArrow(k)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map(s => (
            <React.Fragment key={s.id}>
              <tr className={s.alpha_vs_etf > 0 ? 'row-bullish' : 'row-bearish'} style={{cursor:'pointer'}} onClick={() => setExpanded(expanded === s.id ? null : s.id)}>
                <td style={{maxWidth:300,overflow:'hidden',textOverflow:'ellipsis'}}>{s.study_name} <span className="dim" style={{fontSize:10}}>(D)</span></td>
                <td><span className="study-cat">{s.category}</span></td>
                <td>{s.stock_trades}</td>
                <td className={s.stock_avg_return > 0 ? 'good' : 'bad'} style={{fontWeight:600}}>{s.stock_avg_return > 0 ? '+' : ''}{s.stock_avg_return.toFixed(2)}%</td>
                <td className={s.etf_avg_return > 0 ? 'good' : 'bad'}>{s.etf_avg_return > 0 ? '+' : ''}{s.etf_avg_return.toFixed(2)}%</td>
                <td className={s.alpha_vs_etf > 0 ? 'good' : 'bad'} style={{fontWeight:700}}>{s.alpha_vs_etf > 0 ? '+' : ''}{s.alpha_vs_etf.toFixed(2)}%</td>
                <td>{s.stock_win_rate.toFixed(0)}%</td>
                <td className="bad">{s.stock_max_drawdown.toFixed(1)}%</td>
              </tr>
              {expanded === s.id && (
                <tr className="study-detail-row">
                  <td colSpan={8}>
                    <div className="study-detail" style={{display:'flex',gap:24,flexWrap:'wrap'}}>
                      {s.best_stocks && s.best_stocks.length > 0 && (
                        <div className="study-detail-section">
                          <h4>Best Stocks</h4>
                          {s.best_stocks.map(st => (
                            <div key={st.ticker} className="study-sector-row good">{st.ticker}: {st.avg_return > 0 ? '+' : ''}{st.avg_return}% ({st.trades} trades, {st.win_rate}% WR)</div>
                          ))}
                        </div>
                      )}
                      {s.worst_stocks && s.worst_stocks.length > 0 && (
                        <div className="study-detail-section">
                          <h4>Worst Stocks</h4>
                          {s.worst_stocks.map(st => (
                            <div key={st.ticker} className="study-sector-row bad">{st.ticker}: {st.avg_return > 0 ? '+' : ''}{st.avg_return}% ({st.trades} trades, {st.win_rate}% WR)</div>
                          ))}
                        </div>
                      )}
                      <div className="study-detail-section">
                        <h4>Strategy</h4>
                        <div style={{fontSize:13}}>
                          <b>Signal:</b> {s.signal_name}<br/>
                          <b>Exit:</b> {s.exit_name}<br/>
                          <b>Method:</b> When signal fires on sector ETF, buy the highest rolling beta stock in that sector's top 20 holdings.
                        </div>
                      </div>
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TrendStudiesPage() {
  const [detail, setDetail] = useState(null);
  const [sortBy, setSortBy] = useState('total_return');
  const [sortDir, setSortDir] = useState('desc');
  const [modeFilter, setModeFilter] = useState('all');
  const equityCanvas = useRef(null);
  const MODE_LABELS = { etf: 'ETF', momentum: 'Momentum stock', hibeta: 'Hi-beta stock' };
  const logSort = useSortedRows(detail && detail.trade_log, null, 'desc');

  // Server-side sort + mode filter + infinite scroll (mirrors StudiesPage). Changing any param
  // refetches from offset 0. The detail view fetches its own heavy row (equity/trade_log) separately.
  const params = React.useMemo(() => ({
    ordering: sortBy, dir: sortDir, mode: modeFilter === 'all' ? '' : modeFilter,
  }), [sortBy, sortDir, modeFilter]);
  const { rows, meta, loading, error, hasMore, loadMore } = usePagedList('/trend-studies', params);

  // Draw equity curve for detail view
  useEffect(() => {
    if (!detail || !equityCanvas.current || !detail.equity_curve) return;
    const ec = detail.equity_curve;
    const sc = detail.spy_curve || [];
    const W = equityCanvas.current.parentElement.clientWidth;
    const H = 300;
    equityCanvas.current.width = W; equityCanvas.current.height = H;
    const ctx = equityCanvas.current.getContext('2d');
    const pad = { l: 60, r: 20, t: 25, b: 30 };

    ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);

    const allVals = [...ec.map(p => p.equity), ...sc.map(p => p.spy_equity)];
    const vMin = Math.min(...allVals) * 0.95;
    const vMax = Math.max(...allVals) * 1.05;
    const range = vMax - vMin || 1;
    const cw = (W - pad.l - pad.r) / Math.max(ec.length - 1, 1);
    const scY = v => pad.t + (1 - (v - vMin) / range) * (H - pad.t - pad.b);

    // Grid
    ctx.strokeStyle = '#1e2430'; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = pad.t + i * (H - pad.t - pad.b) / 4;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      const v = vMax - i * range / 4;
      ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
      ctx.fillText((v * 100 - 100).toFixed(0) + '%', pad.l - 5, y + 4);
    }

    // SPY line
    if (sc.length > 0) {
      ctx.strokeStyle = '#7d859080'; ctx.lineWidth = 2; ctx.beginPath();
      sc.forEach((p, i) => { const x = pad.l + i * cw; i === 0 ? ctx.moveTo(x, scY(p.spy_equity)) : ctx.lineTo(x, scY(p.spy_equity)); });
      ctx.stroke();
    }

    // Strategy line
    ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 2.5; ctx.beginPath();
    ec.forEach((p, i) => { const x = pad.l + i * cw; i === 0 ? ctx.moveTo(x, scY(p.equity)) : ctx.lineTo(x, scY(p.equity)); });
    ctx.stroke();

    // Date labels
    ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(ec.length / 8));
    ec.forEach((p, i) => { if (i % step === 0) ctx.fillText(p.date.slice(2, 7), pad.l + i * cw, H - 5); });

    // Legend
    ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'left';
    ctx.fillStyle = '#3fb950'; ctx.fillText(`Strategy: +${detail.total_return}%`, pad.l + 5, 16);
    ctx.fillStyle = '#7d8590'; ctx.fillText(`SPY: +${detail.spy_total}%`, pad.l + 200, 16);
  }, [detail]);

  if (detail) {
    return (
      <div className="studies-page">
        <button className="back-btn" onClick={() => setDetail(null)}>Back to Trend Studies</button>
        <h2>{MODE_LABELS[detail.hold_mode] || 'ETF'} rotation: Look={detail.lookback_months}m Hold={detail.hold_months}m Top {detail.top_n}</h2>
        <div className="regime-bar">
          <span>Total: <b className="good">+{detail.total_return}%</b></span>
          <span>Annual: <b className="good">+{detail.annual_return}%</b></span>
          <span>SPY: <b>+{detail.spy_total}%</b></span>
          <span>Alpha: <b className={detail.alpha > 0 ? 'good' : 'bad'}>{detail.alpha > 0 ? '+' : ''}{detail.alpha}%</b></span>
          <span>Max DD: <b className="bad">{detail.max_drawdown}%</b></span>
          <span>Win Rate: <b>{detail.win_rate}%</b></span>
          <span>Trades: <b>{detail.num_trades}</b></span>
        </div>
        <div className="chart-container" style={{marginTop: 12}}>
          <canvas ref={equityCanvas} />
        </div>
        <p className="dim" style={{fontSize: 11, marginTop: 4}}>Green = strategy | Gray = SPY (buy & hold)</p>

        {detail.trade_log && (
          <div style={{marginTop: 16}}>
            <h3 style={{fontSize:14,marginBottom:8}}>Rotation Log</h3>
            <div className="trades-table-wrap">
              <table className="trades-table">
                <thead><tr><SortTh colKey="date" sort={logSort}><Term k="datecol">Date</Term></SortTh><SortTh colKey="end_date" sort={logSort}><Term k="endcol">End</Term></SortTh><th><Term k="sector">Sectors</Term></th>{detail.hold_mode && detail.hold_mode !== 'etf' && <th><Term k="held">Held ({MODE_LABELS[detail.hold_mode] || detail.hold_mode})</Term></th>}<SortTh colKey="return_pct" sort={logSort}><Term k="retcol">Return</Term></SortTh></tr></thead>
                <tbody>
                  {logSort.rows.map((t, i) => (
                    <tr key={i} className={t.return_pct > 0 ? 'row-bullish' : 'row-bearish'}>
                      <td>{t.date}</td>
                      <td>{t.end_date}</td>
                      <td style={{fontSize:11}}>{(t.sector_names || t.sectors || []).join(', ')}</td>
                      {detail.hold_mode && detail.hold_mode !== 'etf' && <td style={{fontSize:11,fontWeight:600}}>{(t.picks || []).join(', ')}</td>}
                      <td className={t.return_pct > 0 ? 'good' : 'bad'}>{t.return_pct > 0 ? '+' : ''}{t.return_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  }

  if (error && !rows.length) return <div className="studies-page"><ErrorBanner message={error} /></div>;
  if (loading && !rows.length) return <div className="loading">Loading trend studies...</div>;

  // Server already applied the mode filter + sort; render the loaded rows as-is.
  const modes = meta.modes && meta.modes.length ? meta.modes : ['etf'];
  // Only these header keys are in the server's TREND_ORDER whitelist; the rest stay non-sortable.
  const TREND_SORTABLE = new Set(['lookback_months','hold_months','total_return','annual_return','alpha','max_drawdown','win_rate','num_trades','t_stat']);

  const thClick = (col) => { if (sortBy === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc'); else { setSortBy(col); setSortDir('desc'); } };
  const thArrow = (col) => sortBy === col ? (sortDir === 'desc' ? ' \u25BC' : ' \u25B2') : '';

  return (
    <div className="studies-page">
      <h1>Trend Studies <span className="dim">({rows.length.toLocaleString()} of {(meta.total ?? 0).toLocaleString()})</span>
        <LastUpdatedChip value={meta.last_updated} />
      </h1>
      <p className="subtitle">Sector momentum rotation backtests. Buy top N sectors by trailing return, hold, rebalance. 5Y backtest excluding crypto. <b>Hold mode</b> = what you buy in each winning sector: the ETF, its top-momentum stock, or its highest-beta stock (both stock picks point-in-time).</p>

      <div className="filters" style={{marginBottom:10}}>
        <span className="dim" style={{fontSize:11}}>Hold:</span>
        <button className={modeFilter === 'all' ? 'active' : ''} onClick={() => setModeFilter('all')}>All</button>
        {modes.map(m => (
          <button key={m} className={modeFilter === m ? 'active' : ''} onClick={() => setModeFilter(m)}>{MODE_LABELS[m] || m}</button>
        ))}
      </div>

      <table>
        <thead>
          <tr>
            {[['hold_mode','Hold'],['lookback_months','Lookback'],['hold_months','Hold Mo'],['top_n','Top N'],['total_return','Total Ret'],['annual_return','Annual'],['spy_total','SPY'],['alpha','Alpha'],['max_drawdown','Max DD'],['win_rate','Win%'],['num_trades','Trades'],['t_stat','Sig (t)']].map(([k,l]) => (
              TREND_SORTABLE.has(k)
                ? <th key={k} className="sortable" onClick={() => thClick(k)}><Term k={TH_KEY[k]}>{l}</Term>{thArrow(k)}</th>
                : <th key={k}><Term k={TH_KEY[k]}>{l}</Term></th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(s => (
            <tr key={s.id} className={s.alpha > 0 ? 'row-bullish' : 'row-bearish'} style={{cursor:'pointer'}} onClick={() => {
              fetch(`${API}/trend-studies/${s.id}`).then(r => r.json()).then(d => setDetail(d));
            }}>
              <td><span className={`mode-tag mode-${s.hold_mode || 'etf'}`}>{MODE_LABELS[s.hold_mode || 'etf'] || s.hold_mode}</span></td>
              <td>{s.lookback_months}m</td>
              <td>{s.hold_months}m</td>
              <td>{s.top_n}</td>
              <td className={s.total_return > 0 ? 'good' : 'bad'}>{s.total_return > 0 ? '+' : ''}{s.total_return.toFixed(1)}%</td>
              <td className="good">{s.annual_return > 0 ? '+' : ''}{s.annual_return.toFixed(1)}%</td>
              <td className="dim">+{s.spy_total.toFixed(1)}%</td>
              <td className={s.alpha > 0 ? 'good' : 'bad'} style={{fontWeight:600}}>{s.alpha > 0 ? '+' : ''}{s.alpha.toFixed(1)}%</td>
              <td className="bad">{s.max_drawdown.toFixed(1)}%</td>
              <td>{s.win_rate.toFixed(0)}%</td>
              <td className="dim">{s.num_trades}</td>
              <td className={s.robust ? 'good' : (s.t_stat == null ? 'dim' : '')}
                  title={s.robust ? 'robust: >=12 non-overlapping periods AND |t|>=2' : `${s.num_trades} periods (not robust)`}>
                {s.t_stat != null ? s.t_stat.toFixed(1) : '–'}{s.robust ? ' ✓' : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {hasMore && <ScrollSentinel onVisible={loadMore} disabled={loading} />}
      {loading && rows.length > 0 && <div className="loading dim" style={{padding:12}}>Loading more…</div>}
      {!hasMore && rows.length > 0 && <div className="dim" style={{padding:12,textAlign:'center'}}>All {(meta.total ?? rows.length).toLocaleString()} loaded.</div>}
    </div>
  );
}

function TradeJournalPage() {
  const [entries, setEntries] = useState(() => {
    try { return JSON.parse(localStorage.getItem('trade_journal') || '[]'); } catch { return []; }
  });
  const emptyForm = { date: new Date().toISOString().slice(0,10), ticker: '', side: 'BUY', price: '', quantity: '', exit_price: '', exit_date: '', notes: '', strategy: '', sector: '' };
  const [form, setForm] = useState(emptyForm);
  const [filter, setFilter] = useState('all');
  const [editing, setEditing] = useState(null);

  const saveEntries = (updated) => { setEntries(updated); localStorage.setItem('trade_journal', JSON.stringify(updated)); };

  const save = (e) => {
    e.preventDefault();
    if (editing) {
      const updated = entries.map(en => en.id === editing ? { ...en, ...form, price: parseFloat(form.price), exit_price: form.exit_price ? parseFloat(form.exit_price) : null, quantity: parseInt(form.quantity) || 1 } : en);
      saveEntries(updated);
      setEditing(null);
    } else {
      const entry = { ...form, id: Date.now(), price: parseFloat(form.price), exit_price: form.exit_price ? parseFloat(form.exit_price) : null, quantity: parseInt(form.quantity) || 1 };
      saveEntries([entry, ...entries]);
    }
    setForm(emptyForm);
  };

  const remove = (id) => saveEntries(entries.filter(e => e.id !== id));
  const edit = (e) => { setEditing(e.id); setForm({ ...e, price: e.price?.toString() || '', exit_price: e.exit_price?.toString() || '', quantity: e.quantity?.toString() || '1' }); };

  const { lab } = useSectionBacktest();
  const p3 = lab?.phase3;

  // Compute P&L for closed trades
  const closed = entries.filter(e => e.exit_price);
  const totalPnl = closed.reduce((s, e) => {
    const mult = e.side === 'BUY' ? 1 : -1;
    return s + mult * (e.exit_price - e.price) * (e.quantity || 1);
  }, 0);
  const wins = closed.filter(e => (e.side === 'BUY' ? e.exit_price > e.price : e.exit_price < e.price)).length;
  const tickers = [...new Set(entries.map(e => e.ticker))].sort();
  const filtered = filter === 'all' ? entries : entries.filter(e => e.ticker === filter);
  const jSort = useSortedRows(filtered, null, 'desc', {
    pnl: e => (e.exit_price ? (e.side === 'BUY' ? e.exit_price - e.price : e.price - e.exit_price) * (e.quantity || 1) : null),
  });

  return (
    <div className="studies-page">
      <h1>Trade Journal <LastUpdatedChip value={entries[0]?.date} /></h1>
      <p className="subtitle">Log your trades and track performance.</p>

      <BacktestPanel
        title="The strategy being paper-traded vs SPY"
        subtitle="The top-signals portfolio these picks are tracked forward against — backtested vs SPY buy-hold."
        curve={p3?.curve} summary={p3?.summary} />

      {entries.length > 0 && (
        <div className="regime-bar" style={{marginBottom: 16}}>
          <span>Total: <b>{entries.length}</b> trades</span>
          <span>Open: <b>{entries.length - closed.length}</b></span>
          <span>Closed: <b>{closed.length}</b></span>
          {closed.length > 0 && <>
            <span>Win Rate: <b className={wins/closed.length > 0.5 ? 'good' : 'bad'}>{(wins/closed.length*100).toFixed(0)}%</b></span>
            <span>P&L: <b className={totalPnl > 0 ? 'good' : 'bad'}>${totalPnl > 0 ? '+' : ''}{totalPnl.toFixed(2)}</b></span>
          </>}
        </div>
      )}

      <div style={{background:'var(--card-bg, #161b22)',borderRadius:8,padding:16,marginBottom:16,border:'1px solid var(--border, #30363d)'}}>
        <h3 style={{margin:'0 0 12px 0',fontSize:14}}>{editing ? 'Edit Trade' : 'New Trade'}</h3>
        <form onSubmit={save} style={{display:'grid',gridTemplateColumns:'repeat(auto-fill, minmax(140px, 1fr))',gap:8}}>
          <div>
            <label className="dim" style={{fontSize:10,display:'block',marginBottom:2}}>Date</label>
            <input type="date" value={form.date} onChange={e => setForm({...form, date: e.target.value})} required style={{width:'100%'}} />
          </div>
          <div>
            <label className="dim" style={{fontSize:10,display:'block',marginBottom:2}}>Ticker</label>
            <input placeholder="XLK" value={form.ticker} onChange={e => setForm({...form, ticker: e.target.value.toUpperCase()})} required style={{width:'100%'}} />
          </div>
          <div>
            <label className="dim" style={{fontSize:10,display:'block',marginBottom:2}}>Side</label>
            <select value={form.side} onChange={e => setForm({...form, side: e.target.value})} style={{width:'100%'}}>
              <option value="BUY">BUY</option><option value="SELL">SELL</option>
            </select>
          </div>
          <div>
            <label className="dim" style={{fontSize:10,display:'block',marginBottom:2}}>Entry Price</label>
            <input type="number" step="0.01" placeholder="0.00" value={form.price} onChange={e => setForm({...form, price: e.target.value})} required style={{width:'100%'}} />
          </div>
          <div>
            <label className="dim" style={{fontSize:10,display:'block',marginBottom:2}}>Quantity</label>
            <input type="number" placeholder="1" value={form.quantity} onChange={e => setForm({...form, quantity: e.target.value})} style={{width:'100%'}} />
          </div>
          <div>
            <label className="dim" style={{fontSize:10,display:'block',marginBottom:2}}>Exit Price</label>
            <input type="number" step="0.01" placeholder="(open)" value={form.exit_price} onChange={e => setForm({...form, exit_price: e.target.value})} style={{width:'100%'}} />
          </div>
          <div>
            <label className="dim" style={{fontSize:10,display:'block',marginBottom:2}}>Exit Date</label>
            <input type="date" value={form.exit_date} onChange={e => setForm({...form, exit_date: e.target.value})} style={{width:'100%'}} />
          </div>
          <div>
            <label className="dim" style={{fontSize:10,display:'block',marginBottom:2}}>Strategy</label>
            <input placeholder="RSI Cross" value={form.strategy} onChange={e => setForm({...form, strategy: e.target.value})} style={{width:'100%'}} />
          </div>
          <div>
            <label className="dim" style={{fontSize:10,display:'block',marginBottom:2}}>Sector</label>
            <input placeholder="Technology" value={form.sector} onChange={e => setForm({...form, sector: e.target.value})} style={{width:'100%'}} />
          </div>
          <div style={{gridColumn:'1 / -1'}}>
            <label className="dim" style={{fontSize:10,display:'block',marginBottom:2}}>Notes</label>
            <input placeholder="Trade rationale..." value={form.notes} onChange={e => setForm({...form, notes: e.target.value})} style={{width:'100%'}} />
          </div>
          <div style={{gridColumn:'1 / -1',display:'flex',gap:8}}>
            <button type="submit" className="refresh-btn">{editing ? 'Update' : 'Add Trade'}</button>
            {editing && <button type="button" className="refresh-btn" onClick={() => { setEditing(null); setForm(emptyForm); }} style={{background:'transparent',border:'1px solid var(--border)'}}>Cancel</button>}
          </div>
        </form>
      </div>

      {entries.length > 0 && (
        <>
          {tickers.length > 1 && (
            <div className="filters" style={{marginBottom:8}}>
              <button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>All ({entries.length})</button>
              {tickers.map(t => (
                <button key={t} className={filter === t ? 'active' : ''} onClick={() => setFilter(t)}>{t} ({entries.filter(e => e.ticker === t).length})</button>
              ))}
            </div>
          )}
          <div className="trades-table-wrap">
            <table className="trades-table">
              <thead>
                <tr><SortTh colKey="date" sort={jSort}><Term k="datecol">Date</Term></SortTh><SortTh colKey="ticker" sort={jSort}><Term k="ticker">Ticker</Term></SortTh><SortTh colKey="side" sort={jSort}><Term k="side">Side</Term></SortTh><SortTh colKey="price" sort={jSort}><Term k="entry">Entry</Term></SortTh><SortTh colKey="exit_price" sort={jSort}><Term k="exitprice">Exit</Term></SortTh><SortTh colKey="exit_date" sort={jSort}><Term k="exitdate">Exit Date</Term></SortTh><SortTh colKey="quantity" sort={jSort}><Term k="qty">Qty</Term></SortTh><SortTh colKey="pnl" sort={jSort}><Term k="pnl">P&L</Term></SortTh><SortTh colKey="strategy" sort={jSort}><Term k="strategy">Strategy</Term></SortTh><SortTh colKey="notes" sort={jSort}><Term k="notes">Notes</Term></SortTh><th></th></tr>
              </thead>
              <tbody>
                {jSort.rows.map(e => {
                  const pnl = e.exit_price ? (e.side === 'BUY' ? e.exit_price - e.price : e.price - e.exit_price) * (e.quantity || 1) : null;
                  const pnlPct = e.exit_price && e.price ? ((e.side === 'BUY' ? e.exit_price - e.price : e.price - e.exit_price) / e.price * 100) : null;
                  return (
                    <tr key={e.id} className={e.exit_price ? (pnl > 0 ? 'row-bullish' : 'row-bearish') : ''} style={{cursor:'pointer'}} onClick={() => edit(e)}>
                      <td>{e.date}</td>
                      <td style={{fontWeight:600}}>{e.ticker}</td>
                      <td className={e.side === 'BUY' ? 'good' : 'bad'}>{e.side}</td>
                      <td>${e.price?.toFixed(2)}</td>
                      <td>{e.exit_price ? `$${e.exit_price.toFixed(2)}` : <span className="neutral">OPEN</span>}</td>
                      <td className="dim">{e.exit_date || '-'}</td>
                      <td className="dim">{e.quantity || 1}</td>
                      <td className={pnl > 0 ? 'good' : pnl < 0 ? 'bad' : 'dim'}>
                        {pnl != null ? `${pnl > 0 ? '+' : ''}$${pnl.toFixed(2)} (${pnlPct > 0 ? '+' : ''}${pnlPct.toFixed(1)}%)` : '-'}
                      </td>
                      <td className="dim">{e.strategy}</td>
                      <td className="dim" style={{maxWidth:200,overflow:'hidden',textOverflow:'ellipsis'}}>{e.notes}</td>
                      <td><button className="close-btn" onClick={(ev) => { ev.stopPropagation(); remove(e.id); }} style={{padding:'2px 6px',fontSize:10}}>x</button></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      {entries.length === 0 && (
        <div style={{textAlign:'center',padding:40,color:'var(--dim)'}}>
          <p style={{fontSize:16}}>No trades yet</p>
          <p style={{fontSize:13}}>Use the form above to log your first trade.</p>
        </div>
      )}
    </div>
  );
}

function DocsPage() {
  const [histData, setHistData] = useState(null);
  const rateCanvas = useRef(null);
  const vixCanvas = useRef(null);
  const spyCanvas = useRef(null);

  useEffect(() => {
    fetch(`${API}/regime/history`).then(r => r.json()).then(d => setHistData(d.data)).catch(() => {});
  }, []);

  const drawLine = useCallback((canvas, data, key, color, label, zones) => {
    if (!canvas || !data || data.length === 0) return;
    const vals = data.map(d => d[key]).filter(v => v != null);
    if (vals.length === 0) return;
    const W = canvas.parentElement.clientWidth;
    const H = 200;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    const pad = { l: 60, r: 20, t: 25, b: 25 };
    ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);

    const vMin = Math.min(...vals) * 0.95;
    const vMax = Math.max(...vals) * 1.05;
    const range = vMax - vMin || 1;
    const cw = (W - pad.l - pad.r) / data.length;
    const scY = v => pad.t + (1 - (v - vMin) / range) * (H - pad.t - pad.b);

    // Zone backgrounds
    if (zones) {
      data.forEach((d, i) => {
        const z = d[zones.key];
        const col = zones.colors[z];
        if (col) {
          ctx.fillStyle = col;
          ctx.fillRect(pad.l + i * cw, pad.t, cw, H - pad.t - pad.b);
        }
      });
    }

    // Grid
    ctx.strokeStyle = '#1e2430'; ctx.lineWidth = 1;
    for (let i = 0; i < 4; i++) {
      const y = pad.t + i * (H - pad.t - pad.b) / 3;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      const v = vMax - i * range / 3;
      ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
      ctx.fillText(v.toFixed(1), pad.l - 5, y + 4);
    }

    // Line
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    data.forEach((d, i) => {
      if (d[key] == null) return;
      const x = pad.l + i * cw + cw / 2;
      if (!started) { ctx.moveTo(x, scY(d[key])); started = true; } else ctx.lineTo(x, scY(d[key]));
    });
    ctx.stroke();

    // Date labels
    ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(data.length / 8));
    data.forEach((d, i) => { if (i % step === 0) ctx.fillText(d.date.slice(2, 7), pad.l + i * cw + cw / 2, H - 5); });

    // Label
    ctx.fillStyle = color; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(label, pad.l + 5, 16);
  }, []);

  useEffect(() => {
    if (!histData) return;
    drawLine(rateCanvas.current, histData, 'rate_10y', '#58a6ff', '10Y Treasury Rate', {
      key: 'regime', colors: { LOW: 'rgba(63,185,80,0.08)', MEDIUM: 'rgba(210,153,34,0.06)', HIGH: 'rgba(248,81,73,0.08)' }
    });
    drawLine(vixCanvas.current, histData, 'vix', '#d29922', 'VIX', {
      key: 'vix_regime', colors: { LOW_VIX: 'rgba(63,185,80,0.08)', MED_VIX: 'rgba(210,153,34,0.06)', HIGH_VIX: 'rgba(248,81,73,0.08)' }
    });
    drawLine(spyCanvas.current, histData, 'spy', '#3fb950', 'SPY', {
      key: 'spy_trend', colors: { BULL: 'rgba(63,185,80,0.06)', BEAR: 'rgba(248,81,73,0.06)' }
    });
  }, [histData, drawLine]);

  const sections = [
    {
      title: 'Interest Rates',
      key: 'regime',
      canvasRef: rateCanvas,
      states: [
        { name: 'LOW', color: 'good', desc: '3M T-Bill < 2%. Cheap borrowing fuels growth stocks. Sectors like Technology, Consumer Discretionary, and Small Caps tend to outperform. Low rates push investors into equities seeking yield.' },
        { name: 'MEDIUM', color: 'neutral', desc: '3M T-Bill 2-4%. Balanced environment. Most sectors perform normally. The market is neither helped nor hurt by rates. Quality and diversification matter more.' },
        { name: 'HIGH', color: 'bad', desc: '3M T-Bill > 4%. Expensive borrowing hurts growth. Financials and Energy tend to outperform. High-debt companies struggle. Defensive sectors (Utilities, Staples) can benefit from flight to safety.' },
      ]
    },
    {
      title: 'Yield Curve',
      key: 'curve',
      states: [
        { name: 'NORMAL', color: 'good', desc: '10Y > 3M rate. Healthy economic outlook. Banks profit from the spread. Historically, normal curves precede economic expansion. Risk-on sectors outperform.' },
        { name: 'INVERTED', color: 'bad', desc: '3M > 10Y rate. Historically predicts recession within 12-18 months. Investors fleeing to long bonds signals fear. Defensive sectors and quality stocks outperform. Every US recession since 1950 was preceded by an inversion.' },
      ]
    },
    {
      title: 'VIX (Volatility Index)',
      key: 'vix',
      canvasRef: vixCanvas,
      states: [
        { name: 'LOW (< 15)', color: 'good', desc: 'Market complacency. Stocks grind higher in low vol. Momentum strategies work well. However, low VIX can precede sharp corrections — "calm before the storm." Selling options is profitable.' },
        { name: 'MEDIUM (15-25)', color: 'neutral', desc: 'Normal market conditions. Healthy two-way price action. Most strategies work as expected. Standard risk management applies.' },
        { name: 'HIGH (> 25)', color: 'bad', desc: 'Fear and uncertainty. Sharp moves in both directions. Contrarian buying after VIX spikes historically produces strong returns. Our studies show HIGH VIX entries average +11.2% vs +1.8% in LOW VIX.' },
      ]
    },
    {
      title: 'SPY Trend',
      key: 'spy_trend',
      canvasRef: spyCanvas,
      states: [
        { name: 'BULL', color: 'good', desc: 'SPY above its 50-day SMA. Trend-following strategies work. Most sectors rise with the tide. However, crowded trades and complacency can form. Our studies show surprisingly lower returns (+1.3% avg) during BULL — because entries during pullbacks in BEAR markets produce bigger bounces.' },
        { name: 'BEAR', color: 'bad', desc: 'SPY below its 50-day SMA. Counterintuitively, our backtested signals produce +11.1% avg return when entered during BEAR trends. The market overshoots on fear, and mean-reversion signals fire at better prices.' },
      ]
    },
    {
      title: 'Seasonality (Sell in May)',
      key: 'season',
      states: [
        { name: 'NOV-APR', color: 'good', desc: 'Historically the strongest 6 months. Year-end tax loss harvesting, holiday spending, and fund rebalancing drive flows. The "Santa Rally" and "January Effect" boost returns. S&P 500 has averaged ~7% in this period vs ~2% in May-Oct since 1950.' },
        { name: 'MAY-OCT', color: 'neutral', desc: '"Sell in May and go away." Lower volume, vacation season, and historically weaker returns. September is the worst month on average. However, our studies show both periods produce similar returns (+7.7% vs +5.7%), suggesting the effect is weaker for momentum signals.' },
      ]
    },
  ];

  const Card = ({ label, color, children }) => (
    <div className="settings-card">
      <div className="settings-card-header"><span className={`settings-card-label ${color || 'neutral'}`}>{label}</span></div>
      <div className="settings-card-desc" style={{ fontSize: 13, lineHeight: 1.6 }}>{children}</div>
    </div>
  );

  return (
    <div className="studies-page">
      <h1>Documentation</h1>
      <p className="subtitle">What this system does, how every tab works, and what every number means. Hover any <span className="term"><span className="term-q">?</span></span> across the app for an inline definition.</p>

      <div className="settings-section" style={{ marginTop: 8 }}>
        <h2>What this is</h2>
        <div className="settings-grid">
          <Card label="The thesis" color="good">
            A quant workbench for <b>sector rotation</b> and <b>mean-reversion stock-picking</b>. It watches 93 sector ETFs
            (+ SPY/QQQ) and ~1,000 individual stocks, ranks them by risk-adjusted momentum, and backtests thousands of
            entry×exit rules to find edges that survive robustness checks. A second pillar reads <b>news</b> as a signal:
            which headlines actually moved a stock, what type of event, and how it drifted afterward.
          </Card>
          <Card label="How to read it" color="neutral">
            Start on the <b>Rotation Dashboard</b> (which sectors are turning), drop into a sector to see its holdings,
            then use <b>Live Signals</b> for what's firing today and <b>Research</b> for the backtests behind it.
            <b> News</b> tells you why something moved. Everything is descriptive analytics — <i>not</i> financial advice.
          </Card>
          <Card label="Everything is local" color="good">
            All data and models run on-box or in Docker. News is categorized by a <b>local LLM</b> (qwen2.5:14b via Ollama) —
            no headlines are sent to any cloud API. Credentials live only in a gitignored <code>.env</code>.
          </Card>
        </div>
      </div>

      <div className="settings-section" style={{ marginTop: 24 }}>
        <h2>The tabs</h2>
        <div className="settings-grid">
          <Card label="Rotation Dashboard">Sector ETFs ranked by RSI / Sortino / Omega / capture ratios vs SPY, with the current BULLISH / ROTATE-IN signal and any recent price gap. Click a sector for its holdings.</Card>
          <Card label="⚡ Live Signals">Three views of what's actionable now: the <b>Playbook</b> (end-to-end funnel — rotating sectors → capitulations smart money is buying), <b>Firing Now</b> (stocks triggering the top robust signals today), and <b>A/D Divergence</b> (quiet accumulation).</Card>
          <Card label="📰 News">Real, market-moving news. <b>News Effect</b> = headlines that actually moved the stock, with category, same-day move, and results 1mo/3mo/1yr later. <b>Clusters</b> = burst/"propping" detection. <b>Horizon</b> = the validated "good news fades" edge.</Card>
          <Card label="🔬 Research">The lab behind the signals: the sector Study engine, the all-on-all stock sweep with fundamental buckets, trend-rotation studies, a portfolio backtester, and dimension-intersection tests.</Card>
          <Card label="Stock Drilldown">Per-stock deep dives and the individual-stock study results.</Card>
          <Card label="Trade Journal">A place to log and review decisions.</Card>
        </div>
      </div>

      <div className="settings-section" style={{ marginTop: 24 }}>
        <h2>Signals &amp; studies</h2>
        <div className="settings-grid">
          <Card label="BULLISH" color="good">RSI(10) &gt; its SMA(10) <b>and</b> Omega(10) &gt; 1 — momentum is up and the gain/loss odds favor gains.</Card>
          <Card label="TREND TURN" color="neutral">BULLISH <b>and</b> RSI just crossed above its SMA within the last 3 days — a fresh up-turn in the sector's <b>own price</b>. An absolute-trend alert, <b>not</b> a beat-SPY rotation call (it backtests as the worst beat-SPY signal). The real rotation call — regime-leader sectors ∩ cheapest-P/B value pick, entered on an oversold dip — is on the <b>Rotation Call</b> tab.</Card>
          <Card label="FRESH composite" color="neutral">Weekly Sortino &gt; 0 + an RSI crossover + an RSI-of-Sortino crossover, all within the last 14 trading days. 3/3 = FRESH, 2/3 = POTENTIAL.</Card>
          <Card label="The studies engine" color="neutral">Every signal is paired with every exit (hundreds × ~70) and backtested across 5 years. Each trade is tagged with the market regime at entry, so you can filter for what works in <i>today's</i> conditions.</Card>
          <Card label="Robustness gate" color="bad">A signal is only trusted after it survives disaster-rate, forward-return, and time+size checks. Crypto was removed for inflating results; survivorship bias makes backtests an optimistic ceiling. Directional, not guaranteed.</Card>
          <Card label="Smart-money overlay" color="good">SEC 13D (activist), 13G (passive institutional), and insider open-market buys flag conviction the price hasn't shown yet. Click any badge for the individual filings.</Card>
        </div>
      </div>

      <div className="settings-section" style={{ marginTop: 24 }}>
        <h2>News intelligence</h2>
        <div className="settings-grid">
          <Card label="Same-day effect" color="neutral">A β-adjusted abnormal move over the news's reaction session (prior close → reaction close, so it includes the overnight/pre-market gap). "Moved" = ≥2σ vs the stock's own recent volatility (3% floor).</Card>
          <Card label="Junk filter" color="bad">Title-only heuristic that drops opinion/clickbait, performance recaps, award-PR fluff, and multi-ticker roundups ("| Stock Movers") — the "nothing happened / not about this ticker" headlines. ~21% of the corpus.</Card>
          <Card label="Category" color="neutral">Each headline is typed — earnings, guidance, analyst, M&amp;A, partnership, contract, product, clinical, legal, offering, dividend, buyback, insider, mgmt, macro. A title heuristic covers all ~479K; a <b>local LLM</b> refines the ones that moved (✦).</Card>
          <Card label="Results since" color="good">For a past move, the raw total return 1 month / 3 months / 1 year later. Recent items show "pending" until the window matures. Reveals the fade: good-news pops tend to give back over months.</Card>
          <Card label="Clusters" color="neutral">Bursts of headlines on one ticker in a short window — the footprint of a promotion. A high prop-score (volume + bullish skew + junk share + an up push) that then fades is a pump-then-fizzle.</Card>
          <Card label="Horizon / drift" color="bad">The one durable, validated edge in our data: good news FADES, hardest in mid/small caps; bad-news drops partly rebound. Not wired into sizing until it clears the robustness gate.</Card>
        </div>
      </div>

      <div className="settings-section" style={{ marginTop: 24 }}>
        <h2>Glossary — every metric</h2>
        <p className="subtitle">The same definitions shown by the <span className="term"><span className="term-q">?</span></span> icons throughout the app.</p>
        <table className="studies-table">
          <thead><tr><th>Term</th><th>Definition</th></tr></thead>
          <tbody>
            {Object.entries(GLOSSARY).sort((a, b) => a[1][0].localeCompare(b[1][0])).map(([k, [label, def]]) => (
              <tr key={k}><td style={{ whiteSpace: 'nowrap', fontWeight: 700 }}>{label}</td><td className="dim" style={{ fontSize: 13 }}>{def}</td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2 style={{ marginTop: 28 }}>Market regimes</h2>
      <p className="subtitle">The five environment indicators every backtested trade is tagged with, so you can filter for what works now.</p>

      {sections.map(sec => (
        <div key={sec.key} className="settings-section" style={{marginTop: 24}}>
          <h2>{sec.title}</h2>
          {sec.canvasRef && (
            <div className="chart-container" style={{marginBottom: 12}}>
              {histData ? <canvas ref={sec.canvasRef} /> : <div className="loading">Loading chart...</div>}
            </div>
          )}
          <div className="settings-grid">
            {sec.states.map(st => (
              <div key={st.name} className="settings-card">
                <div className="settings-card-header">
                  <span className={`settings-card-label ${st.color}`}>{st.name}</span>
                </div>
                <div className="settings-card-desc">{st.desc}</div>
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="settings-section" style={{marginTop: 24}}>
        <h2>How We Use Regimes</h2>
        <div className="settings-card" style={{maxWidth: 600}}>
          <div className="settings-card-desc" style={{fontSize: 13, lineHeight: 1.6}}>
            Each study is backtested across all 5 years of data. We tag every trade with the regime
            that was active at entry: rate level, yield curve shape, VIX level, SPY trend, and season.
            This lets you filter studies to find strategies that work specifically in the current environment.
            <br/><br/>
            For example, if VIX is HIGH and SPY is in BEAR mode, you can filter for studies that historically
            performed well in those conditions — often contrarian signals like RSI oversold + volume spike.
          </div>
        </div>
      </div>
    </div>
  );
}

function StockStudiesPage() {
  const [search, setSearch] = useState('');
  const [dsearch, setDsearch] = useState('');          // debounced -> server query
  const [catFilter, setCatFilter] = useState('all');
  const [sortBy, setSortBy] = useState('avg_return');
  const [sortDir, setSortDir] = useState('desc');
  const [minTrades, setMinTrades] = useState(30);
  const [sliceDim, setSliceDim] = useState('');      // reframe table by a PIT dimension...
  const [sliceBucket, setSliceBucket] = useState(''); // ...bucket
  const [expanded, setExpanded] = useState(null);
  const [running, setRunning] = useState(false);
  const [sectors, setSectors] = useState([]);
  const [refreshKey, setRefreshKey] = useState(0);      // bump to force a refetch (post-sweep)
  const [drillSector, setDrillSector] = useState({});   // rowKey -> sector name
  const [drillData, setDrillData] = useState({});        // rowKey -> result
  const [drillLoading, setDrillLoading] = useState(null); // rowKey currently loading
  const [firingSignal, setFiringSignal] = useState(null);  // open FiringPane for this signal key

  // Debounce search so a keystroke doesn't re-query the 24k-row table on every character.
  useEffect(() => { const t = setTimeout(() => setDsearch(search), 300); return () => clearTimeout(t); }, [search]);

  // Server-side filter (search/category/min_trades) + sort + infinite scroll. `refreshKey` is an
  // ignored-by-server param that forces a refetch after a sweep completes.
  const params = React.useMemo(() => ({
    ordering: sortBy, dir: sortDir, search: dsearch,
    category: catFilter === 'all' ? '' : catFilter, min_trades: minTrades, refreshKey,
  }), [sortBy, sortDir, dsearch, catFilter, minTrades, refreshKey]);
  const { rows: pageRows, meta, loading, error, hasMore, loadMore } = usePagedList('/stock-studies', params);

  useEffect(() => {
    fetch(`${API}/sectors`).then(r => r.json()).then(d => setSectors(d || [])).catch(() => {});
  }, []);

  const runDrill = (rowKey, r, sector) => {
    if (!sector) return;
    setDrillSector(s => ({ ...s, [rowKey]: sector }));
    setDrillLoading(rowKey);
    const p = new URLSearchParams({ sector, signal: r.signal_key, exit: r.exit_key, recent: '15' });
    fetch(`${API}/sector-drilldown?${p}`).then(r => r.json())
      .then(d => { setDrillData(s => ({ ...s, [rowKey]: d })); setDrillLoading(null); })
      .catch(() => setDrillLoading(null));
  };

  const runSweep = () => {
    setRunning(true);
    fetch(`${API}/stock-studies`, { method: 'POST' })
      .then(r => r.json()).then(() => {
        // Poll for freshness; sweep takes minutes.
        const t = setInterval(() => {
          fetch(`${API}/stock-studies?limit=1`).then(r => r.json()).then(d => {
            if (d.computed) { clearInterval(t); setRunning(false); setRefreshKey(k => k + 1); }
          }).catch(() => {});
        }, 15000);
      }).catch(() => setRunning(false));
  };

  if (error && !pageRows.length) return <div className="studies-page"><ErrorBanner message={error} /></div>;
  if (loading && !pageRows.length) return <div className="loading">Loading stock studies...</div>;

  if (meta.computed === false) {
    return (
      <div className="studies-page">
        <h1>Stock Indicator Studies</h1>
        <p className="subtitle">Every signal × every exit, run across all individual stocks, with fundamental-bucket breakdowns.</p>
        <div className="empty-state" style={{padding:'40px 0'}}>
          <p>{meta.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={runSweep} disabled={running}>
            {running ? 'Running sweep (minutes)...' : 'Run all-on-all sweep'}
          </button>
        </div>
      </div>
    );
  }

  const dimMeta = meta.dimension_meta || {};
  const isSnapshot = (dim) => dimMeta[dim] && dimMeta[dim].pit === false;

  // Slice-by-dimension: reframe every row to a chosen dimension+bucket's stats.
  const slicing = sliceDim && sliceBucket;
  const bucketOf = (r, dim, bucket) => (r.by_dimension?.[dim] || []).find(b => b.bucket === bucket) || null;
  // Metrics shown/sorted: the sliced bucket's when slicing, else the row's overall.
  const metricsOf = (r) => {
    if (slicing) { const b = bucketOf(r, sliceDim, sliceBucket); return b ? { avg_return: b.avg_return, win_rate: b.win_rate, trades: b.trades, avg_mae: b.avg_mae, clean_pct: b.clean_pct } : null; }
    return { avg_return: r.avg_return, win_rate: r.win_rate, trades: r.trades, avg_mae: r.avg_mae, clean_pct: r.clean_pct };
  };

  const categories = meta.categories || [];
  // Server already applied search / category / min_trades / sort. Slice-by-dimension is a client-side
  // reframe of the LOADED rows (each row carries its own by_dimension), so when a slice is active we
  // drop rows missing that bucket and re-sort the loaded rows by the sliced metric.
  let rows = pageRows.map(r => ({ r, m: metricsOf(r) }));
  if (slicing) {
    rows = rows.filter(({ m }) => m);
    rows.sort((x, y) => {
      const av = x.m[sortBy] ?? x.r[sortBy] ?? 0, bv = y.m[sortBy] ?? y.r[sortBy] ?? 0;
      if (typeof av === 'string') return sortDir === 'desc' ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
      return sortDir === 'desc' ? bv - av : av - bv;
    });
  }

  const setSort = (col) => {
    if (sortBy === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    else { setSortBy(col); setSortDir('desc'); }
  };
  const arrow = (col) => sortBy === col ? (sortDir === 'desc' ? ' ▼' : ' ▲') : '';

  // PIT dimensions available to slice by, and buckets for the chosen one (from loaded rows).
  const pitDims = Object.keys(dimMeta).filter(d => !isSnapshot(d));
  const sliceBuckets = sliceDim
    ? [...new Set(pageRows.flatMap(r => (r.by_dimension?.[sliceDim] || []).map(b => b.bucket)))].filter(x => x !== 'NA').sort()
    : [];

  // Best-amplifying POINT-IN-TIME bucket (excludes snapshot dims + the NA no-data bucket,
  // which otherwise surfaces a data-coverage artifact rather than a real edge).
  const bestBucket = (r) => {
    let best = null;
    for (const [dim, brows] of Object.entries(r.by_dimension || {})) {
      if (isSnapshot(dim)) continue;
      for (const b of brows) {
        if (b.bucket === 'NA') continue;
        if (best === null || b.avg_return > best.avg_return) best = { ...b, dim };
      }
    }
    return best;
  };

  return (
    <div className="studies-page">
      <h1>Stock Indicator Studies <span className="dim">({rows.length.toLocaleString()} of {(meta.total ?? meta.n_results ?? 0).toLocaleString()})</span>
        <LastUpdatedChip value={meta.last_updated} />
      </h1>
      <p className="subtitle">
        {(meta.total ?? meta.n_results ?? 0).toLocaleString()} signal × exit combos over {meta.universe_size} stocks (5y daily).
        Expand a row for the fundamental-bucket breakdown. Sorted &amp; filtered server-side; scroll to load more.
      </p>

      <div className="studies-controls">
        <input className="studies-search" type="text" placeholder="Search signal/exit..." value={search} onChange={e => setSearch(e.target.value)} />
        <div className="filters">
          <button className={catFilter === 'all' ? 'active' : ''} onClick={() => setCatFilter('all')}>All</button>
          {categories.map(c => (
            <button key={c} className={catFilter === c ? 'active' : ''} onClick={() => setCatFilter(c)}>{c}</button>
          ))}
        </div>
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Min trades:</span>
          {[20, 30, 50, 100, 250].map(m => (
            <button key={m} className={minTrades === m ? 'active' : ''} onClick={() => setMinTrades(m)}>{m}</button>
          ))}
          <span className="filter-sep">|</span>
          <button className="refresh-btn" onClick={runSweep} disabled={running}>{running ? 'Running...' : 'Re-run sweep'}</button>
        </div>
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Slice by dimension:</span>
          <select value={sliceDim} onChange={e => { setSliceDim(e.target.value); setSliceBucket(''); }}>
            <option value="">(overall)</option>
            {pitDims.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
          {sliceDim && (
            <select value={sliceBucket} onChange={e => setSliceBucket(e.target.value)}>
              <option value="">choose bucket…</option>
              {sliceBuckets.map(b => <option key={b} value={b}>{b}</option>)}
            </select>
          )}
          {slicing && <span className="dim-badge" style={{background:'#0d2847',color:'#1f6feb'}} title="Avg Ret / Win% / Trades below are for this dimension bucket, point-in-time">showing: {sliceDim} = {sliceBucket}</span>}
        </div>
      </div>

      <table className="studies-table">
        <thead>
          <tr>
            <th style={{width:24}}></th>
            <th onClick={() => setSort('signal_name')} style={{cursor:'pointer'}}><Term k="signal">Signal</Term>{arrow('signal_name')}</th>
            <th onClick={() => setSort('exit_name')} style={{cursor:'pointer'}}><Term k="exit">Exit</Term>{arrow('exit_name')}</th>
            <th onClick={() => setSort('category')} style={{cursor:'pointer'}}><Term k="category">Category</Term>{arrow('category')}</th>
            <th onClick={() => setSort('avg_return')} style={{cursor:'pointer', textAlign:'right'}}><Term k="avgreturn">Avg Ret</Term>{arrow('avg_return')}</th>
            <th onClick={() => setSort('win_rate')} style={{cursor:'pointer', textAlign:'right'}}><Term k="winrate">Win%</Term>{arrow('win_rate')}</th>
            <th onClick={() => setSort('trades')} style={{cursor:'pointer', textAlign:'right'}}><Term k="ntrades">Trades</Term>{arrow('trades')}</th>
            <th onClick={() => setSort('t_stat')} style={{cursor:'pointer', textAlign:'right'}}><Term k="sig">Sig (t)</Term>{arrow('t_stat')}</th>
            <th onClick={() => setSort('avg_hold')} style={{cursor:'pointer', textAlign:'right'}}><Term k="avghold">Hold</Term>{arrow('avg_hold')}</th>
            <th onClick={() => setSort('avg_mae')} style={{cursor:'pointer', textAlign:'right'}}><Term k="avgdip">Avg Dip</Term>{arrow('avg_mae')}</th>
            <th onClick={() => setSort('clean_pct')} style={{cursor:'pointer', textAlign:'right'}}><Term k="cleanpct">Clean%</Term>{arrow('clean_pct')}</th>
            <th><Term k="bestfund">Best fundamental bucket</Term></th>
            <th onClick={() => setSort('n_firing')} style={{cursor:'pointer', textAlign:'right'}} title="Names firing this signal in the last 3 bars">Firing 3d{arrow('n_firing')}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ r, m }, i) => {
            const key = `${r.signal_key}|${r.exit_key}`;
            const isOpen = expanded === key;
            return (
              <React.Fragment key={key}>
                <tr className="study-row" onClick={() => setExpanded(isOpen ? null : key)} style={{cursor:'pointer'}}>
                  <td>{isOpen ? '▾' : '▸'}</td>
                  <td title={r.signal_key}>{r.signal_name}</td>
                  <td title={r.exit_key}>{r.exit_name}</td>
                  <td><span className="dim">{r.category}</span></td>
                  <td style={{textAlign:'right'}} className={m.avg_return > 0 ? 'good' : 'bad'}><b>{m.avg_return > 0 ? '+' : ''}{m.avg_return}%</b></td>
                  <td style={{textAlign:'right'}}>{m.win_rate}%</td>
                  <td style={{textAlign:'right'}}>{m.trades}</td>
                  <td style={{textAlign:'right'}} className={r.t_stat == null ? 'dim' : Math.abs(r.t_stat) >= 2 ? 'good' : Math.abs(r.t_stat) < 1 ? 'bad' : ''}
                      title={r.eff_trades != null ? `${r.eff_trades.toLocaleString()} independent (overlap-deduped) trades` : ''}>{r.t_stat != null ? r.t_stat.toFixed(1) : '–'}</td>
                  <td style={{textAlign:'right'}}>{r.avg_hold}d</td>
                  <td style={{textAlign:'right'}} className={m.avg_mae == null ? 'dim' : m.avg_mae >= -3 ? 'good' : m.avg_mae >= -8 ? '' : 'bad'}>{m.avg_mae != null ? `${m.avg_mae.toFixed(1)}%` : '–'}</td>
                  <td style={{textAlign:'right'}} className={m.clean_pct == null ? 'dim' : m.clean_pct >= 40 ? 'good' : m.clean_pct < 20 ? 'bad' : ''}>{m.clean_pct != null ? `${m.clean_pct.toFixed(0)}%` : '–'}</td>
                  <td>{(() => { const b = bestBucket(r); return b ? <span className="drill-chip">{b.bucket} <span className="good">+{b.avg_return}%</span> <span className="dim">({b.trades}tr)</span>{isSnapshot(b.dim) && <span className="dim-badge" title="Uses today's snapshot — carries lookahead bias, not point-in-time">snapshot</span>}</span> : <span className="dim">–</span>; })()}</td>
                  <td style={{textAlign:'right'}}><FiringCell count={r.n_firing} signalKey={r.signal_key} onOpen={setFiringSignal} /></td>
                </tr>
                {isOpen && (
                  <tr className="study-detail-row">
                    <td colSpan={13}>
                      {(() => {
                        const capEntry = Object.entries(r.by_dimension || {}).find(([dim]) => /cap/i.test(dim));
                        if (!capEntry) return null;
                        const capBuckets = (capEntry[1] || [])
                          .filter(b => b.bucket !== 'NA')
                          .map(b => ({ label: b.bucket, value: b.avg_return, n: b.trades }));
                        if (!capBuckets.length) return null;
                        return (
                          <div style={{maxWidth:420, marginBottom:10}}>
                            <div className="subtitle darkpool-muted" style={{margin:'2px 0'}}>Return by market-cap bucket (tail)</div>
                            <TailStrip buckets={capBuckets} />
                          </div>
                        );
                      })()}
                      <div className="fund-dim-title" style={{marginBottom:8}}>Fundamental buckets</div>
                      <div className="fund-buckets">
                        {Object.keys(r.by_dimension || {}).length === 0 && <span className="dim">No fundamental buckets cleared the trade floor.</span>}
                        {Object.entries(r.by_dimension || {}).map(([dim, buckets]) => (
                          <div className="fund-dim" key={dim}>
                            <div className="fund-dim-title">{dim}{isSnapshot(dim) && <span className="dim-badge" title="Uses today's snapshot — carries lookahead bias, not point-in-time">snapshot</span>}</div>
                            <table className="fund-dim-table">
                              <tbody>
                                {buckets.map(b => (
                                  <tr key={b.bucket}>
                                    <td>{b.bucket}</td>
                                    <td className={b.avg_return > 0 ? 'good' : 'bad'} style={{textAlign:'right'}}>{b.avg_return > 0 ? '+' : ''}{b.avg_return}%</td>
                                    <td style={{textAlign:'right'}}>{b.win_rate}%</td>
                                    <td style={{textAlign:'right'}} className="dim">{b.trades}tr</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        ))}
                      </div>

                      <div className="drill-section">
                        <div className="fund-dim-title" style={{marginBottom:8}}>
                          Sector → stocks: which stocks in a sector fire this signal
                        </div>
                        <div className="filters" style={{marginBottom:10}}>
                          <select className="studies-search" style={{maxWidth:280}} value={drillSector[key] || ''}
                            onChange={e => runDrill(key, r, e.target.value)}>
                            <option value="">Pick a sector…</option>
                            {sectors.map(s => <option key={s.etf} value={s.name}>{s.name} ({s.etf})</option>)}
                          </select>
                          {drillLoading === key && <span className="dim">Running…</span>}
                        </div>
                        {drillData[key] && !drillData[key].error && (
                          <div>
                            <p className="dim" style={{margin:'0 0 8px'}}>
                              {drillData[key].n_firing} of {drillData[key].n_stocks} holdings firing now (last {drillData[key].recent_window} bars) · {drillData[key].exit_name}
                            </p>
                            {drillData[key].firing_now.length > 0 && (
                              <div style={{marginBottom:10}}>
                                <span className="good" style={{fontWeight:600}}>Firing now: </span>
                                {drillData[key].firing_now.map(s => (
                                  <span key={s.ticker} className="drill-chip">{s.ticker} <span className="dim">({s.days_ago}d ago, {s.avg_return > 0 ? '+' : ''}{s.avg_return}%)</span></span>
                                ))}
                              </div>
                            )}
                            <table className="fund-dim-table" style={{maxWidth:640}}>
                              <thead><tr><th style={{textAlign:'left'}}><Term k="stock">Stock</Term></th><th style={{textAlign:'right'}}><Term k="avgreturn">Avg Ret</Term></th><th style={{textAlign:'right'}}><Term k="winrate">Win%</Term></th><th style={{textAlign:'right'}}><Term k="ntrades">Trades</Term></th><th style={{textAlign:'right'}}><Term k="avgdip">Avg Dip</Term></th><th style={{textAlign:'right'}}><Term k="cleanpct">Clean%</Term></th><th style={{textAlign:'right'}}><Term k="fired">Firing</Term></th></tr></thead>
                              <tbody>
                                {drillData[key].stocks.filter(s => s.trades > 0).slice(0, 20).map(s => (
                                  <tr key={s.ticker}>
                                    <td>{s.ticker}</td>
                                    <td className={s.avg_return > 0 ? 'good' : 'bad'} style={{textAlign:'right'}}>{s.avg_return > 0 ? '+' : ''}{s.avg_return}%</td>
                                    <td style={{textAlign:'right'}}>{s.win_rate}%</td>
                                    <td style={{textAlign:'right'}} className="dim">{s.trades}</td>
                                    <td style={{textAlign:'right'}} className={s.avg_mae == null ? 'dim' : s.avg_mae >= -3 ? 'good' : s.avg_mae >= -8 ? '' : 'bad'}>{s.avg_mae != null ? `${s.avg_mae.toFixed(1)}%` : '–'}</td>
                                    <td style={{textAlign:'right'}} className={s.clean_pct == null ? 'dim' : s.clean_pct >= 40 ? 'good' : s.clean_pct < 20 ? 'bad' : ''}>{s.clean_pct != null ? `${s.clean_pct.toFixed(0)}%` : '–'}</td>
                                    <td style={{textAlign:'right'}}>{s.firing ? <span className="good">●</span> : <span className="dim">–</span>}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                        {drillData[key] && drillData[key].error && <span className="dim">{drillData[key].error}</span>}
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
      {!slicing && hasMore && <ScrollSentinel onVisible={loadMore} disabled={loading} />}
      {loading && pageRows.length > 0 && <div className="loading dim" style={{padding:12}}>Loading more…</div>}
      {slicing && <p className="dim" style={{marginTop:8}}>Slice view reframes the {pageRows.length} currently-loaded rows; scroll with the slice cleared to load more.</p>}
      {!slicing && !hasMore && pageRows.length > 0 && <div className="dim" style={{padding:12,textAlign:'center'}}>All {(meta.total ?? pageRows.length).toLocaleString()} loaded.</div>}
      <FiringPane signalKey={firingSignal} onClose={() => setFiringSignal(null)} />
    </div>
  );
}

function FiringNowPage() {
  const [sigFilter, setSigFilter] = useState('all');
  const [sectorFilter, setSectorFilter] = useState('all');
  const [maxDays, setMaxDays] = useState(5);
  const [running, setRunning] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);   // bump to force a refetch (post-scan)
  const [peMap, setPeMap] = useState({});
  // Default sort = most-recently-fired first. Filter + sort now happen server-side; scroll loads more.
  const [sortBy, setSortBy] = useState('days_ago');
  const [sortDir, setSortDir] = useState('asc');

  const params = React.useMemo(() => ({
    ordering: sortBy, dir: sortDir,
    signal: sigFilter === 'all' ? '' : sigFilter,
    sector: sectorFilter === 'all' ? '' : sectorFilter,
    max_days: maxDays, refreshKey,
  }), [sortBy, sortDir, sigFilter, sectorFilter, maxDays, refreshKey]);
  const { rows, meta, loading, error, hasMore, loadMore } = usePagedList('/live-signals', params);

  // P/E is sourced from the shared /fundamentals feed by ticker (fallback to the row's own pe_ratio).
  useEffect(() => {
    apiFetch('/fundamentals')
      .then(d => {
        const fr = (d && d.fundamentals) || [];
        const m = {};
        fr.forEach(x => {
          const v = x.forward_pe != null ? x.forward_pe : x.pe_ratio;
          if (x.ticker != null && v != null) m[x.ticker] = v;
        });
        setPeMap(m);
      })
      .catch(() => {});
  }, []);

  const runScan = () => {
    setRunning(true);
    fetch(`${API}/live-signals`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/live-signals?limit=1`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setRefreshKey(k => k + 1); }
      }).catch(() => {}), 8000);
    }).catch(() => {});
  };

  // Shim so the existing <SortTh> headers drive the server-side sort state.
  const setSort = (col) => { if (sortBy === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc'); else { setSortBy(col); setSortDir('desc'); } };
  const fSort = { sortKey: sortBy, sortDir, requestSort: setSort };

  if (error && !rows.length) return <div className="studies-page"><ErrorBanner message={error} /></div>;
  if (loading && !rows.length) return <div className="loading">Loading firing signals...</div>;
  if (meta.computed === false) {
    return (
      <div className="studies-page">
        <h1>Firing Now</h1>
        <p className="subtitle">Stocks currently triggering the top signals across all sectors.</p>
        <div className="empty-state" style={{padding:'40px 0'}}>
          <p>{meta.message || 'Not scanned yet.'}</p>
          <button className="refresh-btn" onClick={runScan} disabled={running}>{running ? 'Scanning...' : 'Run firing scan'}</button>
        </div>
      </div>
    );
  }

  // Server already filtered by signal/sector/max_days; derive sector heat from the loaded rows.
  const sectorCounts = {};
  rows.forEach(r => (r.sectors || []).forEach(s => { sectorCounts[s] = (sectorCounts[s] || 0) + 1; }));
  const sectorHeat = Object.entries(sectorCounts).sort((a, b) => b[1] - a[1]).slice(0, 12);
  const allSectors = [...new Set(rows.flatMap(r => r.sectors || []))].sort();

  return (
    <div className="studies-page">
      <h1>Firing Now <span className="dim">({rows.length.toLocaleString()} of {(meta.total ?? 0).toLocaleString()}{meta.n_firing != null ? ` · ${meta.n_firing.toLocaleString()} firing` : ''})</span>
        <LastUpdatedChip value={meta.last_updated || meta.computed_at} />
      </h1>
      <p className="subtitle">
        Stocks triggering a top signal within the last {maxDays} bars, with the signal's historical edge + fundamentals.
        {meta.computed_at && <span className="dim"> · scanned {new Date(meta.computed_at).toLocaleString()}</span>}
      </p>

      <div className="sector-heat">
        {sectorHeat.map(([s, n]) => (
          <button key={s} className={`drill-chip ${sectorFilter === s ? 'active-chip' : ''}`} onClick={() => setSectorFilter(sectorFilter === s ? 'all' : s)}>
            {s} <span className="good">{n}</span>
          </button>
        ))}
      </div>

      <div className="studies-controls">
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Signal:</span>
          <button className={sigFilter === 'all' ? 'active' : ''} onClick={() => setSigFilter('all')}>All</button>
          {(meta.signals || []).map(s => <button key={s} className={sigFilter === s ? 'active' : ''} onClick={() => setSigFilter(s)}>{s}</button>)}
        </div>
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Fired within:</span>
          {[0, 1, 2, 5, 10].map(d => <button key={d} className={maxDays === d ? 'active' : ''} onClick={() => setMaxDays(d)}>{d === 0 ? 'today' : d + 'd'}</button>)}
          <span className="filter-sep">|</span>
          <select className="studies-search" style={{maxWidth:220}} value={sectorFilter} onChange={e => setSectorFilter(e.target.value)}>
            <option value="all">All sectors</option>
            {allSectors.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <button className="refresh-btn" onClick={runScan} disabled={running}>{running ? 'Scanning...' : 'Re-scan'}</button>
        </div>
      </div>

      <table className="studies-table">
        <thead><tr>
          <SortTh colKey="ticker" sort={fSort}><Term k="ticker">Ticker</Term></SortTh><SortTh colKey="signal_name" sort={fSort}><Term k="signal">Signal</Term></SortTh><SortTh colKey="days_ago" sort={fSort} align="right"><Term k="fired">Fired</Term></SortTh>
          <SortTh colKey="hist_avg_return" sort={fSort} align="right"><Term k="histedge">Hist Ret</Term></SortTh><SortTh colKey="hist_win_rate" sort={fSort} align="right"><Term k="winrate">Hist Win%</Term></SortTh>
          <SortTh colKey="hist_avg_mae" sort={fSort} align="right"><Term k="avgdip">Hist Dip</Term></SortTh><SortTh colKey="hist_clean_pct" sort={fSort} align="right"><Term k="cleanpct">Clean%</Term></SortTh>
          <SortTh colKey="last_close" sort={fSort} align="right"><Term k="lastclose">Last</Term></SortTh><th><Term k="marketcap">Mkt Cap</Term></th><SortTh colKey="forward_pe" sort={fSort} align="right"><Term k="pe">P/E</Term></SortTh><th><Term k="smartmoney">Smart money</Term></th><th><Term k="sector">Sectors</Term></th>
        </tr></thead>
        <tbody>
          {rows.map(r => (
            <tr key={`${r.ticker}|${r.signal_key}`} className="study-row">
              <td><b>{r.ticker}</b></td>
              <td title={r.signal_key} className="dim">{r.signal_name}</td>
              <td style={{textAlign:'right'}}>{r.days_ago === 0 ? <span className="good">today</span> : `${r.days_ago}d`}</td>
              <td style={{textAlign:'right'}} className={r.hist_avg_return > 0 ? 'good' : 'bad'}>{r.hist_avg_return > 0 ? '+' : ''}{r.hist_avg_return}%</td>
              <td style={{textAlign:'right'}}>{r.hist_win_rate}%</td>
              <td style={{textAlign:'right'}} className={r.hist_avg_mae == null ? 'dim' : r.hist_avg_mae >= -3 ? 'good' : r.hist_avg_mae >= -8 ? '' : 'bad'}>{r.hist_avg_mae != null ? `${r.hist_avg_mae.toFixed(1)}%` : '–'}</td>
              <td style={{textAlign:'right'}} className={r.hist_clean_pct == null ? 'dim' : r.hist_clean_pct >= 40 ? 'good' : r.hist_clean_pct < 20 ? 'bad' : ''}>{r.hist_clean_pct != null ? `${r.hist_clean_pct.toFixed(0)}%` : '–'}</td>
              <td style={{textAlign:'right'}} className="dim">{r.last_close}</td>
              <td>{(r.fund_buckets || {})['Market cap'] || <span className="dim">–</span>}</td>
              <td style={{textAlign:'right'}} className="dim">{(() => {
                const v = peMap[r.ticker] != null ? peMap[r.ticker] : r.pe_ratio;
                return v != null ? v.toFixed(1) : '–';
              })()}</td>
              <td style={{fontSize:11}}>{(() => {
                const fmt = v => v >= 1e9 ? `$${(v/1e9).toFixed(1)}B` : v >= 1e6 ? `$${(v/1e6).toFixed(1)}M` : `$${(v/1e3).toFixed(0)}K`;
                const p = [];
                if (r.insider_buy_90d > 0) p.push(<span key="i" className="sm-badge sm-insider" title="Insider open-market buys, ~trailing 6mo (SEC bulk data lags ~1 quarter)">▲ {fmt(r.insider_buy_90d)} insider</span>);
                if (r.recent_13d > 0) p.push(<span key="d" className="sm-badge sm-13d" title="Activist 13D filings, last 180d">13D×{r.recent_13d}</span>);
                if (r.recent_13g > 0) p.push(<span key="g" className="sm-badge sm-13g" title="Passive institutional 13G filings, last 180d">13G×{r.recent_13g}</span>);
                return p.length ? p : <span className="dim">–</span>;
              })()}</td>
              <td className="dim">{(r.sectors || []).slice(0, 2).join(', ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {hasMore && <ScrollSentinel onVisible={loadMore} disabled={loading} />}
      {loading && rows.length > 0 && <div className="loading dim" style={{padding:12}}>Loading more…</div>}
      {!hasMore && rows.length > 0 && <div className="dim" style={{padding:12,textAlign:'center'}}>All {(meta.total ?? rows.length).toLocaleString()} loaded.</div>}
    </div>
  );
}

function NewsHorizonPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [stance, setStance] = useState('FADE');
  const [robustOnly, setRobustOnly] = useState(true);
  const [catFilter, setCatFilter] = useState('all');
  const [running, setRunning] = useState(false);
  const hSort = useSortedRows(data && data.results, null, 'desc');

  const load = () => fetch(`${API}/news-horizon?limit=2000`).then(r => r.json()).then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);

  const runScan = () => {
    setRunning(true);
    fetch(`${API}/news-horizon`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/news-horizon?limit=1`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); load(); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="error">Error: {err}</div>;
  if (!data) return <div className="loading">Loading news-horizon signals...</div>;
  if (!data.computed) {
    return (
      <div className="studies-page">
        <h1>News Horizon</h1>
        <p className="subtitle">Recent material news, evaluated over the horizon that matters for its type.</p>
        <div className="empty-state" style={{padding:'40px 0'}}>
          <p>{data.message || 'Not scanned yet.'}</p>
          <button className="refresh-btn" onClick={runScan} disabled={running}>{running ? 'Scanning...' : 'Run news-horizon scan'}</button>
        </div>
      </div>
    );
  }

  let rows = hSort.rows.filter(r => {
    if (stance !== 'all' && r.stance !== stance) return false;
    if (robustOnly && !r.robust) return false;
    if (catFilter !== 'all' && r.cat !== catFilter) return false;
    return true;
  });

  const hz = { day: 'day', week: 'week', month: '1mo', '3mo': '3mo' };
  const dirTag = d => d > 0 ? <span className="good">▲ bull</span> : d < 0 ? <span className="bad">▼ bear</span> : <span className="dim">– neu</span>;
  const capFmt = v => v == null ? '–' : v >= 1e9 ? `$${(v/1e9).toFixed(1)}B` : `$${(v/1e6).toFixed(0)}M`;

  return (
    <div className="studies-page">
      <h1>News Horizon <span className="dim">({rows.length} shown / {data.n_total}, {data.n_fade} fade)</span>
        <LastUpdatedChip value={data.computed_at} /></h1>
      <p className="subtitle">
        Each recent material headline read at the horizon that matters for its <b>type</b> (M&amp;A→day, analyst→week,
        product→1mo, earnings/guidance→3mo). The one durable edge in our data: <b>good news fades, hardest in mid/small caps.</b>{' '}
        <b>FADE</b> rows are robustness-validated (earnings-beat / product / strong bullish pop); <b>WATCH</b> is informational only.
        {data.computed_at && <span className="dim"> · scanned {new Date(data.computed_at).toLocaleString()}</span>}
      </p>

      <div className="studies-controls">
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Stance:</span>
          {['FADE','WATCH','all'].map(s => <button key={s} className={stance === s ? 'active' : ''} onClick={() => setStance(s)}>{s === 'all' ? 'All' : s}</button>)}
          <span className="filter-sep">|</span>
          <button className={robustOnly ? 'active' : ''} onClick={() => setRobustOnly(!robustOnly)}>validated only</button>
        </div>
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Type:</span>
          <select className="studies-search" style={{maxWidth:200}} value={catFilter} onChange={e => setCatFilter(e.target.value)}>
            <option value="all">All types</option>
            {data.cats.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <button className="refresh-btn" onClick={runScan} disabled={running}>{running ? 'Scanning...' : 'Re-scan'}</button>
        </div>
      </div>

      <table className="studies-table">
        <thead><tr>
          <SortTh colKey="ticker" sort={hSort}><Term k="ticker">Ticker</Term></SortTh><SortTh colKey="cat" sort={hSort}><Term k="newstype">News type</Term></SortTh><SortTh colKey="direction" sort={hSort}><Term k="dir">Dir</Term></SortTh><SortTh colKey="pop_pct" sort={hSort} align="right"><Term k="dayabn">Day-1 β-adj</Term></SortTh>
          <SortTh colKey="horizon" sort={hSort}><Term k="horizon">Horizon</Term></SortTh><SortTh colKey="days_left" sort={hSort} align="right"><Term k="window_ret">Window</Term></SortTh><SortTh colKey="stance" sort={hSort}><Term k="stance">Stance</Term></SortTh>
          <SortTh colKey="exp_drift" sort={hSort} align="right"><Term k="histdrift">Hist drift</Term></SortTh><SortTh colKey="market_cap" sort={hSort} align="right"><Term k="marketcap">Mkt cap</Term></SortTh>
          <SortTh colKey="last_close" sort={hSort} align="right"><Term k="lastclose">Last</Term></SortTh><th><Term k="sector">Sectors</Term></th>
        </tr></thead>
        <tbody>
          {rows.slice(0, 500).map(r => (
            <tr key={`${r.ticker}|${r.news_date}|${r.cat}`} className="study-row">
              <td><b>{r.ticker}</b></td>
              <td title={r.title} className="dim">{r.cat}<span className="dim"> ({r.impact === 3 ? 'major' : 'mod'})</span></td>
              <td>{dirTag(r.direction)}</td>
              <td style={{textAlign:'right'}} className={r.pop_pct == null ? 'dim' : r.pop_pct > 0 ? 'good' : 'bad'}>{r.pop_pct == null ? '–' : (r.pop_pct > 0 ? '+' : '') + r.pop_pct + '%'}</td>
              <td className="dim">{hz[r.horizon] || r.horizon}</td>
              <td style={{textAlign:'right'}} className="dim">{r.days_since}d in / <b>{r.days_left}d left</b></td>
              <td>{r.stance === 'FADE'
                ? <span className="sm-badge sm-13d" title={r.robust ? 'robustness-validated (time + size)' : ''}>FADE{r.robust ? ' ✓' : ''}</span>
                : <span className="dim">watch</span>}</td>
              <td style={{textAlign:'right'}} className={r.exp_drift < 0 ? 'bad' : r.exp_drift > 0 ? 'good' : 'dim'}>{r.exp_drift > 0 ? '+' : ''}{r.exp_drift}%</td>
              <td style={{textAlign:'right'}} className="dim">{capFmt(r.market_cap)} <span style={{fontSize:10}}>{r.cap_bucket}</span></td>
              <td style={{textAlign:'right'}} className="dim">{r.last_close}</td>
              <td className="dim">{(r.sectors || []).slice(0, 2).join(', ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > 500 && <p className="dim" style={{marginTop:8}}>Showing 500 of {rows.length}. Narrow with filters.</p>}
    </div>
  );
}

function NewsEffectPage() {
  const [data, setData] = useState(null);
  const [rows, setRows] = useState([]);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);
  const [effect, setEffect] = useState(true);
  const [hideJunk, setHideJunk] = useState(true);
  const [hideSuspect, setHideSuspect] = useState(true);
  const [hideOff, setHideOff] = useState(true);
  // "classified" = the RICH Anthropic impact-rating layer (llm_rating: dir/impact/horizon), ~13mo /
  // ~13k rows. Distinct from the corpus-wide LOCAL categorization (cat_llm/off_ticker). Requiring it
  // hides most of the categorized corpus, so the lagged/raw lenses turn it OFF.
  const [classifiedOnly, setClassifiedOnly] = useState(true);
  const [dir, setDir] = useState('all');
  const [minImpact, setMinImpact] = useState(0);
  const [cat, setCat] = useState('all');
  const [sort, setSort] = useState('date');
  const [acat, setAcat] = useState('all');
  const [ticker, setTicker] = useState('');
  const [chartTicker, setChartTicker] = useState(null);
  const LIMIT = 300;

  const qstr = (offset) => {
    const p = new URLSearchParams();
    p.set('limit', LIMIT); p.set('offset', offset);
    p.set('effect', effect ? 1 : 0);
    p.set('hide_junk', hideJunk ? 1 : 0);
    p.set('hide_suspect', hideSuspect ? 1 : 0);
    p.set('hide_offticker', hideOff ? 1 : 0);
    p.set('classified', classifiedOnly ? 1 : 0);
    if (dir !== 'all') p.set('dir', dir);
    if (minImpact) p.set('min_impact', minImpact);
    if (cat !== 'all') p.set('cat', cat);
    if (acat !== 'all') p.set('acat', acat);
    if (sort) p.set('sort', sort);
    if (ticker.trim()) p.set('ticker', ticker.trim());
    return p.toString();
  };

  const load = (offset = 0) => {
    setLoading(true);
    fetch(`${API}/news-effect?${qstr(offset)}`).then(r => r.json()).then(d => {
      setData(d);
      setRows(prev => offset === 0 ? d.results : prev.concat(d.results));
      setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  };
  // reload from the top whenever a filter changes
  useEffect(() => { load(0); /* eslint-disable-next-line */ }, [effect, hideJunk, hideSuspect, hideOff, classifiedOnly, dir, minImpact, cat, acat, sort]);

  // Named lenses over the same corpus (user: "expose the clean-catalyst filter … but if there is no
  // move we still want it and categorized because we'll study its effect long term — a move may lag").
  //  • catalyst  = genuine company-specific event that moved the stock (effect ∧ ¬off-ticker ∧ ¬junk).
  //  • lagged    = KEEP the no-move rows (effect OFF) so their forward-return columns expose drift that
  //                shows up days/weeks later; still drop off-ticker + junk (not about the company / not events).
  //  • raw       = the whole classified corpus, nothing hidden.
  const applyLens = (lens) => {
    if (lens === 'catalyst')   { setEffect(true);  setHideOff(true);  setHideJunk(true);  setHideSuspect(true); setClassifiedOnly(true); }
    else if (lens === 'lagged') { setEffect(false); setHideOff(true);  setHideJunk(true);  setHideSuspect(true); setClassifiedOnly(false); }
    else if (lens === 'raw')    { setEffect(false); setHideOff(false); setHideJunk(false); setHideSuspect(false); setClassifiedOnly(false); }
  };
  const activeLens =
    (effect && hideOff && hideJunk && hideSuspect && classifiedOnly) ? 'catalyst' :
    (!effect && hideOff && hideJunk && hideSuspect && !classifiedOnly) ? 'lagged' :
    (!effect && !hideOff && !hideJunk && !hideSuspect && !classifiedOnly) ? 'raw' : 'custom';

  if (err) return <div className="error">Error: {err}</div>;

  const abnCls = v => v == null ? 'dim' : v > 0 ? 'good' : v < 0 ? 'bad' : 'dim';
  const abnFmt = v => v == null ? '–' : (v > 0 ? '+' : '') + v.toFixed(1) + '%';
  const dirTag = d => d > 0 ? <span className="good">▲</span> : d < 0 ? <span className="bad">▼</span> : <span className="dim">–</span>;
  const hz = { day: 'day', week: 'week', month: '1mo', '3mo': '3mo' };
  // forward-return cell: a null is "pending" (news too recent for the window) vs "–" (no candle data).
  const daysSince = s => s ? (Date.now() - new Date(s).getTime()) / 86400000 : 1e9;
  const fwd = (v, dt, calDays) => v != null
    ? <span className={abnCls(v)}>{abnFmt(v)}</span>
    : daysSince(dt) < calDays
      ? <span className="dim" style={{fontSize:10, fontStyle:'italic'}} title="not enough time has passed yet">pending</span>
      : <span className="dim">–</span>;

  return (
    <div className="studies-page">
      <h1>News Effect {data && <span className="dim">({rows.length} shown / {data.n_total} match, {data.n_effect} moved)</span>}
        <LastUpdatedChip value={data && (data.last_updated || data.computed_at)} />
      </h1>
      <p className="subtitle">
        Real news events — <b>headlines that actually moved the stock</b>. <b>Day move</b> = β-adjusted abnormal return over the
        news's reaction session (prior close → close, so it includes the overnight / pre-market gap). <b>Moved</b> = ≥2σ
        abnormal that day. By default we hide <b>junk</b> (opinion / clickbait / award-PR — "nothing happened" headlines),
        <b>suspect</b> bad-candle artifacts, <b>off-ticker</b> headlines (the LLM judged them not about this ticker — macro
        or a different company in the feed), and non-events. Click a <b>ticker</b> to see all its news on the price chart.
      </p>
      <p className="subtitle" style={{marginTop:-6}}>
        <b>Lens</b> — <b>Clean catalyst</b>: a genuine company-specific headline that moved the stock ≥2σ that day (moved ∧ on-ticker,
        junk/suspect dropped). <b>Lagged / no-move</b>: keeps the headlines that <i>didn't</i> move the stock that day — nothing is
        thrown away, because a real reaction can lag as the market digests it, so the <b>1mo / 3mo / 1yr</b> columns are where you
        look for delayed drift. <b>Raw</b>: the whole classified corpus, nothing hidden.
      </p>

      <div className="studies-controls">
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Lens:</span>
          {[['catalyst','Clean catalyst'],['lagged','Lagged / no-move'],['raw','Raw']].map(([v,l]) =>
            <button key={v} className={activeLens === v ? 'active' : ''} onClick={() => applyLens(v)}
                    title={v === 'catalyst' ? 'Moved ≥2σ that day AND about this ticker (junk/suspect hidden) — the clean company-specific catalysts'
                         : v === 'lagged' ? 'Keep the no-move headlines too — study delayed drift via the 1mo/3mo/1yr columns (off-ticker + junk still hidden)'
                         : 'Everything classified, nothing hidden'}>{l}</button>)}
          {activeLens === 'custom' && <span className="dim" style={{fontSize:10, fontStyle:'italic'}}>custom</span>}
          <span className="filter-sep">|</span>
          <button className={effect ? 'active' : ''} onClick={() => setEffect(!effect)}>moved that day only</button>
          <button className={hideJunk ? 'active' : ''} onClick={() => setHideJunk(!hideJunk)}>hide junk</button>
          <button className={hideSuspect ? 'active' : ''} onClick={() => setHideSuspect(!hideSuspect)}>hide suspect</button>
          <button className={hideOff ? 'active' : ''} onClick={() => setHideOff(!hideOff)} title="hide headlines the LLM judged are not about this ticker (macro / a different company in the feed)">hide off-ticker</button>
          <button className={classifiedOnly ? 'active' : ''} onClick={() => setClassifiedOnly(!classifiedOnly)} title="only rows with the rich Anthropic impact rating (dir/impact/horizon, ~13mo). Turn OFF to see the whole locally-categorized corpus incl. rows still awaiting a rating.">rated only</button>
          <span className="filter-sep">|</span>
          <span className="dim" style={{fontSize:11}}>Dir:</span>
          {[['all','All'],['up','▲ up'],['down','▼ down']].map(([v,l]) => <button key={v} className={dir === v ? 'active' : ''} onClick={() => setDir(v)}>{l}</button>)}
          <span className="filter-sep">|</span>
          <span className="dim" style={{fontSize:11}}>Impact≥:</span>
          {[[0,'any'],[2,'material'],[3,'major']].map(([v,l]) => <button key={v} className={minImpact === v ? 'active' : ''} onClick={() => setMinImpact(v)}>{l}</button>)}
        </div>
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Category:</span>
          <select className="studies-search" style={{maxWidth:160}} value={acat} onChange={e => setAcat(e.target.value)}>
            <option value="all">All categories</option>
            {(data?.acats || []).map(c => <option key={c} value={c}>{CAT_LABELS[c] || c}</option>)}
          </select>
          <span className="dim" style={{fontSize:11}}>LLM type:</span>
          <select className="studies-search" style={{maxWidth:150}} value={cat} onChange={e => setCat(e.target.value)}>
            <option value="all">All</option>
            {(data?.cats || []).map(c => <option key={c} value={c}>{c}</option>)}
          </select>
          <span className="dim" style={{fontSize:11}}>Sort:</span>
          {[['date','newest'],['abn','biggest move'],['impact','impact']].map(([v,l]) => <button key={v} className={sort === v ? 'active' : ''} onClick={() => setSort(v)}>{l}</button>)}
          <input className="studies-search" style={{maxWidth:110}} placeholder="ticker…" value={ticker}
                 onChange={e => setTicker(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') load(0); }} />
          <button className="refresh-btn" onClick={() => load(0)} disabled={loading}>{loading ? '…' : 'Go'}</button>
        </div>
      </div>

      {!data ? <div className="loading">Loading news…</div> : (
      <table className="studies-table">
        <thead><tr>
          <th><Term k="datecol">Date</Term></th><th><Term k="ticker">Ticker</Term></th><th><Term k="category">Category</Term></th><th><Term k="newstype">News type</Term></th>
          <th style={{textAlign:'right'}}><Term k="dayabn">Day move</Term></th>
          <th><Term k="dayeffect">Moved?</Term></th><th><Term k="horizon">Horizon</Term></th>
          <th style={{textAlign:'right'}}><Term k="fwdret">1mo</Term></th>
          <th style={{textAlign:'right'}}><Term k="fwdret">3mo</Term></th><th style={{textAlign:'right'}}><Term k="fwdret">1yr</Term></th>
          <th style={{textAlign:'right'}}><Term k="sentiment">Sent.</Term></th><th><Term k="headline">Headline</Term></th>
        </tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.ticker}|${r.dt}|${i}`} className="study-row">
              <td className="dim">{r.dt ? r.dt.slice(0, 10) : '–'}</td>
              <td><button className="link-btn" onClick={() => setChartTicker(r.ticker)} title="Show all news on the price chart"><b>{r.ticker}</b></button>
                {r.off_ticker && <span className="sm-badge" style={{background:'#4a2e5a', marginLeft:4, fontSize:9}} title="LLM judged this headline is NOT about this ticker (macro / different company) — excluded by default">off</span>}</td>
              <td><span className="sm-badge" style={{background: r.cat_llm ? '#243a2e' : '#26324a'}}
                        title={r.cat_llm ? 'local-LLM classified' : 'title heuristic'}>
                {CAT_LABELS[r.cat_llm || r.cat_auto] || r.cat_llm || r.cat_auto || 'other'}
                {r.cat_llm && <span style={{fontSize:9, opacity:0.6}}> ✦</span>}</span></td>
              <td className="dim">{dirTag(r.llm_dir)} {r.llm_cat || 'other'}<span className="dim"> ({r.llm_impact === 3 ? 'major' : r.llm_impact === 2 ? 'mod' : 'minor'})</span>
                {r.grounded_label && (() => {
                  const sc = r.grounded_score;
                  const cls = sc == null || sc === 0 ? 'dim' : sc > 0 ? 'good' : 'bad';
                  return <span className={`sm-badge ${cls}`} style={{marginLeft:4, fontSize:9, display:'inline-block'}}
                    title="Grounded earnings verdict (EPS surprise + forward guidance)">{r.grounded_label.replace(/_/g, ' ')}</span>;
                })()}</td>
              <td style={{textAlign:'right'}} className={abnCls(r.day_abn)}><b>{abnFmt(r.day_abn)}</b></td>
              <td>{r.junk
                ? <span className="sm-badge" style={{background:'#5a4a1e'}} title="opinion / clickbait / PR — not a real event">junk</span>
                : r.day_suspect
                ? <span className="sm-badge" style={{background:'#7a2e2e'}} title="likely bad candle / illiquid OTC — excluded from analysis">suspect</span>
                : r.day_effect
                  ? <span className="sm-badge sm-13d" title="≥2σ abnormal move that day">moved</span>
                  : <span className="dim">–</span>}</td>
              <td className="dim">{hz[r.llm_horizon] || r.llm_horizon || '–'}</td>
              <td style={{textAlign:'right'}}>{fwd(r.ret_1m, r.dt, 31)}</td>
              <td style={{textAlign:'right'}}>{fwd(r.ret_3m, r.dt, 93)}</td>
              <td style={{textAlign:'right'}}>{fwd(r.ret_1y, r.dt, 366)}</td>
              <td style={{textAlign:'right'}} className={r.sentiment == null ? 'dim' : r.sentiment > 0 ? 'good' : r.sentiment < 0 ? 'bad' : 'dim'}>{r.sentiment == null ? '–' : r.sentiment.toFixed(2)}</td>
              <td title={r.title} style={{maxWidth:420, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap'}}>
                {r.url ? <a href={r.url} target="_blank" rel="noreferrer">{r.title}</a> : r.title}</td>
            </tr>
          ))}
        </tbody>
      </table>)}
      {data && rows.length < data.n_total &&
        <button className="refresh-btn" style={{marginTop:10}} onClick={() => load(rows.length)} disabled={loading}>
          {loading ? 'Loading…' : `Load more (${rows.length} / ${data.n_total})`}</button>}

      {chartTicker && <NewsChart ticker={chartTicker} onClose={() => setChartTicker(null)} />}
    </div>
  );
}

function NewsChart({ ticker, onClose, showAll = false }) {
  const canvasRef = useRef(null);
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [hov, setHov] = useState(null);   // {x, y, m} hovered marker
  const geomRef = useRef(null);           // {xAt, yP, close} for hit-testing

  useEffect(() => {
    setD(null); setErr(null);
    // showAll (from the Clusters view) plots EVERY headline, incl. ones that didn't move / junk / PR —
    // a propping cluster is mostly non-movers, so the default event-only filter would hide it.
    const extra = showAll ? '&effect=0&hide_junk=0&hide_suspect=0&classified=0' : '';
    fetch(`${API}/news-effect/chart?ticker=${encodeURIComponent(ticker)}${extra}`)
      .then(r => r.json()).then(j => { if (j.error) setErr(j.error); else setD(j); })
      .catch(e => setErr(e.message));
  }, [ticker, showAll]);

  const draw = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv || !d) return;
    const W = cv.parentElement.clientWidth, H = 380, dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);
    const pad = { l: 8, r: 8, t: 28, b: 24 };
    const n = d.close.length;
    if (!n) return;
    const cw = (W - pad.l - pad.r) / n;
    const xAt = i => pad.l + i * cw + cw / 2;
    const vals = d.close.filter(v => v != null);
    const mn = Math.min(...vals), mx = Math.max(...vals), rg = (mx - mn) || 1;
    const yP = v => v == null ? null : pad.t + (1 - (v - mn) / rg) * (H - pad.t - pad.b);
    geomRef.current = { xAt, yP, n };

    // price line
    ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 1.5; ctx.beginPath();
    let started = false;
    d.close.forEach((v, i) => { const y = yP(v); if (y == null) return; const x = xAt(i); if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); });
    ctx.stroke();

    // news markers — triangle at the price on the reaction bar, colored by direction
    (d.markers || []).forEach(m => {
      const x = xAt(m.idx), y = yP(d.close[m.idx]);
      if (y == null) return;
      const up = m.dir > 0;
      const sz = m.impact >= 3 ? 6 : m.impact === 2 ? 5 : 4;
      const col = m.junk ? '#b8a13a' : m.suspect ? '#d98d3a'
        : up ? '#3fb950' : m.dir < 0 ? '#f85149' : '#7d8590';
      ctx.fillStyle = col; ctx.globalAlpha = m.effect ? 0.95 : 0.4;
      const yy = up ? y - 8 : y + 8;
      ctx.beginPath();
      if (up) { ctx.moveTo(x, yy - sz); ctx.lineTo(x - sz, yy + sz); ctx.lineTo(x + sz, yy + sz); }
      else { ctx.moveTo(x, yy + sz); ctx.lineTo(x - sz, yy - sz); ctx.lineTo(x + sz, yy - sz); }
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
    });
    // legend
    ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillStyle = '#58a6ff'; ctx.fillText('Price', pad.l + 4, 15);
    ctx.fillStyle = '#3fb950'; ctx.fillText('▲ good news', pad.l + 48, 15);
    ctx.fillStyle = '#f85149'; ctx.fillText('▼ bad news', pad.l + 132, 15);
    ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace';
    ctx.textAlign = 'left'; ctx.fillText(d.dates[0], pad.l, H - 7);
    ctx.textAlign = 'right'; ctx.fillText(d.dates[n - 1], W - pad.r, H - 7);
  }, [d]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const onR = () => draw(); window.addEventListener('resize', onR);
    return () => window.removeEventListener('resize', onR);
  }, [draw]);

  const onMove = (e) => {
    const g = geomRef.current;
    if (!g || !d) return;
    const rect = e.target.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bestDist = 12;
    (d.markers || []).forEach(m => {
      const x = g.xAt(m.idx), y = g.yP(d.close[m.idx]);
      if (y == null) return;
      const dist = Math.hypot(mx - x, my - (m.dir > 0 ? y - 8 : y + 8));
      if (dist < bestDist) { bestDist = dist; best = { x, y, m }; }
    });
    setHov(best);
  };

  const abnFmt = v => v == null ? '–' : (v > 0 ? '+' : '') + v.toFixed(1) + '%';

  return (
    <div className="addiv-chart-overlay" onClick={onClose}>
      <div className="addiv-chart-box" onClick={e => e.stopPropagation()}>
        <div className="chart-header">
          <h3 style={{ margin: 0 }}>{ticker} — price &amp; news
            {d && <span className="dim" style={{ fontSize: 12 }}> · {d.n_markers} events</span>}</h3>
          <button className="close-btn" onClick={onClose}>Close</button>
        </div>
        {err && <div className="error">Error: {err}</div>}
        {!d && !err && <div className="loading">Loading chart…</div>}
        {d && <>
          <div style={{ width: '100%', position: 'relative' }}>
            <canvas ref={canvasRef} onMouseMove={onMove} onMouseLeave={() => setHov(null)} />
            {hov && <div className="news-tip" style={{ left: Math.min(hov.x + 12, 400), top: hov.y - 10 }}>
              <div className="dim" style={{ fontSize: 10 }}>{hov.m.dt ? hov.m.dt.slice(0, 10) : ''} · {hov.m.cat || 'other'} · <span className={hov.m.day_abn > 0 ? 'good' : 'bad'}>{abnFmt(hov.m.day_abn)}</span></div>
              <div style={{ fontSize: 12 }}>{hov.m.title}</div>
            </div>}
          </div>
          <p className="subtitle" style={{ margin: '10px 0 6px' }}>
            Blue = price. Each triangle is a real news event on its reaction bar — <span className="good">▲ green</span> moved the stock up,
            <span className="bad"> ▼ red</span> moved it down (bigger = higher impact). Hover a triangle for the headline. List below is newest-first.
          </p>
          <div className="news-marker-list">
            {[...(d.markers || [])].reverse().map((m, i) => (
              <div key={i} className="news-marker-row">
                <span className="dim" style={{ fontSize: 10, minWidth: 66 }}>{m.dt ? m.dt.slice(0, 10) : ''}</span>
                <span className={m.day_abn > 0 ? 'good' : 'bad'} style={{ minWidth: 52, textAlign: 'right', fontWeight: 700 }}>{abnFmt(m.day_abn)}</span>
                <span className="dim" style={{ minWidth: 90, fontSize: 11 }}>{m.cat || 'other'}</span>
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {m.url ? <a href={m.url} target="_blank" rel="noreferrer">{m.title}</a> : m.title}</span>
              </div>
            ))}
          </div>
        </>}
      </div>
    </div>
  );
}

function NewsClusterPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);
  const [minItems, setMinItems] = useState(5);
  const [gap, setGap] = useState(3);
  const [includeJunk, setIncludeJunk] = useState(true);
  const [fadedOnly, setFadedOnly] = useState(false);
  const [sort, setSort] = useState('prop');
  const [ticker, setTicker] = useState('');
  const [chartTicker, setChartTicker] = useState(null);

  const load = () => {
    setLoading(true);
    const p = new URLSearchParams();
    p.set('min_items', minItems); p.set('gap', gap); p.set('sort', sort);
    p.set('include_junk', includeJunk ? 1 : 0); p.set('limit', 400);
    if (ticker.trim()) p.set('ticker', ticker.trim());
    fetch(`${API}/news-clusters?${p.toString()}`).then(r => r.json()).then(d => {
      setData(d); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  };
  useEffect(() => { load(); /* eslint-disable-next-line */ }, [minItems, gap, includeJunk, sort]);

  if (err) return <div className="error">Error: {err}</div>;

  const pct = v => v == null ? '–' : (v > 0 ? '+' : '') + v.toFixed(1) + '%';
  const pcls = v => v == null ? 'dim' : v > 0 ? 'good' : v < 0 ? 'bad' : 'dim';
  const scoreCol = s => s >= 80 ? '#7a2e2e' : s >= 60 ? '#5a4a1e' : '#30363d';
  const rows = (data?.results || []).filter(c => !fadedOnly || c.faded);

  return (
    <div className="studies-page">
      <h1><Term k="cluster">News clusters</Term>{data && <span className="dim"> ({rows.length} shown / {data.n_total}, {data.n_faded} faded)</span>}
        <LastUpdatedChip value={data?.last_updated} /></h1>
      <p className="subtitle">
        Bursts of headlines on one ticker inside a short window — the footprint of a promotion / <b>“propping”</b> campaign.
        A high <Term k="propscore">prop score</Term> (lots of items, one-sided bullish, heavy PR/opinion “junk”, with an up push)
        followed by a <b>negative “results since”</b> is a pump-then-fade. Click a <b>ticker</b> to see every headline on the price chart.
      </p>

      <div className="studies-controls">
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Min items:</span>
          {[4,5,7,10].map(v => <button key={v} className={minItems === v ? 'active' : ''} onClick={() => setMinItems(v)}>{v}+</button>)}
          <span className="filter-sep">|</span>
          <span className="dim" style={{fontSize:11}}>Max gap:</span>
          {[2,3,5,7].map(v => <button key={v} className={gap === v ? 'active' : ''} onClick={() => setGap(v)}>{v}d</button>)}
          <span className="filter-sep">|</span>
          <button className={includeJunk ? 'active' : ''} onClick={() => setIncludeJunk(!includeJunk)}>include junk/PR</button>
          <button className={fadedOnly ? 'active' : ''} onClick={() => setFadedOnly(!fadedOnly)}>faded only</button>
        </div>
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Sort:</span>
          {[['prop','prop score'],['recent','newest'],['items','most items'],['pop','biggest push'],['fade','worst fade']].map(([v,l]) =>
            <button key={v} className={sort === v ? 'active' : ''} onClick={() => setSort(v)}>{l}</button>)}
          <input className="studies-search" style={{maxWidth:110}} placeholder="ticker…" value={ticker}
                 onChange={e => setTicker(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') load(); }} />
          <button className="refresh-btn" onClick={load} disabled={loading}>{loading ? '…' : 'Go'}</button>
        </div>
      </div>

      {!data ? <div className="loading">Loading clusters…</div> : (
      <table className="studies-table">
        <thead><tr>
          <th><Term k="ticker">Ticker</Term></th><th><Term k="window">Window</Term></th><th style={{textAlign:'right'}}><Term k="items">Items</Term></th>
          <th style={{textAlign:'right'}}><Term k="moved">Moved</Term></th><th style={{textAlign:'right'}}><Term k="bullbear">Bull/Bear</Term></th>
          <th style={{textAlign:'right'}}><Term k="junk">Junk</Term></th>
          <th style={{textAlign:'right'}}><Term k="netmove">Net move</Term></th>
          <th style={{textAlign:'right'}}><Term k="fwdret">1mo</Term></th>
          <th style={{textAlign:'right'}}><Term k="fwdret">3mo</Term></th><th style={{textAlign:'right'}}><Term k="fwdret">1yr</Term></th>
          <th style={{textAlign:'right'}}><Term k="propscore">Prop</Term></th><th><Term k="samplehead">Sample headlines</Term></th>
        </tr></thead>
        <tbody>
          {rows.map((c, i) => (
            <tr key={`${c.ticker}|${c.start}|${i}`} className="study-row">
              <td><button className="link-btn" onClick={() => setChartTicker(c.ticker)} title="Show all news on the price chart"><b>{c.ticker}</b></button></td>
              <td className="dim" style={{fontSize:11}}>{c.start} → {c.end} <span className="dim">({c.span_days}d)</span></td>
              <td style={{textAlign:'right'}}><b>{c.n_items}</b></td>
              <td style={{textAlign:'right'}} className={c.n_moved ? '' : 'dim'}>{c.n_moved}</td>
              <td style={{textAlign:'right'}} className="dim"><span className="good">{c.n_bull}</span>/<span className="bad">{c.n_bear}</span></td>
              <td style={{textAlign:'right'}} className="dim">{c.n_junk} <span style={{fontSize:10}}>({Math.round(c.junk_share*100)}%)</span></td>
              <td style={{textAlign:'right'}} className={pcls(c.net_abn)}><b>{pct(c.net_abn)}</b></td>
              <td style={{textAlign:'right'}} className={c.faded ? 'bad' : pcls(c.ret_1m)}>{c.faded ? <b>{pct(c.ret_1m)}</b> : pct(c.ret_1m)}</td>
              <td style={{textAlign:'right'}} className={pcls(c.ret_3m)}>{pct(c.ret_3m)}</td>
              <td style={{textAlign:'right'}} className={pcls(c.ret_1y)}>{pct(c.ret_1y)}</td>
              <td style={{textAlign:'right'}}><span className="sm-badge" style={{background:scoreCol(c.prop_score)}}>{c.prop_score}</span></td>
              <td className="dim" title={(c.headlines || []).join('\n')} style={{maxWidth:360, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', fontSize:11}}>
                {(c.headlines || [])[0] || ''}</td>
            </tr>
          ))}
        </tbody>
      </table>)}

      {chartTicker && <NewsChart ticker={chartTicker} onClose={() => setChartTicker(null)} showAll />}
    </div>
  );
}

const SM_TIP = {
  insider: 'Insider OPEN-MARKET BUYS — the company\'s own officers & directors buying shares with their own money (SEC Form 4). One of the strongest conviction signals: insiders sell for many reasons but buy for only one. ~trailing 6 months; SEC bulk data lags ~1 quarter.',
  d13: 'Schedule 13D — an ACTIVIST investor just disclosed a >5% stake AND an intent to influence the company (board seats, strategy, push for a sale). Must be filed within 10 days of crossing 5%. Often a catalyst. Count = filings in the last 180 days.',
  g13: 'Schedule 13G — a PASSIVE institution disclosed a >5% stake with NO activist intent (index funds, long-only managers). Signals large, price-insensitive accumulation rather than a catalyst. Count = filings in the last 180 days.',
};

function SmartMoney({ r, onOpen }) {
  const fmt = v => v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : `$${(v / 1e3).toFixed(0)}K`;
  const insVal = r.insider_buy_90d || r.insider_buy_180d || 0;
  const p = [];
  if (insVal > 0) p.push(<span key="i" className="sm-badge sm-insider" title={SM_TIP.insider}>▲ {fmt(insVal)} insider</span>);
  else if (r.insider_buy_180d > 0) p.push(<span key="i" className="sm-badge sm-insider" title={SM_TIP.insider}>▲ insider</span>);
  if (r.recent_13d > 0) p.push(<span key="d" className="sm-badge sm-13d" title={SM_TIP.d13}>13D×{r.recent_13d}</span>);
  if (r.recent_13g > 0) p.push(<span key="g" className="sm-badge sm-13g" title={SM_TIP.g13}>13G×{r.recent_13g}</span>);
  if (!p.length) return <span className="dim">–</span>;
  return <button className="sm-cell" onClick={() => onOpen && onOpen(r.ticker)}
                 title="Click for the individual filings & insider trades">{p}</button>;
}

function SmartMoneyPopup({ ticker, onClose }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    setD(null); setErr(null);
    fetch(`${API}/smart-money?ticker=${encodeURIComponent(ticker)}`)
      .then(r => r.json()).then(j => j.error ? setErr(j.error) : setD(j)).catch(e => setErr(e.message));
  }, [ticker]);
  const smSort = useSortedRows(d && d.insider, null, 'desc');
  const money = v => v == null ? '–' : v >= 1e9 ? `$${(v / 1e9).toFixed(2)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(2)}M` : `$${(v / 1e3).toFixed(0)}K`;
  const filingList = (arr, tip) => arr.length
    ? <ul className="sm-pop-list">{arr.map((f, i) => <li key={i}><span className="dim">{f.filed_date}</span> <span className="mono">{f.accession}</span></li>)}</ul>
    : <div className="dim" style={{fontSize:12}}>none on record</div>;
  return (
    <div className="addiv-chart-overlay" onClick={onClose}>
      <div className="addiv-chart-box" onClick={e => e.stopPropagation()} style={{ maxWidth: 660 }}>
        <div className="chart-header">
          <h3 style={{ margin: 0 }}>{ticker} — smart-money filings</h3>
          <button className="close-btn" onClick={onClose}>Close</button>
        </div>
        {err && <div className="error">Error: {err}</div>}
        {!d && !err && <div className="loading">Loading filings…</div>}
        {d && <>
          <p className="subtitle" style={{ margin: '6px 0 12px' }}>
            Every SEC ownership filing (13D activist / 13G passive 5%+ stakes) and insider open-market trade behind the badge.
            {' '}<a href={d.edgar_url} target="_blank" rel="noreferrer">View on SEC EDGAR ↗</a>
          </p>
          <div className="sm-pop-grid">
            <div>
              <div className="sm-pop-h" title={SM_TIP.d13}>13D — activist <span className="dim">({d.totals.n_13d})</span></div>
              {filingList(d.filings_13d)}
            </div>
            <div>
              <div className="sm-pop-h" title={SM_TIP.g13}>13G — passive institutional <span className="dim">({d.totals.n_13g})</span></div>
              {filingList(d.filings_13g)}
            </div>
          </div>
          <div style={{ marginTop: 14 }}>
            <div className="sm-pop-h" title={SM_TIP.insider}>
              Insider open-market trades <span className="dim">— buys {money(d.totals.insider_buy)} ({d.totals.insider_buy_count} tx) · sells {money(d.totals.insider_sell)}</span>
            </div>
            {d.insider.length
              ? <table className="studies-table" style={{ marginTop: 6 }}>
                  <thead><tr><SortTh colKey="filed_date" sort={smSort}><Term k="filed">Filed</Term></SortTh><SortTh colKey="buy_value" sort={smSort} align="right"><Term k="buys">Buys</Term></SortTh><SortTh colKey="buy_count" sort={smSort} align="right"><Term k="ntx">#tx</Term></SortTh><SortTh colKey="sell_value" sort={smSort} align="right"><Term k="sells">Sells</Term></SortTh></tr></thead>
                  <tbody>{smSort.rows.map((it, i) => (
                    <tr key={i}>
                      <td className="dim">{it.filed_date}</td>
                      <td style={{ textAlign: 'right' }} className={it.buy_value ? 'good' : 'dim'}>{it.buy_value ? money(it.buy_value) : '–'}</td>
                      <td style={{ textAlign: 'right' }} className="dim">{it.buy_count || ''}</td>
                      <td style={{ textAlign: 'right' }} className={it.sell_value ? 'bad' : 'dim'}>{it.sell_value ? money(it.sell_value) : '–'}</td>
                    </tr>))}
                  </tbody>
                </table>
              : <div className="dim" style={{fontSize:12}}>none on record</div>}
          </div>
        </>}
      </div>
    </div>
  );
}

function AdDivergenceChart({ ticker, onClose }) {
  const canvasRef = useRef(null);
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setD(null); setErr(null);
    fetch(`${API}/ad-divergence/chart?ticker=${encodeURIComponent(ticker)}`)
      .then(r => r.json()).then(j => { if (j.error) setErr(j.error); else setD(j); })
      .catch(e => setErr(e.message));
  }, [ticker]);

  const draw = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv || !d) return;
    const W = cv.parentElement.clientWidth;
    const H = 360;
    const dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);
    const pad = { l: 8, r: 8, t: 30, b: 24 };
    const n = d.close.length;
    if (!n) return;
    const cw = (W - pad.l - pad.r) / n;
    const xAt = i => pad.l + i * cw + cw / 2;
    const norm = (arr) => {
      const vals = arr.filter(v => v != null);
      const mn = Math.min(...vals), mx = Math.max(...vals), rg = (mx - mn) || 1;
      return v => v == null ? null : pad.t + (1 - (v - mn) / rg) * (H - pad.t - pad.b);
    };
    const yP = norm(d.close), yA = norm(d.adl);

    // shade every bar by its A/D state: 2 accum-divergence (green), 1 accum/trend-up (blue),
    // 0 neutral (unshaded), -1 distribution (red)
    (d.state || []).forEach((s, i) => {
      const col = s === 2 ? 'rgba(63,185,80,0.22)' : s === 1 ? 'rgba(88,166,255,0.13)' : s === -1 ? 'rgba(248,81,73,0.15)' : null;
      if (col) { ctx.fillStyle = col; ctx.fillRect(pad.l + i * cw, pad.t, cw + 0.6, H - pad.t - pad.b); }
    });
    // capitulation fire markers
    (d.fires || []).forEach(f => {
      const x = xAt(f.idx);
      ctx.strokeStyle = 'rgba(248,81,73,0.55)'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#f85149';
      ctx.beginPath(); ctx.moveTo(x, H - pad.b); ctx.lineTo(x - 4, H - pad.b + 7); ctx.lineTo(x + 4, H - pad.b + 7); ctx.closePath(); ctx.fill();
    });
    const line = (arr, yFn, color, dash) => {
      ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.setLineDash(dash || []); ctx.beginPath();
      let started = false;
      arr.forEach((v, i) => { const y = yFn(v); if (y == null) return; const x = xAt(i); if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]);
    };
    line(d.adl_sma, yA, 'rgba(63,185,80,0.35)', [4, 3]);
    line(d.adl, yA, '#3fb950');
    line(d.close, yP, '#58a6ff');
    // legend
    ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillStyle = '#58a6ff'; ctx.fillText('Price', pad.l + 4, 16);
    ctx.fillStyle = '#3fb950'; ctx.fillText('A/D line', pad.l + 52, 16);
    ctx.fillStyle = '#f85149'; ctx.fillText('▲ capitulation fire', pad.l + 118, 16);
    // date axis
    ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace';
    ctx.textAlign = 'left'; ctx.fillText(d.dates[0], pad.l, H - 7);
    ctx.textAlign = 'right'; ctx.fillText(d.dates[n - 1], W - pad.r, H - 7);
  }, [d]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const onR = () => draw(); window.addEventListener('resize', onR);
    return () => window.removeEventListener('resize', onR);
  }, [draw]);

  const st = d && d.state ? d.state[d.state.length - 1] : null;
  const stateLabel = st === 2 ? 'accum divergence' : st === 1 ? 'accum trend-up' : st === -1 ? 'distribution' : st === 0 ? 'neutral' : '–';

  return (
    <div className="addiv-chart-overlay" onClick={onClose}>
      <div className="addiv-chart-box" onClick={e => e.stopPropagation()}>
        <div className="chart-header">
          <h3 style={{ margin: 0 }}>{ticker} — price vs A/D line <span className="dim" style={{ fontSize: 12 }}>· now: {stateLabel}</span></h3>
          <button className="close-btn" onClick={onClose}>Close</button>
        </div>
        {err && <div className="error">Error: {err}</div>}
        {!d && !err && <div className="loading">Loading chart...</div>}
        {d && <>
          <div style={{ width: '100%' }}><canvas ref={canvasRef} /></div>
          <p className="subtitle" style={{ margin: '10px 0 6px' }}>
            Each line is scaled to its own range — what matters is the <b>shape</b>: blue = price, green = the A/D (Accumulation/Distribution) line, dashed = its 10-day average. Red ▲ = a capitulation signal fired (new 52-wk low / RSI&lt;20). Every bar is shaded by its money-flow state:
          </p>
          <div className="ad-legend">
            <div className="ad-leg-row"><span className="ad-sw" style={{ background: 'rgba(63,185,80,0.7)' }} /><b>Accumulation divergence</b> — price flat or <i>falling</i> while the A/D line <i>rises</i>. Smart money buying into weakness — the strongest bullish tell. <b>This is the Mode-A entry setup.</b></div>
            <div className="ad-leg-row"><span className="ad-sw" style={{ background: 'rgba(88,166,255,0.7)' }} /><b>Accumulation / trend-up</b> — A/D line rising <i>together with</i> price. Healthy buying pressure confirming an existing uptrend (not a divergence — no hidden signal, just participation).</div>
            <div className="ad-leg-row"><span className="ad-sw" style={{ background: 'rgba(125,133,144,0.35)', border: '1px solid #30363d' }} /><b>Neutral</b> (unshaded) — no clear money-flow bias; A/D drifting sideways with price.</div>
            <div className="ad-leg-row"><span className="ad-sw" style={{ background: 'rgba(248,81,73,0.7)' }} /><b>Distribution</b> — A/D line <i>falling</i>. Smart money selling or stepping away. <b>This is the Mode-A exit trigger</b> — you sell when accumulation flips to distribution.</div>
          </div>
        </>}
      </div>
    </div>
  );
}

function StrategyForwardPanel() {
  const ref = useRef(null);
  const [d, setD] = useState(null);
  const [mode, setMode] = useState('A_gated');
  const [running, setRunning] = useState(false);

  const load = () => fetch(`${API}/strategy-forward`).then(r => r.json()).then(setD).catch(() => {});
  useEffect(() => { load(); }, []);
  const run = () => {
    setRunning(true);
    fetch(`${API}/strategy-forward`, { method: 'POST' }).then(() => {
      const t = setInterval(() => fetch(`${API}/strategy-forward`).then(r => r.json()).then(j => {
        if (j.computed && j.computed_at !== (d && d.computed_at)) { clearInterval(t); setRunning(false); setD(j); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  const draw = useCallback(() => {
    const cv = ref.current;
    if (!cv || !d || !d.computed || !d.modes || !d.modes[mode]) return;
    const curve = d.modes[mode].curve;
    const W = cv.parentElement.clientWidth, H = 220, dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);
    const pad = { l: 38, r: 38, t: 18, b: 20 };
    const med = curve.map(c => c.median), win = curve.map(c => c.win);
    const mmin = Math.min(0, ...med), mmax = Math.max(...med), mr = (mmax - mmin) || 1;
    const xAt = i => pad.l + i / (curve.length - 1) * (W - pad.l - pad.r);
    const yMed = v => pad.t + (1 - (v - mmin) / mr) * (H - pad.t - pad.b);
    const yWin = v => pad.t + (1 - (v - 40) / 40) * (H - pad.t - pad.b);
    ctx.strokeStyle = '#30363d'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(pad.l, yMed(0)); ctx.lineTo(W - pad.r, yMed(0)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#7d8590'; ctx.font = '9px monospace'; ctx.textAlign = 'right'; ctx.fillText('0%', pad.l - 4, yMed(0) + 3);
    ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 1; ctx.beginPath(); win.forEach((v, i) => { const x = xAt(i), y = yWin(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 2; ctx.beginPath(); med.forEach((v, i) => { const x = xAt(i), y = yMed(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillStyle = '#3fb950'; ctx.fillText('median return %', pad.l + 2, 12);
    ctx.fillStyle = '#58a6ff'; ctx.fillText('win % (40–80)', pad.l + 118, 12);
    ctx.fillStyle = '#7d8590'; ctx.font = '10px monospace'; ctx.textAlign = 'left'; ctx.fillText('day 1', pad.l, H - 6);
    ctx.textAlign = 'right'; ctx.fillText(`day ${curve.length}`, W - pad.r, H - 6);
  }, [d, mode]);
  useEffect(() => { draw(); }, [draw]);
  useEffect(() => { const o = () => draw(); window.addEventListener('resize', o); return () => window.removeEventListener('resize', o); }, [draw]);

  if (!d) return null;
  const m = d.computed && d.modes && d.modes[mode];
  return (
    <div className="fwd-panel">
      <div className="fwd-head">
        <h3 style={{ margin: 0 }}>Where the trade sits after you buy it <span className="dim" style={{ fontSize: 12 }}>· average forward path</span></h3>
        <div className="filters">
          <button className={mode === 'A_gated' ? 'active' : ''} onClick={() => setMode('A_gated')}>Mode A (distress+div)</button>
          <button className={mode === 'either_gated' ? 'active' : ''} onClick={() => setMode('either_gated')}>Unified (either)</button>
          <button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Computing…' : 'Recompute'}</button>
        </div>
      </div>
      {!m ? <p className="dim" style={{ padding: '10px 0' }}>{d.message || 'Not computed yet.'} <button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Computing…' : 'Compute'}</button></p> : <>
        <div className="fwd-cards">
          <div className="fwd-card"><div className="fwd-num good">{m.day_final.median > 0 ? '+' : ''}{m.day_final.median}%</div><div className="fwd-lbl">median @ day {m.day_final.day}</div></div>
          <div className="fwd-card"><div className="fwd-num">{m.day_final.avg > 0 ? '+' : ''}{m.day_final.avg}%</div><div className="fwd-lbl">average @ day {m.day_final.day}</div></div>
          <div className="fwd-card"><div className="fwd-num">{m.day_final.win}%</div><div className="fwd-lbl">in profit @ day {m.day_final.day}</div></div>
          <div className="fwd-card"><div className="fwd-num dim">+{m.peak.avg}% · d{m.peak.day}</div><div className="fwd-lbl">peak avg (tail-driven)</div></div>
          <div className="fwd-card"><div className="fwd-num dim">{m.n}</div><div className="fwd-lbl">trades</div></div>
        </div>
        <div style={{ width: '100%' }}><canvas ref={ref} /></div>
        <p className="subtitle" style={{ marginTop: 8 }}>Median (green) is the honest central path; average (headline) is inflated by a few big winners. Win% (blue) is scaled 40–80%. {d.computed_at && <span className="dim">Computed {new Date(d.computed_at).toLocaleString()}.</span>}</p>
      </>}
    </div>
  );
}

function AdDivergencePage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);
  const [chartTicker, setChartTicker] = useState(null);
  const [smTicker, setSmTicker] = useState(null);
  const [hideKnives, setHideKnives] = useState(true);
  const [hideLowQual, setHideLowQual] = useState(true);
  const primedSort = useSortedRows(data && data.results, null, 'desc');
  const watchSort = useSortedRows(data && data.results, null, 'desc');

  const load = () => fetch(`${API}/ad-divergence?limit=1000`).then(r => r.json()).then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);

  const runScan = () => {
    setRunning(true);
    fetch(`${API}/ad-divergence`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/ad-divergence?limit=1`).then(r => r.json()).then(d => {
        if (d.computed && d.computed_at !== (data && data.computed_at)) { clearInterval(t); setRunning(false); load(); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="error">Error: {err}</div>;
  if (!data) return <div className="loading">Loading A/D divergence...</div>;

  const intro = (
    <>
      <h1>A/D Divergence <span className="dim">{data.computed ? `(${data.n_primed} primed · ${data.n_watch} watch)` : ''}</span>
        <LastUpdatedChip value={data.computed_at} /></h1>
      <p className="subtitle">
        Stocks whose Accumulation/Distribution line is in <b>accum divergence</b> right now — price flat/down while the ADL keeps rising (read as slope + divergence vs price, never sign). The study found this slice roughly <b>triples the edge</b> on price-capitulation signals, so a stock is <b>primed</b> when it's <i>also</i> firing a capitulation signal (new 52-wk low / RSI&lt;20).
        {data.computed_at && <span className="dim"> · scanned {new Date(data.computed_at).toLocaleString()}</span>}
        <button className="refresh-btn" style={{marginLeft:10}} onClick={runScan} disabled={running}>{running ? 'Scanning...' : 'Re-scan'}</button>
      </p>
    </>
  );

  if (!data.computed) {
    return (
      <div className="studies-page">
        {intro}
        <div className="empty-state" style={{padding:'40px 0'}}>
          <p>{data.message || 'Not scanned yet.'}</p>
          <button className="refresh-btn" onClick={runScan} disabled={running}>{running ? 'Scanning...' : 'Run A/D-divergence scan'}</button>
        </div>
      </div>
    );
  }

  const drop = r => (hideKnives && r.knife) || (hideLowQual && r.low_quality);
  const primedAll = data.results.filter(r => r.primed);
  const nKnives = primedAll.filter(r => r.knife).length;
  const nLowQual = primedAll.filter(r => r.low_quality && !r.knife).length;
  const primed = primedSort.rows.filter(r => r.primed && !drop(r));
  const watch = watchSort.rows.filter(r => !r.primed && !drop(r));
  const bucket = r => (r.fund_buckets || {})['Market cap'] || null;

  return (
    <div className="studies-page">
      {intro}

      <StrategyForwardPanel />

      <h2 style={{marginTop:18}}>🔥 Primed <span className="dim">— accum divergence + capitulation signal firing ({primed.length}{nKnives && hideKnives ? ` · ${nKnives} knives` : ''}{nLowQual && hideLowQual ? ` · ${nLowQual} low-qual` : ''}{(nKnives && hideKnives) || (nLowQual && hideLowQual) ? ' hidden' : ''})</span></h2>
      <p className="subtitle">
        The highest-conviction read: price just capitulated but money-flow is diverging up. <b>Fires 60d</b> = how many times the trigger fired in the last 60 bars, and <b>% off low</b> = how far price has lifted off its 60-bar low — a stock firing over and over while still pinned at its low is a <b>falling knife</b> ⚠, not a bounce. <b>Low-quality</b> ⚠ = micro-cap / penny / unprofitable, where the −33%+ blowups concentrate. Both are hidden by default.
        <label style={{marginLeft:12, fontSize:12, cursor:'pointer'}}>
          <input type="checkbox" checked={hideKnives} onChange={e => setHideKnives(e.target.checked)} /> Hide knives
        </label>
        <label style={{marginLeft:10, fontSize:12, cursor:'pointer'}}>
          <input type="checkbox" checked={hideLowQual} onChange={e => setHideLowQual(e.target.checked)} /> Hide low-quality
        </label>
      </p>
      <table className="studies-table">
        <thead><tr>
          <SortTh colKey="ticker" sort={primedSort}><Term k="ticker">Ticker</Term></SortTh><th><Term k="trigger">Capitulation trigger</Term></th><SortTh colKey="min_days_ago" sort={primedSort} align="right"><Term k="fired">Fired</Term></SortTh>
          <SortTh colKey="fires_60d" sort={primedSort} align="right"><Term k="fires60d">Fires 60d</Term></SortTh><SortTh colKey="pct_above_low" sort={primedSort} align="right"><Term k="pctofflow">% off low</Term></SortTh>
          <SortTh colKey="hist_avg_return" sort={primedSort} align="right"><Term k="histedge">Hist Ret</Term></SortTh><SortTh colKey="hist_win_rate" sort={primedSort} align="right"><Term k="winrate">Hist Win%</Term></SortTh><SortTh colKey="hist_trades" sort={primedSort} align="right"><Term k="ntrades">Trades</Term></SortTh>
          <SortTh colKey="best_exit_key" sort={primedSort}><Term k="bestexit">Best exit</Term></SortTh><SortTh colKey="last_close" sort={primedSort} align="right"><Term k="lastclose">Last</Term></SortTh><th><Term k="marketcap">Mkt Cap</Term></th><SortTh colKey="pe_ratio" sort={primedSort}><Term k="pe">PE</Term></SortTh><th><Term k="smartmoney">Smart money</Term></th><th><Term k="sector">Sectors</Term></th>
        </tr></thead>
        <tbody>
          {primed.map(r => (
            <tr key={r.ticker} className={`study-row${r.knife || r.low_quality ? ' knife-row' : ''}`}>
              <td>
                <button className="ticker-link" onClick={() => setChartTicker(r.ticker)} title="Show price vs A/D-line divergence">{r.ticker}</button>
                {r.knife && <span className="knife-badge" title="Falling knife: fired 4+ times in 60 bars and still within 5% of its low — repeated new-lows, not a bounce">⚠ knife</span>}
                {r.low_quality && <span className="lowqual-badge" title="Low-quality: micro-cap (<$300M) / penny (<$5) / unprofitable — where the −33%+ blowups concentrate">⚠ low-qual</span>}
              </td>
              <td className="dim">{(r.firing || []).map(f => f.signal_name).join(', ')}</td>
              <td style={{textAlign:'right'}}>{r.min_days_ago === 0 ? <span className="good">today</span> : `${r.min_days_ago}d`}</td>
              <td style={{textAlign:'right'}} className={r.fires_60d >= 4 ? 'bad' : r.fires_60d <= 2 ? 'good' : ''}>{r.fires_60d}</td>
              <td style={{textAlign:'right'}} className={r.pct_above_low != null && r.pct_above_low < 5 ? 'bad' : ''}>{r.pct_above_low != null ? `+${r.pct_above_low}%` : '–'}</td>
              <td style={{textAlign:'right'}} className={r.hist_avg_return > 0 ? 'good' : 'bad'}>{r.hist_avg_return != null ? `${r.hist_avg_return > 0 ? '+' : ''}${r.hist_avg_return.toFixed(1)}%` : '–'}</td>
              <td style={{textAlign:'right'}}>{r.hist_win_rate != null ? `${r.hist_win_rate.toFixed(0)}%` : '–'}</td>
              <td style={{textAlign:'right'}} className="dim">{r.hist_trades != null ? r.hist_trades : '–'}</td>
              <td className="dim" title={r.best_exit_key}>{r.best_exit_key || '–'}</td>
              <td style={{textAlign:'right'}} className="dim">{r.last_close}</td>
              <td>{bucket(r) || <span className="dim">–</span>}</td>
              <td className="dim">{r.pe_ratio != null ? r.pe_ratio.toFixed(1) : '–'}</td>
              <td style={{fontSize:11}}><SmartMoney r={r} onOpen={setSmTicker} /></td>
              <td className="dim">{(r.sectors || []).slice(0, 2).join(', ')}</td>
            </tr>
          ))}
          {!primed.length && <tr><td colSpan={14} className="dim" style={{padding:'14px 0'}}>{primedAll.length ? 'All primed stocks are falling knives right now — untick "Hide falling knives" to see them.' : 'No stocks are both in accum divergence AND firing a capitulation signal right now.'}</td></tr>}
        </tbody>
      </table>

      <h2 style={{marginTop:26}}>👀 Watch <span className="dim">— accum divergence, no capitulation trigger ({watch.length})</span></h2>
      <p className="subtitle">The weaker version of the read: money-flow is diverging up, but no price-capitulation signal is firing yet. Worth watching for a trigger.</p>
      <table className="studies-table">
        <thead><tr>
          <SortTh colKey="ticker" sort={watchSort}><Term k="ticker">Ticker</Term></SortTh><SortTh colKey="last_close" sort={watchSort} align="right"><Term k="lastclose">Last</Term></SortTh><th><Term k="marketcap">Mkt Cap</Term></th><SortTh colKey="pe_ratio" sort={watchSort}><Term k="pe">PE</Term></SortTh><th><Term k="smartmoney">Smart money</Term></th><th><Term k="sector">Sectors</Term></th>
        </tr></thead>
        <tbody>
          {watch.map(r => (
            <tr key={r.ticker} className={`study-row${r.low_quality ? ' knife-row' : ''}`}>
              <td>
                <button className="ticker-link" onClick={() => setChartTicker(r.ticker)} title="Show price vs A/D-line divergence">{r.ticker}</button>
                {r.low_quality && <span className="lowqual-badge" title="Low-quality: micro-cap / penny / unprofitable">⚠ low-qual</span>}
              </td>
              <td style={{textAlign:'right'}} className="dim">{r.last_close}</td>
              <td>{bucket(r) || <span className="dim">–</span>}</td>
              <td className="dim">{r.pe_ratio != null ? r.pe_ratio.toFixed(1) : '–'}</td>
              <td style={{fontSize:11}}><SmartMoney r={r} onOpen={setSmTicker} /></td>
              <td className="dim">{(r.sectors || []).slice(0, 2).join(', ')}</td>
            </tr>
          ))}
          {!watch.length && <tr><td colSpan={6} className="dim" style={{padding:'14px 0'}}>None.</td></tr>}
        </tbody>
      </table>

      {chartTicker && <AdDivergenceChart ticker={chartTicker} onClose={() => setChartTicker(null)} />}
      {smTicker && <SmartMoneyPopup ticker={smTicker} onClose={() => setSmTicker(null)} />}
    </div>
  );
}

function IntersectionPage() {
  const [meta, setMeta] = useState(null);        // {signals:[{key,name}], dims:[...]}
  const [signal, setSignal] = useState('obv_div_sort_pos');
  const [exitKey, setExitKey] = useState('6m');
  const [dims, setDims] = useState(['Market cap', 'PB', 'Insider buying']);
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const pollRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/stock-studies?limit=1000`).then(r => r.json()).then(d => {
      const sig = {};
      (d.results || []).forEach(r => { sig[r.signal_key] = r.signal_name; });
      const pitDims = Object.entries(d.dimension_meta || {}).filter(([, v]) => v.pit).map(([k]) => k);
      setMeta({ signals: Object.entries(sig).map(([key, name]) => ({ key, name })).sort((a, b) => a.name.localeCompare(b.name)), dims: pitDims });
    }).catch(() => setMeta({ signals: [], dims: [] }));
    fetch(`${API}/dim-intersection`).then(r => r.json()).then(d => { if (d.computed) setResult(d); }).catch(() => {});
    return () => clearInterval(pollRef.current);
  }, []);

  const toggleDim = (d) => setDims(ds => ds.includes(d) ? ds.filter(x => x !== d) : (ds.length < 3 ? [...ds, d] : ds));

  const run = () => {
    if (!dims.length) return;
    setRunning(true);
    fetch(`${API}/dim-intersection`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ signal, exit: exitKey, dims }) })
      .then(() => {
        clearInterval(pollRef.current);
        let n = 0;
        pollRef.current = setInterval(() => {
          n++;
          fetch(`${API}/dim-intersection`).then(r => r.json()).then(d => {
            const match = d.computed && d.signal === signal && d.exit === exitKey && JSON.stringify(d.dims) === JSON.stringify(dims);
            if (match) { clearInterval(pollRef.current); setResult(d); setRunning(false); }
            else if (n > 40) { clearInterval(pollRef.current); setRunning(false); }
          }).catch(() => {});
        }, 3000);
      }).catch(() => setRunning(false));
  };

  const iSort = useSortedRows(result && result.rows, null, 'desc', {
    combo0: r => (r.combo || [])[0],
    combo1: r => (r.combo || [])[1],
    combo2: r => (r.combo || [])[2],
  });
  if (!meta) return <div className="loading">Loading…</div>;
  return (
    <div className="studies-page">
      <h1>Dimension Intersection <span className="dim">— do amplifiers stack?</span>
        <LastUpdatedChip value={result?.computed_at || result?.last_updated} /></h1>
      <p className="subtitle">Pick a signal + up to 3 point-in-time dimensions; see which <b>combinations</b> beat the signal's baseline (vs. the one-at-a-time slices in Stock Indicator Studies). Runs on demand (~1 min).</p>
      <div className="studies-controls">
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Signal:</span>
          <select value={signal} onChange={e => setSignal(e.target.value)} className="studies-search" style={{maxWidth:340}}>
            {meta.signals.map(s => <option key={s.key} value={s.key}>{s.name}</option>)}
          </select>
          <span className="dim" style={{fontSize:11}}>Exit:</span>
          <input value={exitKey} onChange={e => setExitKey(e.target.value)} style={{width:70}} />
          <button className="refresh-btn" onClick={run} disabled={running || !dims.length}>{running ? 'Running (~1 min)…' : 'Run intersection'}</button>
        </div>
        <div className="filters">
          <span className="dim" style={{fontSize:11}}>Dimensions (2-3):</span>
          {meta.dims.map(d => (
            <button key={d} className={dims.includes(d) ? 'active' : ''} onClick={() => toggleDim(d)}>{d}</button>
          ))}
        </div>
      </div>
      {result && (
        <>
          <div className="regime-bar" style={{marginBottom:10}}>
            <span>Signal: <b>{result.signal}</b> → {result.exit}</span>
            <span>Baseline: <b className={result.baseline > 0 ? 'good' : 'bad'}>{result.baseline > 0 ? '+' : ''}{result.baseline}%</b></span>
            <span>Win: <b>{result.baseline_wr}%</b></span>
            <span>Trades: <b>{result.baseline_trades}</b></span>
            <span className="dim">{(result.dims || []).join(' ∩ ')}</span>
          </div>
          <table className="studies-table">
            <thead><tr>{(result.dims || []).map((d, di) => <SortTh key={d} colKey={`combo${di}`} sort={iSort}><Term k="dimcol">{d}</Term></SortTh>)}<SortTh colKey="avg" sort={iSort} align="right"><Term k="avgreturn">Avg</Term></SortTh><SortTh colKey="lift" sort={iSort} align="right"><Term k="lift">Lift</Term></SortTh><SortTh colKey="wr" sort={iSort} align="right"><Term k="winrate">Win%</Term></SortTh><SortTh colKey="trades" sort={iSort} align="right"><Term k="ntrades">Trades</Term></SortTh></tr></thead>
            <tbody>
              {iSort.rows.map((r, i) => (
                <tr key={i} className={r.lift > 0 ? 'row-bullish' : 'row-bearish'}>
                  {r.combo.map((c, j) => <td key={j} style={{fontSize:11}}>{c}</td>)}
                  <td style={{textAlign:'right'}} className={r.avg > 0 ? 'good' : 'bad'}><b>{r.avg > 0 ? '+' : ''}{r.avg}%</b></td>
                  <td style={{textAlign:'right'}} className={r.lift > 0 ? 'good' : 'bad'}>{r.lift > 0 ? '+' : ''}{r.lift}%</td>
                  <td style={{textAlign:'right'}}>{r.wr}%</td>
                  <td style={{textAlign:'right'}} className="dim">{r.trades}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

function PlaybookPage() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);
  const load = () => fetch(`${API}/playbook`).then(r => r.json()).then(setD).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);
  const run = () => {
    setRunning(true);
    fetch(`${API}/playbook`, { method: 'POST' }).then(() => {
      const t = setInterval(() => fetch(`${API}/playbook`).then(r => r.json()).then(j => {
        if (j.computed && j.computed_at !== (d && d.computed_at)) { clearInterval(t); setRunning(false); setD(j); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };
  const pbSort = useSortedRows(d && d.candidates, null, 'desc', {
    risk: c => (c.risk ? c.risk.score : null),
    histedge: c => c.hist_avg_return,
  });
  if (err) return <div className="error">Error: {err}</div>;
  if (!d) return <div className="loading">Loading playbook...</div>;

  const money = v => v == null ? '–' : v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : `$${(v / 1e6).toFixed(0)}M`;
  const smart = c => {
    const p = [];
    if (c.insider_buy_180d > 0) p.push(<span key="i" className="sm-badge sm-insider" title="Insider open-market buys ~6mo">▲ insider</span>);
    if (c.recent_13d > 0) p.push(<span key="d" className="sm-badge sm-13d">13D×{c.recent_13d}</span>);
    if (c.recent_13g > 0) p.push(<span key="g" className="sm-badge sm-13g">13G×{c.recent_13g}</span>);
    return p.length ? p : null;
  };
  const stateCls = s => (s === 'IN' || s === 'STRONG' || s === 'LEADER') ? 'pb-in' : s === 'TURNING' ? 'pb-turn' : 'pb-out';
  const cands = d.candidates || [];

  const Step = ({ n, title, sub, children }) => (
    <div className="pb-step">
      <div className="pb-step-head"><span className="pb-num">{n}</span><div><b>{title}</b>{sub && <div className="subtitle" style={{ margin: '2px 0 0' }}>{sub}</div>}</div></div>
      {children}
    </div>
  );

  return (
    <div className="studies-page">
      <h1>▶ The Playbook <span className="dim" style={{ fontSize: 13 }}>— buy deep capitulations that smart money is accumulating, in sectors rotating in, ride winners with a trailing stop</span>
        <LastUpdatedChip value={d.computed_at} /></h1>
      <p className="subtitle">
        {d.computed ? <>
          <b className={d.spy_riskoff ? 'bad' : 'good'}>Market: {d.market_regime}</b>{d.vix_regime && <span> · <b className={d.vix_regime === 'spiking' ? 'good' : d.vix_regime === 'calm' ? 'dim' : ''}>VIX {d.vix_level} ({d.vix_regime})</b>{d.vix_regime === 'calm' && <span className="dim"> — distress signals here are idiosyncratic knives, downgrade Mode A</span>}{d.vix_regime === 'spiking' && <span className="good"> — golden window to buy distress</span>}</span>} · {d.n_in} IN · {d.n_leader || 0} leader · {d.n_turning} turning · {d.n_commodity || 0} commodity themes · {d.n_a} Mode-A + {d.n_b} Mode-B candidates
        </> : (d.message || 'Not computed yet.')}
        {d.computed_at && <span className="dim"> · {new Date(d.computed_at).toLocaleString()}</span>}
        <button className="refresh-btn" style={{ marginLeft: 10 }} onClick={run} disabled={running}>{running ? 'Computing…' : 'Refresh'}</button>
      </p>

      <Step n="1" title="Pick the sector — GICS sectors + commodities, trade where rotating IN / leading" sub="IN = ETF outperforming SPY (63d). LEADER/STRONG = above its OWN 200dMA (catches bear-market leaders + trending commodities). Commodities (🛢) are first-class sectors — traded both Mode A & B via their proxy stocks. TURNING = inflecting. OUT = skip.">
        <div className="pb-sectors">
          {(d.sectors || []).filter(s => s.kind !== 'commodity').map(s => (
            <div key={s.sector} className={`pb-sect ${stateCls(s.state)}`}>
              <div className="pb-sect-name">{s.sector} <span className="dim">{s.etf}</span></div>
              <div className="pb-sect-rs">{s.rs_63d > 0 ? '+' : ''}{s.rs_63d}% vs SPY{s.abs_uptrend ? ' · ↑200d' : ''}</div>
              <div className="pb-sect-state">{s.state}{s.state === 'LEADER' ? ' 🐻' : ''}{(s.state === 'TURNING') && (s.rs_turning_up || s.sector_accum) ? ' ↑' : ''}</div>
              {s.top_beta && <div className="pb-sect-beta" title="Highest-beta quality name in the sector (vs SPY, ~1y) — most leveraged play on the move">⚡ {s.top_beta.ticker} <span className="dim">β{s.top_beta.beta}</span></div>}
            </div>
          ))}
        </div>
        <div className="subtitle" style={{ margin: '10px 0 4px' }}>🛢 Commodities (proxy stocks buyable both modes):</div>
        <div className="pb-sectors">
          {(d.sectors || []).filter(s => s.kind === 'commodity').map(s => (
            <div key={s.sector} className={`pb-sect ${stateCls(s.state)}`} style={{ minWidth: 210 }}>
              <div className="pb-sect-name">🛢 {s.sector} <span className="dim">{s.etf}</span></div>
              <div className="pb-sect-rs">{s.rs_63d != null ? `${s.rs_63d > 0 ? '+' : ''}${s.rs_63d}% vs SPY` : ''}{s.abs_uptrend ? ' · ↑200d' : ''}</div>
              <div className="pb-sect-state">{s.state}{s.state === 'LEADER' ? ' 🐻' : ''}</div>
              {s.proxies && s.proxies.length > 0 && <div style={{ marginTop: 5, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {s.proxies.slice(0, 6).map(p => (
                  <span key={p.ticker} className={`pxchip ${p.uptrend ? 'pxup' : ''}`} title={`${p.ret_63d > 0 ? '+' : ''}${p.ret_63d}% 63d${p.uptrend ? ' · uptrend' : ''}`}>{p.ticker}</span>
                ))}
              </div>}
            </div>
          ))}
        </div>
      </Step>

      <Step n="2" title="Pick the mode — whichever is present" sub="Mode A = buy the distress (capitulation + accumulation divergence), then exit when the A/D line flips to DISTRIBUTION (smart money leaves — thesis-symmetric, ~18d). Mode B = buy the dip in an uptrend (price above its 200dMA + RSI(10)<30) and RIDE it ~6 months. Commodities are just Mode B with the commodity's trend as the gate — no separate mode. Together the portfolio backtests +44.7% CAGR / Sharpe 1.69 (OOS +49.3%) vs SPY +20.3%/1.22. Take A when present, else B.">
      </Step>

      <Step n="3" title="Today's candidates through the full funnel" sub="Ranked A-first, IN-sector first, freshest fire first. Quality-filtered (≥$300M, ≥$5, no penny-microcaps). Historical edge = the trigger's best exit from the sweep (directional, survivorship-caveated).">
        <table className="studies-table">
          <thead><tr>
            <SortTh colKey="mode" sort={pbSort}><Term k="mode">Mode</Term></SortTh><SortTh colKey="ticker" sort={pbSort}><Term k="ticker">Ticker</Term></SortTh><SortTh colKey="sector" sort={pbSort}><Term k="sector">Sector</Term></SortTh><SortTh colKey="trigger" sort={pbSort}><Term k="trigger">Trigger</Term></SortTh><SortTh colKey="risk" sort={pbSort}><Term k="risk">Risk</Term></SortTh><SortTh colKey="days_ago" sort={pbSort} align="right"><Term k="fired">Fired</Term></SortTh>
            <SortTh colKey="ad_state" sort={pbSort}><Term k="adline">A/D</Term></SortTh><SortTh colKey="pct_52w" sort={pbSort} align="right"><Term k="pct52w">%52w</Term></SortTh><SortTh colKey="histedge" sort={pbSort} align="right"><Term k="histedge">Hist edge</Term></SortTh>
            <SortTh colKey="last_close" sort={pbSort} align="right"><Term k="lastclose">Last</Term></SortTh><SortTh colKey="market_cap" sort={pbSort}><Term k="marketcap">Mkt cap</Term></SortTh><th><Term k="smartmoney">Smart money</Term></th>
          </tr></thead>
          <tbody>
            {pbSort.rows.map(c => (
              <tr key={c.ticker} className="study-row">
                <td><span className={`pb-mode pb-mode-${c.mode.toLowerCase()}`}>{c.mode}</span></td>
                <td><b>{c.ticker}</b></td>
                <td className="dim" style={{ fontSize: 11 }}>{c.sector} <span className={stateCls(c.sector_state)}>({c.sector_state})</span>{c.off_gate && <span className="offgate-badge" title="Sector is OUT (not leading). Surfaced anyway — sector is a mention of potential, not a filter. Lower-potential / discretionary: no sector tailwind, so judge it on its own merits.">sector OUT</span>}</td>
                <td className="dim" style={{ fontSize: 11 }} title={c.trigger}>{c.trigger}</td>
                <td>{c.risk && <span className={`risk-badge risk-${c.risk.level.toLowerCase().replace(' ', '')}`} title={(c.risk.drivers || []).join('\n')}>{c.risk.level}<span className="risk-score"> {c.risk.score}</span></span>}</td>
                <td style={{ textAlign: 'right' }}>{c.days_ago === 0 ? <span className="good">today</span> : `${c.days_ago}d`}</td>
                <td className="dim" style={{ fontSize: 11 }}>{c.ad_state}</td>
                <td style={{ textAlign: 'right' }} className="dim">{c.pct_52w != null ? `${c.pct_52w}%` : '–'}</td>
                <td style={{ textAlign: 'right' }} className={c.hist_avg_return > 0 ? 'good' : 'dim'}>{c.hist_avg_return != null ? `+${c.hist_avg_return}% / ${c.hist_win_rate}%` : '–'}</td>
                <td style={{ textAlign: 'right' }} className="dim">{c.last_close}</td>
                <td className="dim" style={{ fontSize: 11 }}>{money(c.market_cap)}</td>
                <td style={{ fontSize: 11 }}>{smart(c) || <span className="dim">–</span>}</td>
              </tr>
            ))}
            {!cands.length && <tr><td colSpan={12} className="dim" style={{ padding: '14px 0' }}>No candidates through the funnel right now.</td></tr>}
          </tbody>
        </table>
        {d.n_candidates_total > (d.n_shown || cands.length) && <p className="subtitle" style={{ margin: '8px 0 0' }}>Showing top {d.n_shown || cands.length} of {d.n_candidates_total} — ranked A-first, then IN/LEADER sector, then smart-money conviction, then freshest dip. Mode B fires broadly on market-wide pullback days; the cream is up top.</p>}
      </Step>

      <Step n="4" title="Exit — ride winners, cut losers" sub={d.exit_plan}>
        <p className="subtitle" style={{ margin: 0 }}>~1 in 12 hits +50%; median trade ~+9%. The asymmetry (avg win ≈ 4.6× avg loss) is the edge — many small losses paid for by a few big runners. Size for the base rate, not the dream.</p>
      </Step>

      <div className="pb-caveat">
        ⚠️ <b>Reality check the data can't see:</b> survivorship bias makes backtested numbers an optimistic ceiling (the zeros were deleted). Before buying, apply judgment the model lacks — solvency (can the balance sheet survive the drawdown?), <i>why</i> it's down (one-time vs. structural vs. fraud), and binary catalysts. Smart-money flags (insider / 13D / 13G) are the data's proxy for conviction; your read is the rest.
      </div>
    </div>
  );
}

function PaperTrackRecord() {
  const [d, setD] = useState(null);
  const [running, setRunning] = useState(false);
  const load = () => fetch(`${API}/paper-trades`).then(r => r.json()).then(setD).catch(() => {});
  useEffect(() => { load(); }, []);
  const run = () => { setRunning(true); fetch(`${API}/paper-trades`, { method: 'POST' }).then(() => setTimeout(() => { load(); setRunning(false); }, 12000)).catch(() => setRunning(false)); };
  const groupedTrades = d
    ? (d.trades || []).filter(t => t.status === 'open').concat((d.trades || []).filter(t => t.status === 'closed'))
    : null;
  const pSort = useSortedRows(groupedTrades, null, 'desc', {
    nowexit: t => (t.status === 'closed' ? t.exit_price : t.last_price),
  });
  if (!d) return null;
  const s = d.summary || {};
  const trades = (d.trades || []);
  return (
    <div className="pb-step" style={{ marginTop: 18 }}>
      <div className="pb-step-head"><span className="pb-num">✓</span><div><b>Live paper track record</b><div className="subtitle" style={{ margin: '2px 0 0' }}>Forward out-of-sample: each Playbook pick is paper-bought when it first appears and closed on the sort_gt1 exit. This is the real evidence, accumulating over time.
        <button className="refresh-btn" style={{ marginLeft: 10 }} onClick={run} disabled={running}>{running ? 'Snapshotting…' : 'Snapshot today'}</button></div></div></div>
      {!d.computed ? <p className="dim">{d.message}</p> : <>
        <div className="fwd-cards">
          <div className="fwd-card"><div className="fwd-num">{s.n_open}</div><div className="fwd-lbl">open</div></div>
          <div className="fwd-card"><div className="fwd-num">{s.n_closed}</div><div className="fwd-lbl">closed</div></div>
          <div className="fwd-card"><div className="fwd-num good">{s.closed_win_rate != null ? s.closed_win_rate + '%' : '–'}</div><div className="fwd-lbl">closed win rate</div></div>
          <div className="fwd-card"><div className={`fwd-num ${(s.closed_avg_ret || 0) > 0 ? 'good' : ''}`}>{s.closed_avg_ret != null ? (s.closed_avg_ret > 0 ? '+' : '') + s.closed_avg_ret + '%' : '–'}</div><div className="fwd-lbl">closed avg return</div></div>
          <div className="fwd-card"><div className="fwd-num dim">{s.open_unrealized_avg != null ? (s.open_unrealized_avg > 0 ? '+' : '') + s.open_unrealized_avg + '%' : '–'}</div><div className="fwd-lbl">open unrealized</div></div>
        </div>
        <table className="studies-table">
          <thead><tr><SortTh colKey="status" sort={pSort}><Term k="status">Status</Term></SortTh><SortTh colKey="mode" sort={pSort}><Term k="mode">Mode</Term></SortTh><SortTh colKey="ticker" sort={pSort}><Term k="ticker">Ticker</Term></SortTh><SortTh colKey="sector" sort={pSort}><Term k="sector">Sector</Term></SortTh><SortTh colKey="entry_date" sort={pSort}><Term k="entrydate">Entry</Term></SortTh><SortTh colKey="entry_price" sort={pSort} align="right"><Term k="entry">Entry $</Term></SortTh><SortTh colKey="nowexit" sort={pSort} align="right"><Term k="exitprice">Now/Exit $</Term></SortTh><SortTh colKey="ret_pct" sort={pSort} align="right"><Term k="retcol">Return</Term></SortTh></tr></thead>
          <tbody>
            {pSort.rows.slice(0, 60).map(t => (
              <tr key={`${t.ticker}|${t.entry_date}`} className="study-row">
                <td><span className={t.status === 'open' ? 'good' : 'dim'}>{t.status}</span></td>
                <td><span className={`pb-mode pb-mode-${(t.mode || 'a').toLowerCase()}`}>{t.mode}</span></td>
                <td><b>{t.ticker}</b></td>
                <td className="dim" style={{ fontSize: 11 }}>{t.sector}</td>
                <td className="dim" style={{ fontSize: 11 }}>{t.entry_date}</td>
                <td style={{ textAlign: 'right' }} className="dim">{t.entry_price}</td>
                <td style={{ textAlign: 'right' }} className="dim">{t.status === 'closed' ? t.exit_price : t.last_price}</td>
                <td style={{ textAlign: 'right' }} className={t.ret_pct > 0 ? 'good' : t.ret_pct < 0 ? 'bad' : 'dim'}>{t.ret_pct != null ? (t.ret_pct > 0 ? '+' : '') + t.ret_pct + '%' : '–'}</td>
              </tr>
            ))}
            {!trades.length && <tr><td colSpan={8} className="dim" style={{ padding: '12px 0' }}>No paper trades yet — snapshot to start the forward record.</td></tr>}
          </tbody>
        </table>
      </>}
    </div>
  );
}

function BacktestPage() {
  const ref = useRef(null);
  const [d, setD] = useState(null);
  const [running, setRunning] = useState(false);
  const [gate, setGate] = useState('aggressive');
  const load = () => fetch(`${API}/equity-curve`).then(r => r.json()).then(j => { setD(j); if (j.default_gate) setGate(j.default_gate); }).catch(() => {});
  useEffect(() => { load(); }, []);
  const run = () => {
    setRunning(true);
    fetch(`${API}/equity-curve`, { method: 'POST' }).then(() => {
      const t = setInterval(() => fetch(`${API}/equity-curve`).then(r => r.json()).then(j => {
        if (j.computed && j.computed_at !== (d && d.computed_at)) { clearInterval(t); setRunning(false); setD(j); }
      }).catch(() => {}), 15000);
    }).catch(() => setRunning(false));
  };
  const draw = useCallback(() => {
    const cv = ref.current;
    const v = (d && d.gate_modes && d.gate_modes[gate]) || d;
    if (!cv || !d || !d.computed || !v || !v.dates) return;
    const W = cv.parentElement.clientWidth, H = 320, dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);
    const pad = { l: 44, r: 12, t: 16, b: 22 };
    const all = v.strategy.concat(v.spy); const mn = Math.min(...all), mx = Math.max(...all), rg = (mx - mn) || 1;
    const n = v.dates.length;
    const xAt = i => pad.l + i / (n - 1) * (W - pad.l - pad.r);
    const yAt = v => pad.t + (1 - (v - mn) / rg) * (H - pad.t - pad.b);
    [100, mx].forEach(gl => { const y = yAt(gl); ctx.strokeStyle = '#20262e'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = '#7d8590'; ctx.font = '9px monospace'; ctx.textAlign = 'right'; ctx.fillText(gl.toFixed(0), pad.l - 4, y + 3); });
    const line = (arr, color, w) => { ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath(); arr.forEach((v, i) => { const x = xAt(i), y = yAt(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); };
    line(v.spy, '#7d8590', 1.3); line(v.strategy, '#3fb950', 2);
    ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillStyle = '#3fb950'; ctx.fillText('Strategy', pad.l + 4, 12);
    ctx.fillStyle = '#7d8590'; ctx.fillText('SPY buy-hold', pad.l + 66, 12);
    ctx.font = '10px monospace'; ctx.fillText(v.dates[0], pad.l, H - 6); ctx.textAlign = 'right'; ctx.fillText(v.dates[n - 1], W - pad.r, H - 6);
  }, [d, gate]);
  useEffect(() => { draw(); }, [draw]);
  useEffect(() => { const o = () => draw(); window.addEventListener('resize', o); return () => window.removeEventListener('resize', o); }, [draw]);

  if (!d) return <div className="loading">Loading backtest...</div>;
  const v = (d.gate_modes && d.gate_modes[gate]) || d;
  const gm = d.gate_modes || null;
  const Card = ({ lab, s, b, unit, better }) => (
    <div className="fwd-card"><div className={`fwd-num ${better ? 'good' : ''}`}>{s > 0 && unit === '%' ? '+' : ''}{s}{unit}</div><div className="fwd-lbl">{lab} · SPY {b > 0 && unit === '%' ? '+' : ''}{b}{unit}</div></div>
  );
  return (
    <div className="studies-page">
      <h1>Backtest <span className="dim" style={{ fontSize: 13 }}>— capital-constrained portfolio vs SPY buy-hold</span></h1>
      <p className="subtitle">
        {d.computed ? d.config : (d.message || 'Not computed yet.')}
        {d.computed && <span className="dim"> · {v.n_trades} trades · {new Date(d.computed_at).toLocaleString()}</span>}
        <button className="refresh-btn" style={{ marginLeft: 10 }} onClick={run} disabled={running}>{running ? 'Running (few min)…' : 'Re-run'}</button>
      </p>
      {d.computed && gm && <div className="gate-toggle" style={{ display: 'flex', gap: 8, margin: '4px 0 12px' }}>
        {['aggressive', 'defensive'].filter(k => gm[k]).map(k => (
          <button key={k} onClick={() => setGate(k)} className={`gate-btn ${gate === k ? 'on' : ''}`}
            title={gm[k].label}>
            {k === 'aggressive' ? '⚔️ Aggressive gate (63d RS)' : '🛡️ Defensive gate (21d RS + 200dMA)'}
          </button>
        ))}
        {gm[gate] && <span className="dim" style={{ fontSize: 11, alignSelf: 'center' }}>{gm[gate].label}</span>}
      </div>}
      {!d.computed ? <div className="empty-state" style={{ padding: '30px 0' }}><button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Running…' : 'Run backtest'}</button></div> : <>
        <div className="fwd-cards">
          <Card lab={`CAGR (full)`} s={v.full.strategy.cagr} b={v.full.spy.cagr} unit="%" better={v.full.strategy.cagr > v.full.spy.cagr} />
          <Card lab={`Max drawdown`} s={v.full.strategy.maxdd} b={v.full.spy.maxdd} unit="%" better={v.full.strategy.maxdd > v.full.spy.maxdd} />
          <Card lab={`Sharpe (full)`} s={v.full.strategy.sharpe} b={v.full.spy.sharpe} unit="" better={v.full.strategy.sharpe > v.full.spy.sharpe} />
          {v.oos && <Card lab={`CAGR (out-of-sample '25-26)`} s={v.oos.strategy.cagr} b={v.oos.spy.cagr} unit="%" better={v.oos.strategy.cagr > v.oos.spy.cagr} />}
          {v.oos && <Card lab={`Sharpe (OOS)`} s={v.oos.strategy.sharpe} b={v.oos.spy.sharpe} unit="" better={v.oos.strategy.sharpe > v.oos.spy.sharpe} />}
        </div>
        <div style={{ width: '100%' }}><canvas ref={ref} /></div>
        <p className="subtitle" style={{ marginTop: 8 }}>Both start at 100. Green = strategy, grey = SPY buy-hold. ⚠️ Survivorship-biased (delisted names absent) → absolute figures are an optimistic ceiling; the strategy's edge is the lower drawdown and out-of-sample lead.</p>
      </>}
      <PaperTrackRecord />
    </div>
  );
}

// Lightweight inline SVG equity curve (strategy vs SPY), both normalized together to the
// same y-range. Intentionally NO charting library (CSP/deps forbidden) — a plain <svg>
// with two <polyline>s that scales to its container. Colors come from CSS (theme-agnostic).
function BtCurve({ curve, height = 150 }) {
  if (!curve || curve.length < 2) return null;
  const W = 1000, H = height; // viewBox units; the element scales to container width
  const strat = curve.map(p => p.strat);
  const spy = curve.map(p => p.spy);
  const all = strat.concat(spy).filter(v => v != null);
  if (!all.length) return null;
  const mn = Math.min(...all), mx = Math.max(...all), rg = (mx - mn) || 1;
  const n = curve.length;
  const X = i => (i / (n - 1)) * W;
  const Y = v => H - ((v - mn) / rg) * (H - 4) - 2;
  const pts = arr => arr.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  return (
    <svg className="bt-curve" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Equity curve: strategy vs SPY">
      <polyline className="bt-line-spy" points={pts(spy)} />
      <polyline className="bt-line-strat" points={pts(strat)} />
    </svg>
  );
}

// Shared fetcher for the two small DB-backed backtest payloads. Each analytical section calls
// this and extracts the one curve it cares about. Tolerant of failure → returns nulls (the
// section simply renders no panel). Fetched per-page (endpoints are tiny); no cross-section cache.
function useSectionBacktest() {
  const [lab, setLab] = useState(null);
  const [decomp, setDecomp] = useState(null);
  useEffect(() => {
    let alive = true;
    apiFetch('/backtest-lab').then(d => { if (alive) setLab(d); }).catch(() => {});
    apiFetch('/backtest-decomp').then(d => { if (alive) setDecomp(d); }).catch(() => {});
    return () => { alive = false; };
  }, []);
  return { lab, decomp };
}

// Reusable embedded "backtest vs SPY" equity-curve panel. Sits at the top of an analytical section.
// Default COLLAPSED to a one-line header (title + total-return / vs-SPY chips) so it doesn't push
// the section's real content down; the ▸/▾ toggle expands to stat chips + legend + the curve.
function BacktestPanel({ title, subtitle, curve, summary, collapsible = true }) {
  const [open, setOpen] = useState(!collapsible);
  if (!curve || curve.length < 2) return null;
  const s = summary || {};
  const fmtPct = (v, plus = true) => (v == null || isNaN(v)) ? '–' : `${plus && v > 0 ? '+' : ''}${Number(v).toFixed(1)}%`;
  const fmtNum = (v, d = 2) => (v == null || isNaN(v)) ? '–' : Number(v).toFixed(d);
  // Some summaries carry vs_spy directly; others give total_return + spy_total to subtract.
  const vs = s.vs_spy != null ? s.vs_spy
    : (s.total_return != null && s.spy_total != null ? s.total_return - s.spy_total : null);
  const beat = (s.total_return != null && s.spy_total != null) ? s.total_return > s.spy_total
    : (vs != null && vs > 0);
  const sig = s.t_stat != null && Math.abs(s.t_stat) >= 2;

  const Chip = ({ label, value, cls }) => (
    <div className="bt-embed-chip">
      <div className="bt-embed-chip-label">{label}</div>
      <div className={`bt-embed-chip-val ${cls || ''}`}>{value}</div>
    </div>
  );

  return (
    <div className="bt-embed">
      <div className={`bt-embed-head${collapsible ? ' clickable' : ''}`}
           onClick={collapsible ? () => setOpen(o => !o) : undefined}>
        {collapsible && <span className="bt-embed-toggle">{open ? '▾' : '▸'}</span>}
        <span className="bt-embed-title">{title}</span>
        <span className="bt-embed-head-chips">
          <span className={`bt-embed-mini ${beat ? 'good' : 'bad'}`}>{fmtPct(s.total_return)}</span>
          <span className={`bt-embed-mini ${vs != null && vs > 0 ? 'good' : 'bad'}`}>vs SPY {fmtPct(vs)}</span>
        </span>
      </div>
      {open && (
        <div className="bt-embed-body">
          <div className="bt-embed-chips">
            <Chip label="Total Return" value={fmtPct(s.total_return)} cls={beat ? 'good' : 'bad'} />
            <Chip label="vs SPY" value={fmtPct(vs)} cls={vs != null && vs > 0 ? 'good' : 'bad'} />
            <Chip label="Sharpe" value={fmtNum(s.sharpe)} />
            <Chip label="Max DD" value={fmtPct(s.max_drawdown, false)} cls="bad" />
            <Chip label="Sig (t)" value={fmtNum(s.t_stat)} cls={sig ? 'good' : ''} />
          </div>
          <div className="bt-curve-head">
            <span className="bt-legend"><span className="bt-swatch strat" /> strategy</span>
            <span className="bt-legend"><span className="bt-swatch spy" /> SPY buy-hold</span>
          </div>
          <BtCurve curve={curve} height={120} />
          {subtitle && <p className="subtitle bt-embed-note">{subtitle}</p>}
        </div>
      )}
    </div>
  );
}

// Dashboard-section wrapper: the live board's own RSI+Omega BULLISH rotation rule vs SPY.
function DashboardBacktestPanel() {
  const { lab } = useSectionBacktest();
  const strat = (lab?.phase1?.strategies || []).find(s => (s.label || '').toLowerCase().includes('dashboard rule'));
  if (!strat) return null;
  return (
    <BacktestPanel
      title="Does the live dashboard rule actually beat SPY?"
      subtitle="This is the RSI(10)>SMA(10) + Omega(10)>1 sector-rotation rule the board below scores every sector on — backtested vs SPY buy-hold."
      curve={strat.curve} summary={strat.summary} />
  );
}

// Backtest Lab — read-only view of /backtest-lab. Three phases: (1) rank the rotation RULES,
// (2) check each signal in- vs out-of-sample, (3) a combined portfolio — all vs SPY buy-hold.
// POST triggers a (re)compute; we poll until `computed` flips.
// Rotation-edge decomposition: is the edge the sector ROTATION, the stock PICK, or the COMBINATION?
// Renders the 3-arm decomposition, the 200-day-MA rotation with BOTH numbers (ETF vs pick-after),
// and the value×technical section (buy the cheapest-P/B firer).
function DecompSection({ decomp }) {
  if (!decomp || !decomp.computed) return null;
  const pct = (v, plus = true) => (v == null || isNaN(v)) ? '–' : `${plus && v > 0 ? '+' : ''}${Number(v).toFixed(1)}%`;
  const num = (v, d = 2) => (v == null || isNaN(v)) ? '–' : Number(v).toFixed(d);
  const Row = ({ label, s, strong }) => {
    if (!s) return null;
    const beat = (s.vs_spy ?? 0) > 0;
    const sig = s.t_stat != null && Math.abs(s.t_stat) >= 2;
    return (
      <tr className={strong ? 'bt-beat' : ''}>
        <td>{strong ? <b>{label}</b> : label}</td>
        <td style={{ textAlign: 'right' }} className={beat ? 'good' : ''}><b>{pct(s.total_return)}</b></td>
        <td style={{ textAlign: 'right' }} className={beat ? 'good' : 'bad'}>{pct(s.vs_spy)}</td>
        <td style={{ textAlign: 'right' }}>{pct(s.annual_return)}</td>
        <td style={{ textAlign: 'right' }}>{num(s.sharpe)}</td>
        <td style={{ textAlign: 'right' }} className="bad">{pct(s.max_drawdown, false)}</td>
        <td style={{ textAlign: 'right' }} className={sig ? 'good' : s.t_stat != null && Math.abs(s.t_stat) < 1 ? 'bad' : ''}>{num(s.t_stat)}</td>
        <td style={{ textAlign: 'right' }}>{s.periods ?? '–'}</td>
      </tr>
    );
  };
  const Head = () => (
    <thead><tr>
      <th>Arm</th><th style={{ textAlign: 'right' }}>Total</th><th style={{ textAlign: 'right' }}>vs SPY</th>
      <th style={{ textAlign: 'right' }}>Ann%</th><th style={{ textAlign: 'right' }}>Sharpe</th>
      <th style={{ textAlign: 'right' }}>Max DD</th><th style={{ textAlign: 'right' }}>Sig (t)</th>
      <th style={{ textAlign: 'right' }}>Periods</th>
    </tr></thead>
  );
  const dec = decomp.decomposition || {};
  const sma = decomp.sma200_rotation;
  const vxt = decomp.value_x_technical;
  return (
    <>
      <h2 className="bt-section">Where the edge comes from — 3-arm decomposition <span className="dim" style={{ fontSize: 12 }}>(monthly, equal-weight, point-in-time)</span></h2>
      <p className="subtitle">If <b>rotation + pick</b> ≈ <b>pick only</b>, the edge is the stock PICK; if it ≈ <b>rotation only</b>, it's the ROTATION. The cheapest-P/B (<b>lowpb</b>) pick is the standout.</p>
      {['lowpb', 'momentum'].map(kind => dec[kind] && (
        <div key={kind} style={{ marginBottom: 14 }}>
          <h3 className="bt-subhead">{kind === 'lowpb' ? 'Cheapest-P/B (value) pick' : 'Momentum pick'}</h3>
          <div className="bt-table-wrap"><table className="studies-table bt-table"><Head />
            <tbody>
              <Row label="Pick only (whole universe)" s={dec[kind].arm1_pick_only?.summary} strong={kind === 'lowpb'} />
              <Row label="Rotation only (top momentum sectors, hold ETF)" s={dec[kind].arm2_rotation_only?.summary} />
              <Row label="Rotation + pick (rotate, then pick 1 stock/sector)" s={dec[kind].arm3_rotation_plus_pick?.summary} strong={kind === 'lowpb'} />
            </tbody>
          </table></div>
        </div>
      ))}

      {sma && <>
        <h2 className="bt-section">200-day MA rotation — both numbers</h2>
        <p className="subtitle">Rotate into sectors trading above their 200-day MA, then compare holding the <b>ETF</b> vs taking the <b>stock after</b> (the key comparison).</p>
        <div className="bt-table-wrap"><table className="studies-table bt-table"><Head />
          <tbody>
            <Row label="Rotation only (hold the ETF)" s={sma.rotation_only?.summary} />
            <Row label="+ cheapest-P/B stock pick after" s={sma.rotation_plus_pick?.lowpb?.summary} strong />
            <Row label="+ momentum stock pick after" s={sma.rotation_plus_pick?.momentum?.summary} />
          </tbody>
        </table></div>
      </>}

      {vxt && (vxt.signals || []).length > 0 && <>
        <h2 className="bt-section">Value × technical — buy the cheapest-P/B firer</h2>
        <p className="subtitle">Among the stocks firing a technical signal each month, buy the single cheapest-P/B one, vs the signal alone.</p>
        <div className="bt-table-wrap"><table className="studies-table bt-table"><Head />
          <tbody>
            <Row label="Cheapest-P/B alone (whole universe)" s={vxt.lowpb_alone?.summary} strong />
            {(vxt.signals || []).filter(sg => sg.present && sg.signal_alone).map(sg => [
              <Row key={sg.signal_key + '-a'} label={`${sg.signal_name} — signal alone`} s={sg.signal_alone?.summary} />,
              <Row key={sg.signal_key + '-b'} label={`${sg.signal_name} — + cheapest-P/B pick`} s={sg.signal_plus_lowpb?.summary} strong />,
            ])}
          </tbody>
        </table></div>
      </>}
    </>
  );
}

function BacktestLabPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [sortBy, setSortBy] = useState('total_return');
  const [sortDir, setSortDir] = useState('desc');
  const [decomp, setDecomp] = useState(null);
  const p2Sort = useSortedRows(data && data.phase2 && data.phase2.signals, null, 'desc', {
    is_avg: s => (s.is || {}).avg,
    is_t: s => (s.is || {}).t,
    is_n: s => (s.is || {}).n,
    oos_avg: s => (s.oos || {}).avg,
    oos_t: s => (s.oos || {}).t,
    oos_n: s => (s.oos || {}).n,
  });

  const load = () => {
    setLoading(true); setErr(null);
    apiFetch('/backtest-lab')
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
    apiFetch('/backtest-decomp').then(d => setDecomp(d)).catch(() => {});
  };
  useEffect(() => { load(); }, []);

  const run = () => {
    setRunning(true);
    fetch(`${API}/backtest-lab`, { method: 'POST' }).then(() => {
      const t = setInterval(() => {
        apiFetch('/backtest-lab').then(j => {
          if (j.computed) { clearInterval(t); setRunning(false); setData(j); }
        }).catch(() => {});
      }, 15000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><h1>Backtest Lab</h1><ErrorBanner message={err} onRetry={load} /></div>;
  if (loading || !data) return <div className="loading">Loading backtest lab...</div>;

  if (!data.computed) {
    return (
      <div className="studies-page">
        <h1>Backtest Lab</h1>
        <p className="subtitle">Validates the rotation rules, then checks each signal in- vs out-of-sample, then a combined portfolio — all vs SPY buy-hold.</p>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Running (few min)…' : 'Run backtest'}</button>
        </div>
      </div>
    );
  }

  const fmtPct = (v, plus = true) => (v == null || isNaN(v)) ? '–' : `${plus && v > 0 ? '+' : ''}${Number(v).toFixed(1)}%`;
  const fmtNum = (v, d = 2) => (v == null || isNaN(v)) ? '–' : Number(v).toFixed(d);

  const p1 = data.phase1 || {};
  const strategies = (p1.strategies || []).slice();
  const setSort = (col) => { if (sortBy === col) setSortDir(x => x === 'desc' ? 'asc' : 'desc'); else { setSortBy(col); setSortDir('desc'); } };
  const arrow = (col) => sortBy === col ? (sortDir === 'desc' ? ' ▼' : ' ▲') : '';
  const accessor = (s) => {
    const su = s.summary || {};
    switch (sortBy) {
      case 'label': return s.label || '';
      case 'hold_months': return s.hold_months ?? 0;
      case 'total_return': return su.total_return ?? 0;
      case 'vs_spy': return (su.total_return ?? 0) - (su.spy_total ?? 0);
      case 'annual_return': return su.annual_return ?? 0;
      case 'sharpe': return su.sharpe ?? 0;
      case 'max_drawdown': return su.max_drawdown ?? 0;
      case 't_stat': return su.t_stat ?? 0;
      case 'periods': return su.periods ?? 0;
      default: return 0;
    }
  };
  strategies.sort((a, b) => {
    const av = accessor(a), bv = accessor(b);
    if (typeof av === 'string') return sortDir === 'desc' ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
    return sortDir === 'desc' ? bv - av : av - bv;
  });

  // Years union across all strategies' by_year, for the per-year beats-SPY grid.
  const years = [...new Set(strategies.flatMap(s => Object.keys(s.by_year || {})))].sort();

  // Best strategy by total return, for the equity-curve preview.
  const topStrat = (p1.strategies || []).reduce((best, s) =>
    (!best || (s.summary?.total_return ?? -Infinity) > (best.summary?.total_return ?? -Infinity)) ? s : best, null);

  const p2 = data.phase2;
  const p3 = data.phase3;
  const updated = data.last_updated || data.computed_at;

  return (
    <div className="studies-page">
      <h1>Backtest Lab <span className="dim" style={{ fontSize: 13 }}>— rules, robustness &amp; portfolio vs SPY</span>
        <LastUpdatedChip value={updated} />
      </h1>
      <p className="subtitle">
        Three phases: (1) rank the rotation <b>rules</b>, (2) check each <b>signal</b> in- vs out-of-sample, (3) a combined <b>portfolio</b> — all against SPY buy-hold.
        {data.universe != null && <> Universe: {data.universe}.</>}
        <button className="refresh-btn" style={{ marginLeft: 10 }} onClick={run} disabled={running}>{running ? 'Running (few min)…' : 'Re-run'}</button>
      </p>

      {strategies.length > 0 && <>
        <h2 className="bt-section">Phase 1 — Rotation rules</h2>
        <div className="bt-table-wrap">
          <table className="studies-table bt-table">
            <thead><tr>
              <th onClick={() => setSort('label')} style={{ cursor: 'pointer' }}>Strategy{arrow('label')}</th>
              <th onClick={() => setSort('hold_months')} style={{ cursor: 'pointer', textAlign: 'right' }}>Hold (mo){arrow('hold_months')}</th>
              <th onClick={() => setSort('total_return')} style={{ cursor: 'pointer', textAlign: 'right' }}>Total Return{arrow('total_return')}</th>
              <th onClick={() => setSort('vs_spy')} style={{ cursor: 'pointer', textAlign: 'right' }}>vs SPY{arrow('vs_spy')}</th>
              <th onClick={() => setSort('annual_return')} style={{ cursor: 'pointer', textAlign: 'right' }}><Term k="cagr">Ann%</Term>{arrow('annual_return')}</th>
              <th onClick={() => setSort('sharpe')} style={{ cursor: 'pointer', textAlign: 'right' }}><Term k="sharpe">Sharpe</Term>{arrow('sharpe')}</th>
              <th onClick={() => setSort('max_drawdown')} style={{ cursor: 'pointer', textAlign: 'right' }}><Term k="drawdown">Max DD</Term>{arrow('max_drawdown')}</th>
              <th onClick={() => setSort('t_stat')} style={{ cursor: 'pointer', textAlign: 'right' }}>Sig (t){arrow('t_stat')}</th>
              <th onClick={() => setSort('periods')} style={{ cursor: 'pointer', textAlign: 'right' }}>Periods{arrow('periods')}</th>
            </tr></thead>
            <tbody>
              {strategies.map((s, i) => {
                const su = s.summary || {};
                const beat = (su.total_return ?? 0) > (su.spy_total ?? 0);
                const delta = (su.total_return ?? 0) - (su.spy_total ?? 0);
                const sig = su.t_stat != null && Math.abs(su.t_stat) >= 2;
                return (
                  <tr key={s.label || i} className={beat ? 'bt-beat' : ''}>
                    <td>{s.label}</td>
                    <td style={{ textAlign: 'right' }}>{s.hold_months ?? '–'}</td>
                    <td style={{ textAlign: 'right' }} className={beat ? 'good' : ''}><b>{fmtPct(su.total_return)}</b></td>
                    <td style={{ textAlign: 'right' }} className={delta > 0 ? 'good' : 'bad'} title={`SPY ${fmtPct(su.spy_total)}`}>{fmtPct(delta)}</td>
                    <td style={{ textAlign: 'right' }}>{fmtPct(su.annual_return)}</td>
                    <td style={{ textAlign: 'right' }}>{fmtNum(su.sharpe)}</td>
                    <td style={{ textAlign: 'right' }} className="bad">{fmtPct(su.max_drawdown, false)}</td>
                    <td style={{ textAlign: 'right' }} className={sig ? 'good' : su.t_stat != null && Math.abs(su.t_stat) < 1 ? 'bad' : ''}>{fmtNum(su.t_stat)}</td>
                    <td style={{ textAlign: 'right' }}>{su.periods ?? '–'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {decomp && decomp.computed && (() => {
          const dc = decomp.decomposition || {};
          const rpp = (decomp.sma200_rotation || {}).rotation_plus_pick || {};
          const pickArms = [
            { label: '6-mo momentum rotation → cheapest-P/B (value) stock pick', s: dc.lowpb && dc.lowpb.arm3_rotation_plus_pick && dc.lowpb.arm3_rotation_plus_pick.summary },
            { label: '6-mo momentum rotation → momentum stock pick', s: dc.momentum && dc.momentum.arm3_rotation_plus_pick && dc.momentum.arm3_rotation_plus_pick.summary },
            { label: '200-day-MA rotation → cheapest-P/B (value) stock pick', s: rpp.lowpb && rpp.lowpb.summary },
            { label: '200-day-MA rotation → momentum stock pick', s: rpp.momentum && rpp.momentum.summary },
          ].filter(a => a.s);
          if (!pickArms.length) return null;
          return (
            <div className="bt-pick-beats" style={{ marginTop: 20 }}>
              <h2 className="bt-section">…but add the stock pick and it beats SPY</h2>
              <p className="subtitle">ETF rotation alone doesn't beat SPY — rotating then buying the cheapest-P/B stock in each winning sector does (+264%, t2.19).</p>
              <div className="bt-table-wrap">
                <table className="studies-table bt-table">
                  <thead><tr>
                    <th>Rotation + pick</th>
                    <th style={{ textAlign: 'right' }}>Total Return</th>
                    <th style={{ textAlign: 'right' }}>vs SPY</th>
                    <th style={{ textAlign: 'right' }}><Term k="sharpe">Sharpe</Term></th>
                    <th style={{ textAlign: 'right' }}><Term k="drawdown">Max DD</Term></th>
                    <th style={{ textAlign: 'right' }}>Sig (t)</th>
                    <th style={{ textAlign: 'right' }}>Periods</th>
                  </tr></thead>
                  <tbody>
                    {pickArms.map((a, i) => {
                      const s = a.s;
                      const beat = (s.vs_spy ?? 0) > 0;
                      const sig = s.t_stat != null && Math.abs(s.t_stat) >= 2;
                      return (
                        <tr key={i} className={beat ? 'bt-beat' : ''}>
                          <td>{beat ? <b>{a.label}</b> : a.label}</td>
                          <td style={{ textAlign: 'right' }} className={beat ? 'good' : ''}><b>{fmtPct(s.total_return)}</b></td>
                          <td style={{ textAlign: 'right' }} className={beat ? 'good' : 'bad'}>{fmtPct(s.vs_spy)}</td>
                          <td style={{ textAlign: 'right' }}>{fmtNum(s.sharpe)}</td>
                          <td style={{ textAlign: 'right' }} className="bad">{fmtPct(s.max_drawdown, false)}</td>
                          <td style={{ textAlign: 'right' }} className={sig ? 'good' : s.t_stat != null && Math.abs(s.t_stat) < 1 ? 'bad' : ''}>{fmtNum(s.t_stat)}</td>
                          <td style={{ textAlign: 'right' }}>{s.periods ?? '–'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })()}

        {topStrat && topStrat.curve && topStrat.curve.length > 1 && <>
          <div className="bt-curve-head">
            <span className="bt-legend"><span className="bt-swatch strat" /> {topStrat.label} <span className="dim">(top)</span></span>
            <span className="bt-legend"><span className="bt-swatch spy" /> SPY buy-hold</span>
            <span className="dim" style={{ fontSize: 11 }}>{topStrat.curve[0].date} → {topStrat.curve[topStrat.curve.length - 1].date}</span>
          </div>
          <BtCurve curve={topStrat.curve} />
        </>}

        {years.length > 0 && <>
          <h3 className="bt-subhead">Beats SPY by year <span className="dim">(green = strategy beat SPY that year)</span></h3>
          <div className="bt-table-wrap">
            <table className="studies-table bt-year-grid">
              <thead><tr><th>Strategy</th>{years.map(y => <th key={y} style={{ textAlign: 'center' }}>{y}</th>)}</tr></thead>
              <tbody>
                {strategies.map((s, i) => (
                  <tr key={s.label || i}>
                    <td>{s.label}</td>
                    {years.map(y => {
                      const cell = (s.by_year || {})[y];
                      if (!cell) return <td key={y} className="bt-cell dim" style={{ textAlign: 'center' }}>–</td>;
                      return <td key={y} className={`bt-cell ${cell.beats ? 'bt-win' : 'bt-lose'}`} style={{ textAlign: 'center' }}
                        title={`${s.label} ${y}: ${fmtPct(cell.strat)} vs SPY ${fmtPct(cell.spy)}`}>{fmtPct(cell.strat)}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>}
      </>}

      {p2 && p2.signals && p2.signals.length > 0 && <>
        <h2 className="bt-section">Phase 2 — Signal robustness (in vs out-of-sample)</h2>
        <p className="subtitle">In-sample before <b>{p2.cutoff_date || '—'}</b>, out-of-sample after. A signal holds up when its out-of-sample t-stat stays ≥ 2 (green row).</p>
        <div className="bt-table-wrap">
          <table className="studies-table bt-table">
            <thead><tr>
              <SortTh colKey="signal" sort={p2Sort}>Signal</SortTh><SortTh colKey="exit" sort={p2Sort}>Exit</SortTh>
              <SortTh colKey="is_avg" sort={p2Sort} align="right">IS avg</SortTh><SortTh colKey="is_t" sort={p2Sort} align="right">IS t</SortTh><SortTh colKey="is_n" sort={p2Sort} align="right">IS n</SortTh>
              <SortTh colKey="oos_avg" sort={p2Sort} align="right">OOS avg</SortTh><SortTh colKey="oos_t" sort={p2Sort} align="right">OOS t</SortTh><SortTh colKey="oos_n" sort={p2Sort} align="right">OOS n</SortTh>
            </tr></thead>
            <tbody>
              {p2Sort.rows.map((s, i) => {
                const is = s.is || {}, oos = s.oos || {};
                const robust = oos.t != null && Math.abs(oos.t) >= 2;
                return (
                  <tr key={(s.signal_key || s.signal || i) + '|' + (s.exit || i)} className={robust ? 'bt-beat' : ''}>
                    <td title={s.signal_key}>{s.signal}</td>
                    <td>{s.exit}</td>
                    <td style={{ textAlign: 'right' }} className={is.avg > 0 ? 'good' : is.avg < 0 ? 'bad' : ''}>{fmtPct(is.avg)}</td>
                    <td style={{ textAlign: 'right' }} className={is.t != null && Math.abs(is.t) >= 2 ? 'good' : ''}>{fmtNum(is.t)}</td>
                    <td style={{ textAlign: 'right' }} className="dim">{is.n ?? '–'}</td>
                    <td style={{ textAlign: 'right' }} className={oos.avg > 0 ? 'good' : oos.avg < 0 ? 'bad' : ''}>{fmtPct(oos.avg)}</td>
                    <td style={{ textAlign: 'right' }} className={robust ? 'good' : oos.t != null && Math.abs(oos.t) < 1 ? 'bad' : ''}><b>{fmtNum(oos.t)}</b></td>
                    <td style={{ textAlign: 'right' }} className="dim">{oos.n ?? '–'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </>}

      {p3 && <>
        <h2 className="bt-section">Phase 3 — Combined portfolio{p3.label ? ` — ${p3.label}` : ''}</h2>
        {p3.summary && (
          <p className="subtitle">
            {p3.summary.total_return != null && <>Total <b className={(p3.summary.total_return ?? 0) > (p3.summary.spy_total ?? 0) ? 'good' : ''}>{fmtPct(p3.summary.total_return)}</b> vs SPY {fmtPct(p3.summary.spy_total)}. </>}
            {p3.summary.annual_return != null && <>Ann {fmtPct(p3.summary.annual_return)}. </>}
            {p3.summary.sharpe != null && <>Sharpe {fmtNum(p3.summary.sharpe)}. </>}
            {p3.summary.max_drawdown != null && <>Max DD {fmtPct(p3.summary.max_drawdown, false)}. </>}
            {p3.summary.t_stat != null && <>Sig (t) {fmtNum(p3.summary.t_stat)}.</>}
          </p>
        )}
        {p3.curve && p3.curve.length > 1 && <>
          <div className="bt-curve-head">
            <span className="bt-legend"><span className="bt-swatch strat" /> Portfolio</span>
            <span className="bt-legend"><span className="bt-swatch spy" /> SPY buy-hold</span>
          </div>
          <BtCurve curve={p3.curve} />
        </>}
        <p className="subtitle" style={{ marginTop: 6 }}>The combined portfolio blends the surviving signals into one capital-constrained book.</p>
      </>}

      <DecompSection decomp={decomp} />

      <p className="subtitle" style={{ marginTop: 14 }}>⚠️ In-sample figures are survivorship-biased — use them for <i>ranking</i>, not promises. The out-of-sample column (Phase 2) is the honest robustness check.</p>
    </div>
  );
}

function ResearchPage() {
  const [d, setD] = useState(null);
  const [running, setRunning] = useState(false);
  const load = () => fetch(`${API}/research`).then(r => r.json()).then(setD).catch(() => {});
  useEffect(() => { load(); }, []);
  const run = () => {
    setRunning(true);
    fetch(`${API}/research`, { method: 'POST' }).then(() => {
      const t = setInterval(() => fetch(`${API}/research`).then(r => r.json()).then(j => {
        if (j.computed && j.computed_at !== (d && d.computed_at)) { clearInterval(t); setRunning(false); setD(j); }
      }).catch(() => {}), 15000);
    }).catch(() => setRunning(false));
  };
  const rMatrix = useSortedRows(d && d.matrix, null, 'desc');
  const rTf = useSortedRows(d && d.timeframe, null, 'desc');
  const rReg = useSortedRows(d && d.regimes, null, 'desc');
  const rCap = useSortedRows(d && d.capband, null, 'desc');
  const rTrig = useSortedRows(d && d.triggers, null, 'desc');
  if (!d) return <div className="loading">Loading research...</div>;
  const pct = (v, plus) => v == null ? '–' : `${plus && v > 0 ? '+' : ''}${v}%`;
  const best = (arr, key) => Math.max(...arr.map(x => x[key] ?? -1e9));
  const Sec = ({ t, sub, children }) => <div className="pb-step" style={{ marginTop: 14 }}><div className="pb-step-head"><span className="pb-num" style={{ background: '#6e40c9' }}>◔</span><div><b>{t}</b>{sub && <div className="subtitle" style={{ margin: '2px 0 0' }}>{sub}</div>}</div></div>{children}</div>;

  return (
    <div className="studies-page">
      <h1>Research / Lab <span className="dim" style={{ fontSize: 13 }}>— every strategy comparison we ran, cached & re-runnable</span>
        <LastUpdatedChip value={d.computed_at} /></h1>
      <p className="subtitle">
        {d.computed ? `${d.n_daily} daily · ${d.n_weekly} weekly · ${d.n_mix} mix entries · computed ${new Date(d.computed_at).toLocaleString()}` : (d.message || 'Not computed yet.')}
        <button className="refresh-btn" style={{ marginLeft: 10 }} onClick={run} disabled={running}>{running ? 'Running (few min)…' : 'Re-run all'}</button>
        <span className="dim"> · all portfolio metrics use SPY-overlay + SPY&lt;200dMA cash · survivorship-caveated</span>
      </p>
      {!d.computed ? <div className="empty-state" style={{ padding: '30px 0' }}><button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Running…' : 'Run research'}</button></div> : <>

        <Sec t="① Trigger × Exit matrix" sub="Which entry trigger and exit gives the best portfolio. Green = best CAGR / Sharpe in column.">
          <table className="studies-table"><thead><tr><SortTh colKey="trigger" sort={rMatrix}><Term k="signal">Trigger</Term></SortTh><SortTh colKey="exit" sort={rMatrix}><Term k="exit">Exit</Term></SortTh><SortTh colKey="cagr" sort={rMatrix} align="right"><Term k="cagr">CAGR</Term></SortTh><SortTh colKey="dd" sort={rMatrix} align="right"><Term k="drawdown">Max DD</Term></SortTh><SortTh colKey="sharpe" sort={rMatrix} align="right"><Term k="sharpe">Sharpe</Term></SortTh><SortTh colKey="oos_cagr" sort={rMatrix} align="right"><Term k="cagr">OOS CAGR</Term></SortTh></tr></thead>
            <tbody>{rMatrix.rows.map((m, i) => (
              <tr key={i} className="study-row">
                <td>{m.trigger}</td><td className="dim">{m.exit}</td>
                <td style={{ textAlign: 'right' }} className={m.cagr === best(d.matrix, 'cagr') ? 'good' : ''}>{pct(m.cagr, 1)}</td>
                <td style={{ textAlign: 'right' }} className="dim">{pct(m.dd)}</td>
                <td style={{ textAlign: 'right' }} className={m.sharpe === best(d.matrix, 'sharpe') ? 'good' : ''}>{m.sharpe}</td>
                <td style={{ textAlign: 'right' }} className="dim">{pct(m.oos_cagr, 1)}</td>
              </tr>))}</tbody></table>
        </Sec>

        <Sec t="② Entry timeframe" sub="Daily vs weekly vs multi-timeframe mix (exit = sort_gt1).">
          <table className="studies-table"><thead><tr><SortTh colKey="tf" sort={rTf}><Term k="timeframe">Timeframe</Term></SortTh><SortTh colKey="n" sort={rTf} align="right"><Term k="ntrades">n</Term></SortTh><SortTh colKey="cagr" sort={rTf} align="right"><Term k="cagr">CAGR</Term></SortTh><SortTh colKey="dd" sort={rTf} align="right"><Term k="drawdown">Max DD</Term></SortTh><SortTh colKey="sharpe" sort={rTf} align="right"><Term k="sharpe">Sharpe</Term></SortTh></tr></thead>
            <tbody>{rTf.rows.map((t, i) => (<tr key={i} className="study-row"><td><b>{t.tf}</b></td><td style={{ textAlign: 'right' }} className="dim">{t.n}</td><td style={{ textAlign: 'right' }}>{pct(t.cagr, 1)}</td><td style={{ textAlign: 'right' }} className="dim">{pct(t.dd)}</td><td style={{ textAlign: 'right' }} className={t.sharpe === best(d.timeframe, 'sharpe') ? 'good' : ''}>{t.sharpe}</td></tr>))}</tbody></table>
        </Sec>

        <Sec t="③ Market regime (when to be in cash)" sub="Price (SPY<200dMA) vs curve-steepening vs always-in.">
          <table className="studies-table"><thead><tr><SortTh colKey="regime" sort={rReg}><Term k="regime">Regime</Term></SortTh><SortTh colKey="cagr" sort={rReg} align="right"><Term k="cagr">CAGR</Term></SortTh><SortTh colKey="dd" sort={rReg} align="right"><Term k="drawdown">Max DD</Term></SortTh><SortTh colKey="sharpe" sort={rReg} align="right"><Term k="sharpe">Sharpe</Term></SortTh></tr></thead>
            <tbody>{rReg.rows.map((x, i) => (<tr key={i} className="study-row"><td>{x.regime}</td><td style={{ textAlign: 'right' }}>{pct(x.cagr, 1)}</td><td style={{ textAlign: 'right' }} className="dim">{pct(x.dd)}</td><td style={{ textAlign: 'right' }} className={x.sharpe === best(d.regimes, 'sharpe') ? 'good' : ''}>{x.sharpe}</td></tr>))}</tbody></table>
        </Sec>

        <Sec t="④ Risk by market-cap band" sub="Where the win rate and the 50%+ upside live.">
          <table className="studies-table"><thead><tr><SortTh colKey="band" sort={rCap}><Term k="capband">Cap band</Term></SortTh><SortTh colKey="n" sort={rCap} align="right"><Term k="ntrades">n</Term></SortTh><SortTh colKey="win" sort={rCap} align="right"><Term k="winrate">Win%</Term></SortTh><SortTh colKey="avg" sort={rCap} align="right"><Term k="avgreturn">Avg</Term></SortTh><SortTh colKey="hit50" sort={rCap} align="right"><Term k="hit50">Hit +50%</Term></SortTh></tr></thead>
            <tbody>{rCap.rows.map((c, i) => (<tr key={i} className="study-row"><td>{c.band}</td><td style={{ textAlign: 'right' }} className="dim">{c.n}</td><td style={{ textAlign: 'right' }} className={c.win === best(d.capband, 'win') ? 'good' : ''}>{c.win}%</td><td style={{ textAlign: 'right' }}>{pct(c.avg, 1)}</td><td style={{ textAlign: 'right' }} className={c.hit50 === best(d.capband, 'hit50') ? 'good' : 'dim'}>{c.hit50}%</td></tr>))}</tbody></table>
        </Sec>

        {d.mpt && <Sec t="⑤ MPT allocation across cap bands" sub="Inverse-vol weights + per-band Sharpe (bands are weakly correlated → diversification helps).">
          <table className="studies-table"><thead><tr><th><Term k="capband">Band</Term></th><th style={{ textAlign: 'right' }}><Term k="invvol">Inverse-vol weight</Term></th><th style={{ textAlign: 'right' }}><Term k="sharpe">Sharpe</Term></th></tr></thead>
            <tbody>{d.mpt.bands.map((b, i) => (<tr key={i} className="study-row"><td>{b}</td><td style={{ textAlign: 'right' }}>{d.mpt.inverse_vol_weights[i]}%</td><td style={{ textAlign: 'right' }} className="dim">{d.mpt.sharpe_per_band[b]}</td></tr>))}</tbody></table>
        </Sec>}

        {d.triggers && <Sec t="⑥ Trigger × Accumulation grid" sub="Per-trade (sort_gt1 exit, Mode A only). Shows each trigger with vs without the accum-divergence requirement — accumulation is the biggest single lever.">
          <table className="studies-table"><thead><tr><SortTh colKey="trigger" sort={rTrig}><Term k="trigger">Trigger</Term></SortTh><SortTh colKey="accum" sort={rTrig}><Term k="accumdiv">Accum-div?</Term></SortTh><SortTh colKey="n" sort={rTrig} align="right"><Term k="ntrades">n</Term></SortTh><SortTh colKey="win" sort={rTrig} align="right"><Term k="winrate">Win%</Term></SortTh><SortTh colKey="avg" sort={rTrig} align="right"><Term k="avgreturn">Avg</Term></SortTh><SortTh colKey="median" sort={rTrig} align="right"><Term k="median">Median</Term></SortTh></tr></thead>
            <tbody>{rTrig.rows.map((t, i) => (
              <tr key={i} className="study-row">
                <td>{t.trigger}</td>
                <td className={t.accum ? 'good' : 'dim'}>{t.accum ? 'required' : 'ignored'}</td>
                <td style={{ textAlign: 'right' }} className="dim">{t.n}</td>
                <td style={{ textAlign: 'right' }} className={t.win === best(d.triggers, 'win') ? 'good' : ''}>{t.win}%</td>
                <td style={{ textAlign: 'right' }} className={t.avg === best(d.triggers, 'avg') ? 'good' : ''}>+{t.avg}%</td>
                <td style={{ textAlign: 'right' }} className="dim">+{t.median}%</td>
              </tr>))}</tbody></table>
        </Sec>}

        <p className="subtitle" style={{ marginTop: 12 }}>⚠️ All in-sample (2021–2026) and survivorship-biased — use for <i>ranking</i> options, not absolute promises. The paper tracker (Backtest tab) is the unbiased forward test.</p>
      </>}
    </div>
  );
}

// ---- Live hub: Short-term Burst ----------------------------------------------
// Names bursting right now (trigger fired in the last ~2 bars), tagged momentum vs reversal.
function ShortTermPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);
  const [burstFilter, setBurstFilter] = useState('all');   // all | momentum | reversal (client-side)

  const filteredRows = React.useMemo(
    () => ((data && data.results) || []).filter(r => burstFilter === 'all' || r.burst_type === burstFilter),
    [data, burstFilter]);
  const sort = useSortedRows(filteredRows, 'days_ago', 'asc');

  const load = () => apiFetch('/short-term').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);

  const runScan = () => {
    setRunning(true);
    fetch(`${API}/short-term`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/short-term`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading short-term burst signals...</div>;
  if (data.computed === false) {
    return (
      <div className="studies-page">
        <h1>Short-term Burst</h1>
        <p className="subtitle">Names bursting right now, tagged momentum or reversal.</p>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.message || 'Not scanned yet.'}</p>
          <button className="refresh-btn" onClick={runScan} disabled={running}>{running ? 'Scanning...' : 'Run scan'}</button>
        </div>
      </div>
    );
  }

  const capFmt = v => v == null ? '–' : v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(0)}M` : `$${(v / 1e3).toFixed(0)}K`;
  const signCls = v => v == null ? 'dim' : v > 0 ? 'good' : v < 0 ? 'bad' : 'dim';
  const burstChip = t => t === 'reversal'
    ? <span className="sm-badge sm-13g" title="Reversal — snapping back off a low">reversal</span>
    : <span className="sm-badge sm-insider" title="Momentum — thrusting up">momentum</span>;
  const smChips = r => {
    const fmt = v => v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : `$${(v / 1e3).toFixed(0)}K`;
    const p = [];
    if (r.insider_buy_90d > 0) p.push(<span key="i" className="sm-badge sm-insider" title="Insider open-market buys, ~trailing 6mo (SEC bulk data lags ~1 quarter)">▲ {fmt(r.insider_buy_90d)} insider</span>);
    if (r.recent_13d > 0) p.push(<span key="d" className="sm-badge sm-13d" title="Activist 13D filings, last 180d">13D×{r.recent_13d}</span>);
    if (r.recent_13g > 0) p.push(<span key="g" className="sm-badge sm-13g" title="Passive institutional 13G filings, last 180d">13G×{r.recent_13g}</span>);
    return p.length ? p : <span className="dim">–</span>;
  };

  return (
    <div className="studies-page">
      <h1>Short-term Burst <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle">
        Names bursting right now (trigger fired in the last ~2 bars), tagged momentum (thrusting up) or reversal (snapping back).
        Edge is the trigger's short-horizon historical result.
        <span className="dim"> · <b className="good">{data.n_momentum}</b> momentum · <b>{data.n_reversal}</b> reversal</span>
      </p>

      <div className="studies-controls">
        <div className="filters">
          <span className="dim" style={{ fontSize: 11 }}>Burst:</span>
          {['all', 'momentum', 'reversal'].map(b => (
            <button key={b} className={burstFilter === b ? 'active' : ''} onClick={() => setBurstFilter(b)}>
              {b === 'all' ? 'All' : b === 'momentum' ? 'Momentum' : 'Reversal'}
            </button>
          ))}
          <button className="refresh-btn" onClick={runScan} disabled={running}>{running ? 'Scanning...' : 'Re-scan'}</button>
        </div>
      </div>

      <table className="studies-table">
        <thead><tr>
          <SortTh colKey="ticker" sort={sort}><Term k="ticker">Ticker</Term></SortTh>
          <SortTh colKey="burst_type" sort={sort}>Burst</SortTh>
          <SortTh colKey="signal_name" sort={sort}><Term k="signal">Signal</Term></SortTh>
          <SortTh colKey="days_ago" sort={sort} align="right"><Term k="fired">Days ago</Term></SortTh>
          <SortTh colKey="day1_move" sort={sort} align="right">Day-1 move</SortTh>
          <SortTh colKey="z_shock" sort={sort} align="right">z (σ)</SortTh>
          <SortTh colKey="intraday_rsi" sort={sort}>Intraday</SortTh>
          <SortTh colKey="hist_avg_return" sort={sort} align="right"><Term k="histedge">Short edge</Term></SortTh>
          <SortTh colKey="hist_win_rate" sort={sort} align="right"><Term k="winrate">Win%</Term></SortTh>
          <SortTh colKey="hist_trades" sort={sort} align="right">Trades</SortTh>
          <SortTh colKey="market_cap" sort={sort} align="right"><Term k="marketcap">Mkt cap</Term></SortTh>
          <SortTh colKey="forward_pe" sort={sort} align="right"><Term k="pe">Fwd P/E</Term></SortTh>
          <th><Term k="sector">Sector</Term></th>
          <th><Term k="smartmoney">Smart money</Term></th>
        </tr></thead>
        <tbody>
          {sort.rows.map(r => (
            <tr key={`${r.ticker}|${r.signal_key}`} className="study-row">
              <td><b>{r.ticker}</b></td>
              <td>{burstChip(r.burst_type)}</td>
              <td title={r.signal_key} className="dim">{r.signal_name}</td>
              <td style={{ textAlign: 'right' }}>{r.days_ago === 0 ? <span className="good">today</span> : `${r.days_ago}d`}</td>
              <td style={{ textAlign: 'right' }} className={signCls(r.day1_move)}>{r.day1_move == null ? '–' : `${r.day1_move > 0 ? '+' : ''}${r.day1_move}%`}</td>
              <td style={{ textAlign: 'right' }} className={signCls(r.z_shock)}>{r.z_shock == null ? '–' : r.z_shock.toFixed(1)}</td>
              <td style={{ fontSize: 11 }}>{r.intraday_signal ? <span className={`sm-badge ${/RSI↑/.test(r.intraday_signal) ? 'good' : 'dim'}`} title="Intraday entry timing">{r.intraday_signal}</span> : <span className="dim">–</span>}</td>
              <td style={{ textAlign: 'right' }} className={r.hist_avg_return == null ? 'dim' : signCls(r.hist_avg_return)}>{r.hist_avg_return == null ? '–' : `${r.hist_avg_return > 0 ? '+' : ''}${r.hist_avg_return}%`}</td>
              <td style={{ textAlign: 'right' }}>{r.hist_win_rate == null ? <span className="dim">–</span> : `${r.hist_win_rate}%`}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.hist_trades == null ? '–' : r.hist_trades}</td>
              <td style={{ textAlign: 'right' }} className="dim">{capFmt(r.market_cap)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.forward_pe != null ? r.forward_pe.toFixed(1) : '–'}</td>
              <td className="dim">{(r.sectors || []).slice(0, 2).join(', ')}</td>
              <td style={{ fontSize: 11 }}>{smChips(r)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---- Live hub: Global Confluence ---------------------------------------------
// A live short-term burst (required) confirmed by our other validated layers, scored 0-100.
function GlobalPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const rowsAll = (data && data.results) || [];
  const sort = useSortedRows(rowsAll, 'global_score', 'desc');

  const load = () => apiFetch('/global').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);

  const runScan = () => {
    setRunning(true);
    fetch(`${API}/global`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/global`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading global confluence...</div>;
  if (data.computed === false) {
    return (
      <div className="studies-page">
        <h1>Global Confluence</h1>
        <p className="subtitle">High-conviction confluence across our validated layers.</p>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.message || 'Not scanned yet.'}</p>
          <button className="refresh-btn" onClick={runScan} disabled={running}>{running ? 'Scanning...' : 'Run scan'}</button>
        </div>
      </div>
    );
  }

  const weights = data.weights || {};
  const capFmt = v => v == null ? '–' : v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(0)}M` : `$${(v / 1e3).toFixed(0)}K`;
  const signCls = v => v == null ? 'dim' : v > 0 ? 'good' : v < 0 ? 'bad' : 'dim';
  const scoreCls = s => s == null ? 'dim' : s >= 70 ? 'good' : s >= 40 ? '' : 'bad';
  const adCls = s => !s ? 'dim' : /accum/.test(s) ? 'good' : /distribution/.test(s) ? 'bad' : 'dim';
  const burstChip = t => t === 'reversal'
    ? <span className="sm-badge sm-13g" title="Reversal">reversal</span>
    : <span className="sm-badge sm-insider" title="Momentum">momentum</span>;
  // 9 component pills: B(urst) E(dge) A/D D(arkpool) S(mart money) F(undamentals) R(egime) N(ews) I(ntraday), shaded by 0..1.
  const COMP = [['burst', 'B'], ['edge', 'E'], ['ad', 'A'], ['darkpool', 'D'], ['smart_money', 'S'], ['fundamentals', 'F'], ['regime', 'R'], ['news', 'N', 'high = the burst is bouncing off a recent grounded news-overreaction (PODD-type)'], ['intraday', 'I', 'high = an oversold intraday RSI cross-up timing the reversal entry (8h/12h)']];
  const compPills = c => (
    <span style={{ whiteSpace: 'nowrap' }}>
      {COMP.map(([k, lbl, note]) => {
        const v = c && typeof c[k] === 'number' ? c[k] : 0;
        return (
          <span key={k} title={`${k} ${v.toFixed(2)}${note ? ' — ' + note : ''}`}
            style={{ display: 'inline-block', width: 15, textAlign: 'center', fontSize: 10, fontWeight: 600, marginRight: 2, borderRadius: 3, color: v >= 0.45 ? '#fff' : '#8b949e', background: `rgba(63,185,80,${(0.12 + 0.78 * v).toFixed(3)})` }}>
            {lbl}
          </span>
        );
      })}
    </span>
  );
  const smChips = r => {
    const fmt = v => v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : `$${(v / 1e3).toFixed(0)}K`;
    const p = [];
    if (r.insider_buy_90d > 0) p.push(<span key="i" className="sm-badge sm-insider" title="Insider open-market buys, ~trailing 6mo">▲ {fmt(r.insider_buy_90d)} insider</span>);
    if (r.recent_13d > 0) p.push(<span key="d" className="sm-badge sm-13d" title="Activist 13D filings, last 180d">13D×{r.recent_13d}</span>);
    if (r.recent_13g > 0) p.push(<span key="g" className="sm-badge sm-13g" title="Passive institutional 13G filings, last 180d">13G×{r.recent_13g}</span>);
    return p.length ? p : <span className="dim">–</span>;
  };

  return (
    <div className="studies-page">
      <h1>Global Confluence <span className="dim">({(data.n_global ?? rowsAll.length).toLocaleString()})</span>
        <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle">
        High-conviction confluence — a live short-term burst (required) confirmed by our other validated layers, scored 0-100.{' '}
        {data.regime_bull
          ? <span className="sm-badge sm-insider" title="SPY regime">SPY BULL</span>
          : <span className="sm-badge sm-13d" title="SPY regime" style={{ color: '#f85149', background: '#3d1416' }}>SPY BEAR</span>}
      </p>
      <p className="subtitle" style={{ marginTop: 0, fontSize: 11 }}>
        <span className="dim">Weights: </span>
        {COMP.map(([k], i) => (
          <span key={k} className="dim">{i ? ' · ' : ''}{k} {weights[k] != null ? weights[k] : '–'}</span>
        ))}
      </p>

      <div className="studies-controls">
        <div className="filters">
          <button className="refresh-btn" onClick={runScan} disabled={running}>{running ? 'Scanning...' : 'Re-scan'}</button>
        </div>
      </div>

      <table className="studies-table">
        <thead><tr>
          <SortTh colKey="global_score" sort={sort} align="right">Score</SortTh>
          <th>Components</th>
          <SortTh colKey="burst_signal_name" sort={sort}>Burst</SortTh>
          <SortTh colKey="hist_avg_return" sort={sort} align="right"><Term k="histedge">Edge</Term></SortTh>
          <SortTh colKey="ad_state" sort={sort}>A/D</SortTh>
          <SortTh colKey="darkpool_off_pct" sort={sort} align="right">Dark pool</SortTh>
          <th><Term k="smartmoney">Smart money</Term></th>
          <th>Fundamentals</th>
          <SortTh colKey="market_cap" sort={sort} align="right"><Term k="marketcap">Mkt cap</Term></SortTh>
          <th><Term k="sector">Sector</Term></th>
        </tr></thead>
        <tbody>
          {sort.rows.map(r => (
            <tr key={r.ticker} className="study-row">
              <td style={{ textAlign: 'right', minWidth: 66 }}>
                <b className={scoreCls(r.global_score)} style={{ fontSize: 14 }}>{r.global_score}</b>
                <div style={{ height: 4, marginTop: 3, borderRadius: 2, background: '#1f6feb', width: `${Math.max(0, Math.min(100, r.global_score || 0))}%`, maxWidth: '100%' }} />
              </td>
              <td>{compPills(r.components)}</td>
              <td>
                <b>{r.ticker}</b> {burstChip(r.burst_type)}
                {r.intraday_signal ? <span className={`sm-badge ${/RSI↑/.test(r.intraday_signal) ? 'good' : 'dim'}`} title="Intraday entry timing">{r.intraday_signal}</span> : null}
                <div className="dim" style={{ fontSize: 10 }} title={r.burst_signal_key}>{r.burst_signal_name} · {r.burst_days_ago === 0 ? 'today' : `${r.burst_days_ago}d`}</div>
              </td>
              <td style={{ textAlign: 'right' }} className={r.hist_avg_return == null ? 'dim' : signCls(r.hist_avg_return)}>{r.hist_avg_return == null ? '–' : `${r.hist_avg_return > 0 ? '+' : ''}${r.hist_avg_return}%`}</td>
              <td className={adCls(r.ad_state)} style={{ fontSize: 11 }}>{r.ad_state || <span className="dim">–</span>}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.darkpool_off_pct == null ? '–' : `${r.darkpool_off_pct}%`}{r.darkpool_rising ? <span className="good"> ▲</span> : ''}</td>
              <td style={{ fontSize: 11 }}>{smChips(r)}</td>
              <td className="dim" style={{ fontSize: 11 }}>{(() => {
                const vals = Object.values(r.fund_buckets || {}).filter(Boolean).slice(0, 2);
                return vals.length ? vals.join(' · ') : '–';
              })()}</td>
              <td style={{ textAlign: 'right' }} className="dim">{capFmt(r.market_cap)}</td>
              <td className="dim">{(r.sectors || []).slice(0, 2).join(', ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Small helper: a tab bar backed by URL hash. `tabs` = [{key,label,hash,el}].
function HubTabs({ tabs, initKey }) {
  const pick = () => {
    const h = window.location.pathname;
    for (const t of tabs) if (t.match && t.match.some(m => h.includes(m))) return t.key;
    return initKey || tabs[0].key;
  };
  const [tab, setTab] = useState(pick);
  const active = tabs.find(t => t.key === tab) || tabs[0];
  return (
    <div>
      <div className="studies-tabs">
        {tabs.map(t => (
          <button key={t.key} className={tab === t.key ? 'active' : ''}
                  onClick={() => { setTab(t.key); navigate(t.hash); }}>{t.label}</button>
        ))}
      </div>
      {active.el}
    </div>
  );
}

function LiveSignalsHub() {
  const { lab } = useSectionBacktest();
  const p3 = lab?.phase3;
  return (
    <div>
      <BacktestPanel
        title="Top-signals portfolio vs SPY"
        subtitle="Combined portfolio of the top-ranked signals — the strategy behind the live playbook — backtested vs SPY buy-hold."
        curve={p3?.curve} summary={p3?.summary} />
      <HubTabs initKey="playbook" tabs={[
        { key: 'playbook', label: '▶ Playbook', hash: '/live/playbook', match: ['playbook'], el: <PlaybookPage /> },
        { key: 'firing', label: '⚡ Firing Now', hash: '/live/firing', match: ['firing'], el: <FiringNowPage /> },
        { key: 'shortterm', label: 'Short-term', hash: '/live/shortterm', match: ['shortterm'], el: <ShortTermPage /> },
        { key: 'global', label: 'Global', hash: '/live/global', match: ['global'], el: <GlobalPage /> },
        { key: 'addiv', label: 'A/D Divergence', hash: '/live/addiv', match: ['addiv'], el: <AdDivergencePage /> },
      ]} />
    </div>
  );
}

function NewsEventStudyPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const load = () => fetch(`${API}/news-event-study`).then(r => r.json()).then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);

  const run = () => {
    setRunning(true);
    fetch(`${API}/news-event-study`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/news-event-study`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="error">Error: {err}</div>;
  if (!data) return <div className="loading">Loading news event study...</div>;
  if (!data.computed) {
    return (
      <div className="studies-page">
        <h1>News Event Study</h1>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Running...' : 'Run study'}</button>
        </div>
      </div>
    );
  }

  const g = data.groupings || {};
  const pct = v => v == null ? '–' : `${v > 0 ? '+' : ''}${v.toFixed(2)}%`;
  const num = v => v == null ? '–' : v.toFixed(2);
  const cls = v => v == null ? 'dim' : v > 0 ? 'good' : v < 0 ? 'bad' : 'dim';
  const sigCls = v => (v == null ? 'dim' : Math.abs(v) >= 2 ? 'good' : 'dim');   // |t|>=2 ≈ stands out
  const RET_COLS = [
    { key: 'avg_ar_d0', label: 'AR d0' },
    { key: 't_ar0', label: 'd0 t', sig: true },
    { key: 'avg_gap1', label: 'gap +1 (AM)' },
    { key: 'avg_car5', label: 'CAR-5' },
    { key: 'avg_car20', label: 'CAR-20' },
    { key: 'n_car20', label: 'n₂₀', raw: true, int: true },
    { key: 't_car20', label: 'CAR-20 t', sig: true },
    { key: 'avg_iv', label: 'avg IV', raw: true },
    { key: 'avg_surprise', label: 'surprise σ', raw: true },
  ];
  const cell = (r, c) => c.sig
    ? <td key={c.key} className={sigCls(r[c.key])}>{r[c.key] == null ? '–' : r[c.key].toFixed(1)}</td>
    : c.int
    ? <td key={c.key} className="dim">{r[c.key] == null ? '–' : r[c.key]}</td>
    : <td key={c.key} className={c.raw ? '' : cls(r[c.key])}>{r[c.key] == null ? '–' : (c.raw ? num(r[c.key]) : pct(r[c.key]))}</td>;

  const Table = ({ title, note, rows, k0, k1 }) => {
    const tSort = useSortedRows(rows, null, 'desc');
    return (
    <div style={{ marginBottom: 26 }}>
      <h3 style={{ marginBottom: 2 }}>{title}</h3>
      {note && <p className="subtitle" style={{ marginTop: 0 }}>{note}</p>}
      <div style={{ overflowX: 'auto' }}>
        <table className="studies-table">
          <thead><tr>
            <SortTh colKey={k0[0]} sort={tSort}>{k0[1]}</SortTh><SortTh colKey={k1[0]} sort={tSort}>{k1[1]}</SortTh><SortTh colKey="n" sort={tSort}>n</SortTh>
            {RET_COLS.map(c => <SortTh key={c.key} colKey={c.key} sort={tSort}>{c.label}</SortTh>)}
          </tr></thead>
          <tbody>
            {tSort.rows.map((r, i) => (
              <tr key={i}>
                <td>{r[k0[0]]}</td><td>{r[k1[0]]}</td><td>{r.n}</td>
                {RET_COLS.map(c => cell(r, c))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
    );
  };

  return (
    <div className="studies-page">
      <h1>News Event Study <span className="dim">({data.n_events} events / {data.n_tickers} tickers)</span>
        <LastUpdatedChip value={data.computed_at} /></h1>
      <p className="subtitle">
        Market-adjusted news reaction, keyed on <b>our model's read</b> (direction/impact), not EODHD sentiment.
        <b> AR = stock return − β·SPY</b> (news effect, market stripped). <b>AR d0</b> = event-day reaction;
        <b> gap +1</b> = next-morning (pre-market) abnormal gap; <b>CAR-5/20</b> = abnormal drift over the next 5/20
        days (excludes day 0); <b>surprise σ</b> = day-0 move in IV-implied sigmas.
        {data.computed_at && <span className="dim"> · {new Date(data.computed_at).toLocaleString()}</span>}
        <button className="refresh-btn" style={{ marginLeft: 10 }} onClick={run} disabled={running}>{running ? 'Running...' : 'Re-run'}</button>
      </p>
      <div className="dim" style={{ border: '1px solid #7a5533', borderRadius: 6, padding: '8px 12px', margin: '0 0 18px', fontSize: 12, lineHeight: 1.5 }}>
        <b>Caveats:</b> Universe = <b>today's listed names</b> → survivorship bias (delisted losers missing) makes returns optimistic. ·{' '}
        <b>gap +1</b> is the overnight slice of day +1 — it's <i>inside</i> CAR-5/20, not additive. ·{' '}
        <b>surprise</b> = raw move ÷ IV-implied 1σ (matches IV Calibration). ·{' '}
        CAR requires a <b>full</b> forward window (recent events lacking 20 days ahead are excluded — see <b>n₂₀</b>, the true CAR-20 sample, which is smaller than n). ·{' '}
        <b>t-columns</b> (d0 t, CAR-20 t) are one-sample t vs 0; <b>|t| ≥ 2</b> (highlighted) is a real move, small-|t| cells are noise. Same-ticker windows overlap, so t slightly overstates significance — treat as a filter, not proof. ·{' '}
        Sample skews recent (2025–26), on-ticker classified news only; small beta/IV cells (n≈30–130) are noisy.
      </div>

      <div style={{ marginBottom: 26 }}>
        <h3 style={{ marginBottom: 2 }}>Overall — average across ALL news events (all stocks together)</h3>
        <p className="subtitle" style={{ marginTop: 0 }}>
          The market-adjusted move of a typical classified on-ticker news event, blended across every stock, then split by
          our model's direction. The <b>ALL events</b> row nets up- and down-news together (so it trends toward the up-news
          majority); the per-direction rows are the cleaner read.
        </p>
        <div style={{ overflowX: 'auto' }}>
          <table className="studies-table">
            <thead><tr><th>scope</th><th>n</th>{RET_COLS.map(c => <th key={c.key}>{c.label}</th>)}</tr></thead>
            <tbody>
              {[{ label: 'ALL events', ...(g.overall || {}) }, ...(g.by_direction || []).map(d => ({ label: d.model_dir, ...d }))].map((r, i) => (
                <tr key={i} style={i === 0 ? { fontWeight: 700 } : {}}>
                  <td>{r.label}</td><td>{r.n == null ? '–' : r.n}</td>
                  {RET_COLS.map(c => cell(r, c))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <Table title="Model direction × impact" note="Does our model's call + conviction predict the market-adjusted move? (down news is the bigger surprise)"
             rows={g.by_model_dir_impact} k0={['model_dir', 'dir']} k1={['impact', 'impact']} />
      <Table title="Model direction × beta (252-day)" note="Under-reaction (good news on low-beta drifts up over 20d) vs over-reaction (high-beta pops then fades)."
             rows={g.by_model_dir_beta} k0={['model_dir', 'dir']} k1={['beta_bucket', 'beta 252d']} />
      <Table title="Model direction × beta (60-day)" note="Short-window beta — catches names whose character shifted recently."
             rows={g.by_model_dir_beta60} k0={['model_dir', 'dir']} k1={['beta60_bucket', 'beta 60d']} />
      <Table title="IV regime × direction" note="High-IV names over-react on good news and bleed over 20d on flat/uncertain news."
             rows={g.by_iv_dir} k0={['iv_bucket', 'IV regime']} k1={['model_dir', 'dir']} />
      <Table title="News type × beta" note="By event category (earnings / guidance / analyst / product / …)."
             rows={g.by_news_type_beta} k0={['news_type', 'type']} k1={['beta_bucket', 'beta 252d']} />
    </div>
  );
}

function IvCalibrationPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);
  const [order, setOrder] = useState('over');
  const [q, setQ] = useState('');
  const ivSort = useSortedRows(data && data.per_ticker, null, 'desc');

  const load = () => fetch(`${API}/iv-calibration`).then(r => r.json()).then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);
  const run = () => {
    setRunning(true);
    fetch(`${API}/iv-calibration`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/iv-calibration`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="error">Error: {err}</div>;
  if (!data) return <div className="loading">Loading IV calibration...</div>;
  if (!data.computed) {
    return (
      <div className="studies-page"><h1>IV Calibration</h1>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Running...' : 'Run study'}</button>
        </div>
      </div>
    );
  }

  const a = data.aggregate || {};
  // Header-click sort (ivSort) takes precedence when active; otherwise use the over/under preset.
  let rows = ivSort.rows.filter(r => !q || r.ticker.toLowerCase().includes(q.toLowerCase()));
  if (!ivSort.sortKey) rows = [...rows].sort((x, y) => order === 'over' ? x.median_ratio - y.median_ratio : y.median_ratio - x.median_ratio);
  rows = rows.slice(0, 60);
  const ratioCls = v => v < 0.5 ? 'bad' : v > 0.8 ? 'good' : 'dim';
  const cards = [
    ['Median ratio', a.median_ratio, 'fair ~0.67'],
    ['% exceed 1σ', a.pct_1sig != null ? a.pct_1sig + '%' : '–', 'fair ~32%'],
    ['% exceed 2σ', a.pct_2sig != null ? a.pct_2sig + '%' : '–', 'fair ~5%'],
    ['% < half implied', a.pct_half != null ? a.pct_half + '%' : '–', 'IV too hot'],
  ];

  return (
    <div className="studies-page">
      <h1>IV Calibration <span className="dim">({a.n_days ? a.n_days.toLocaleString() : 0} stock-days / {data.n_tickers} tickers)</span>
        <LastUpdatedChip value={data.computed_at} /></h1>
      <p className="subtitle">
        Is ATM implied vol a good predictor of the next-day move? Ratio = <b>actual move ÷ IV-implied 1σ daily move</b> (atm_iv/√252).
        A calibrated normal has median ~0.67 and clears 1σ ~32% of days.
        {data.computed_at && <span className="dim"> · {new Date(data.computed_at).toLocaleString()}</span>}
        <button className="refresh-btn" style={{ marginLeft: 10 }} onClick={run} disabled={running}>{running ? 'Running...' : 'Re-run'}</button>
      </p>

      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', margin: '12px 0 16px' }}>
        {cards.map(([label, val, ref]) => (
          <div key={label} style={{ border: '1px solid #333', borderRadius: 8, padding: '10px 14px', minWidth: 120 }}>
            <div className="dim" style={{ fontSize: 11 }}>{label}</div>
            <div style={{ fontSize: 20, fontWeight: 600 }}>{val == null ? '–' : val}</div>
            <div className="dim" style={{ fontSize: 10 }}>{ref}</div>
          </div>
        ))}
      </div>
      <p className="subtitle" style={{ marginTop: 0 }}>
        <b>Take:</b> IV runs hot — the typical day moves ~half of what's implied and clears 1σ far less than a fair market would;
        the 2σ tails are near-calibrated. Per-ticker below: <span className="bad">low ratio</span> = options systematically
        over-priced (moves fizzle); <span className="good">high ratio</span> = moves blow past IV (event-prone / IV too cheap).
        <br /><span style={{ fontSize: 11 }}>Caveats: atm_iv is ~30-day ATM annualized; ÷√252 assumes a flat term structure (standard proxy). Universe = today's listed names (survivorship).</span>
      </p>

      <div className="studies-controls">
        <div className="filters">
          <button className={order === 'over' ? 'active' : ''} onClick={() => setOrder('over')}>Most over-priced</button>
          <button className={order === 'under' ? 'active' : ''} onClick={() => setOrder('under')}>Moves blow past IV</button>
        </div>
        <input className="studies-search" placeholder="ticker…" value={q} onChange={e => setQ(e.target.value)} />
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="studies-table">
          <thead><tr><SortTh colKey="ticker" sort={ivSort}>Ticker</SortTh><SortTh colKey="n" sort={ivSort}>days</SortTh><SortTh colKey="median_ratio" sort={ivSort}>median ratio</SortTh><SortTh colKey="pct_1sig" sort={ivSort}>% exceed 1σ</SortTh><SortTh colKey="avg_iv" sort={ivSort}>avg IV</SortTh></tr></thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.ticker}>
                <td>{r.ticker}</td><td>{r.n}</td>
                <td className={ratioCls(r.median_ratio)}>{r.median_ratio.toFixed(2)}</td>
                <td>{r.pct_1sig}%</td><td>{r.avg_iv}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function NewsHub() {
  const { decomp } = useSectionBacktest();
  const vxt = decomp?.value_x_technical;
  // Prefer a news-catalyst signal combined with the cheapest-P/B pick; else the P/B pick alone.
  const sig = (vxt?.signals || []).find(sg => sg?.signal_plus_lowpb?.curve);
  const combo = sig?.signal_plus_lowpb;
  const src = combo || vxt?.lowpb_alone;
  return (
    <div>
      <BacktestPanel
        title={combo ? 'News-catalyst signal + cheapest-P/B vs SPY' : 'Cheapest-P/B pick vs SPY'}
        subtitle={combo
          ? 'Among stocks firing a technical/news catalyst each month, buy the single cheapest-P/B one — backtested vs SPY buy-hold.'
          : 'No signal+P/B curve available; showing the cheapest-P/B pick alone vs SPY buy-hold.'}
        curve={src?.curve} summary={src?.summary} />
      <HubTabs initKey="newsfx" tabs={[
        { key: 'newsfx', label: 'News Effect', hash: '/news/effect', match: ['effect', 'newsfx'], el: <NewsEffectPage /> },
        { key: 'newscl', label: 'Clusters', hash: '/news/clusters', match: ['clusters', 'newscl'], el: <NewsClusterPage /> },
        { key: 'newshz', label: 'News Horizon', hash: '/news/horizon', match: ['horizon', 'newshz'], el: <NewsHorizonPage /> },
        { key: 'newsev', label: 'Event Study', hash: '/news/eventstudy', match: ['eventstudy', 'newsev'], el: <NewsEventStudyPage /> },
        { key: 'overreaction', label: 'Overreaction', hash: '/news/overreaction', match: ['overreaction'], el: <NewsOverreactionPage /> },
        { key: 'ivcal', label: 'IV Calibration', hash: '/news/ivcal', match: ['ivcal'], el: <IvCalibrationPage /> },
      ]} />
    </div>
  );
}

function ResearchHub() {
  const { lab } = useSectionBacktest();
  // Best rotation rule by total return.
  const best = (lab?.phase1?.strategies || []).reduce((b, s) =>
    (!b || (s.summary?.total_return ?? -Infinity) > (b.summary?.total_return ?? -Infinity)) ? s : b, null);
  return (
    <div>
      <BacktestPanel
        title="Best rotation rule vs SPY"
        subtitle={`Top-performing rotation rule by total return${best?.label ? ` — ${best.label}` : ''} — backtested vs SPY buy-hold.`}
        curve={best?.curve} summary={best?.summary} />
      <HubTabs initKey="sectors" tabs={[
        { key: 'sectors', label: 'Sectors', hash: '/research/sectors', match: ['sectors'], el: <StudiesPage /> },
        { key: 'stocks', label: 'Stock Indicator Studies', hash: '/research/stocks', match: ['stocks'], el: <StockStudiesPage /> },
        { key: 'fundamentals', label: 'Fundamentals', hash: '/research/fundamentals', match: ['fundamentals'], el: <FundamentalsPage /> },
        { key: 'trends', label: 'Trend Studies', hash: '/research/trends', match: ['trends', 'trend'], el: <TrendStudiesPage /> },
        { key: 'backtest', label: 'Backtest', hash: '/research/backtest', match: ['backtest'], el: <BacktestPage /> },
        { key: 'lab', label: 'Research/Lab', hash: '/research/lab', match: ['lab'], el: <ResearchPage /> },
        { key: 'intersect', label: 'Intersections', hash: '/research/intersect', match: ['intersect'], el: <IntersectionPage /> },
        { key: 'rsiintraday', label: 'RSI Intraday', hash: '/research/rsiintraday', match: ['rsiintraday'], el: <RsiIntradayPage /> },
      ]} />
    </div>
  );
}

// ── Alt-data validation (exploratory) ────────────────────────────────────────
// Two OBSERVATIONAL studies surfaced side-by-side. NOT wired into signals/risk-rating.
function AltDataPage() {
  const [congress, setCongress] = useState(null);
  const [delisted, setDelisted] = useState(null);
  const [cErr, setCErr] = useState(null);
  const [dErr, setDErr] = useState(null);
  const [cRunning, setCRunning] = useState(false);
  const [dRunning, setDRunning] = useState(false);
  const [horizon, setHorizon] = useState(null);
  const [congressBt, setCongressBt] = useState(null);

  const loadCongress = () => apiFetch('/congress-study').then(setCongress).catch(e => setCErr(e.message));
  const loadDelisted = () => apiFetch('/delisted-survivorship').then(setDelisted).catch(e => setDErr(e.message));
  useEffect(() => {
    loadCongress();
    loadDelisted();
    apiFetch('/congress-backtest').then(setCongressBt).catch(() => {});
  }, []);

  // Poll after a POST until the study reports computed:true.
  const runCongress = () => {
    setCRunning(true);
    fetch(`${API}/congress-study`, { method: 'POST' }).then(() => {
      const t = setInterval(() => apiFetch('/congress-study').then(d => {
        if (d.computed) { clearInterval(t); setCRunning(false); setCongress(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setCRunning(false));
  };
  const runDelisted = () => {
    setDRunning(true);
    fetch(`${API}/delisted-survivorship`, { method: 'POST' }).then(() => {
      const t = setInterval(() => apiFetch('/delisted-survivorship').then(d => {
        if (d.computed) { clearInterval(t); setDRunning(false); setDelisted(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setDRunning(false));
  };

  const arCls = (v) => (v == null ? 'dim' : v > 0 ? 'good' : 'bad');
  const tCls = (v) => (v != null && Math.abs(v) >= 2 ? 'good' : 'dim');
  const fmtAr = (v) => (v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(2)}%`);
  const fmtT = (v) => (v == null ? '–' : Number(v).toFixed(2));
  const fmtWr = (v) => (v == null ? '–' : `${Number(v).toFixed(1)}%`);
  const fmtN = (v) => (v == null ? '–' : Number(v).toLocaleString());
  const updated = (d) => d && (d.last_updated || d.computed_at);

  // Render a {label: {n,avg_ar,t_stat,win_rate}} slice as a compact sub-table.
  const Slice = ({ title, obj, limit }) => {
    const entries = Object.entries(obj || {}).sort((a, b) => (b[1]?.n ?? 0) - (a[1]?.n ?? 0));
    const baseRows = limit ? entries.slice(0, limit) : entries;
    const sliceSort = useSortedRows(baseRows, null, 'desc', {
      label: r => r[0],
      n: r => r[1]?.n,
      avg_ar: r => r[1]?.avg_ar,
      win_rate: r => r[1]?.win_rate,
      t_stat: r => r[1]?.t_stat,
    });
    if (!baseRows.length) return null;
    return (
      <div style={{ minWidth: 260, flex: '1 1 300px' }}>
        <h4 style={{ margin: '4px 0 6px' }}>{title}</h4>
        <table className="studies-table">
          <thead><tr>
            <SortTh colKey="label" sort={sliceSort} align="left">Bucket</SortTh>
            <SortTh colKey="n" sort={sliceSort} align="right">n</SortTh>
            <SortTh colKey="avg_ar" sort={sliceSort} align="right">AR</SortTh>
            <SortTh colKey="win_rate" sort={sliceSort} align="right">Win%</SortTh>
            <SortTh colKey="t_stat" sort={sliceSort} align="right">t</SortTh>
          </tr></thead>
          <tbody>
            {sliceSort.rows.map(([label, s]) => (
              <tr key={label}>
                <td>{label}</td>
                <td style={{ textAlign: 'right' }}>{fmtN(s.n)}</td>
                <td style={{ textAlign: 'right' }} className={arCls(s.avg_ar)}>{fmtAr(s.avg_ar)}</td>
                <td style={{ textAlign: 'right' }} className="dim">{fmtWr(s.win_rate)}</td>
                <td style={{ textAlign: 'right' }} className={tCls(s.t_stat)}>{fmtT(s.t_stat)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Chosen horizon for the slice breakdowns (default to first available).
  const horizons = (congress && congress.horizons) || [];
  const hSel = horizon != null ? horizon : (horizons[0] != null ? horizons[0] : null);
  const hData = congress && congress.by_horizon ? congress.by_horizon[String(hSel)] : null;

  // Top-N of a plain {label: count} breakdown.
  const topEntries = (obj, n) => Object.entries(obj || {}).sort((a, b) => b[1] - a[1]).slice(0, n);

  return (
    <div className="altdata-page">
      <h1>&#127963; Alt-Data</h1>
      <p className="subtitle">Alt-data validation — exploratory, not wired into signals/risk-rating.</p>

      {/* ── Card 1: Congressional trades ─────────────────────────────── */}
      <div className="altdata-card">
        <div className="altdata-card-head">
          <h2>Congressional Trades</h2>
          <LastUpdatedChip value={updated(congress)} />
        </div>
        {cErr && <ErrorBanner message={cErr} onRetry={() => { setCErr(null); loadCongress(); }} onDismiss={() => setCErr(null)} />}
        {!congress && !cErr && <div className="loading">Loading congressional-trade study...</div>}
        {congress && congress.computed === false && (
          <div className="empty-state" style={{ padding: '24px 0' }}>
            <p>{congress.message || 'Not computed yet.'}</p>
            <button className="refresh-btn" onClick={runCongress} disabled={cRunning}>{cRunning ? 'Running…' : 'Run study'}</button>
          </div>
        )}
        {congress && congress.computed !== false && congress.by_horizon && (
          <>
            <p className="subtitle" style={{ marginTop: 4 }}>
              Point-in-time on the public <b>disclosure (report) date</b>; market-adjusted vs {congress.benchmark || 'SPY'}
              {' '}(AR = stock return − benchmark return). <b>OBSERVATIONAL — not a tradeable backtest and NOT wired to risk-rating.</b>
            </p>
            <div className="altdata-meta">
              {congress.universe_size != null && <span>Universe: <b>{fmtN(congress.universe_size)}</b></span>}
              {congress.trades_used != null && <span>Disclosures used: <b>{fmtN(congress.trades_used)}</b></span>}
            </div>

            <h4 style={{ margin: '10px 0 4px' }}>Buy vs sell, market-adjusted, by holding horizon</h4>
            <table className="studies-table">
              <thead><tr>
                <th style={{ textAlign: 'left' }}>Horizon</th>
                <th style={{ textAlign: 'left' }}>Side</th>
                <th style={{ textAlign: 'right' }}>n</th>
                <th style={{ textAlign: 'right' }}>AR</th>
                <th style={{ textAlign: 'right' }}>Win%</th>
                <th style={{ textAlign: 'right' }}>t</th>
              </tr></thead>
              <tbody>
                {horizons.map(h => {
                  const s = congress.by_horizon[String(h)] || {};
                  return ['buy', 'sell', 'overall'].map((side, i) => {
                    const v = s[side];
                    if (!v) return null;
                    return (
                      <tr key={`${h}-${side}`}>
                        <td>{i === 0 ? <b>{h}d</b> : ''}</td>
                        <td className="dim">{side}</td>
                        <td style={{ textAlign: 'right' }}>{fmtN(v.n)}</td>
                        <td style={{ textAlign: 'right' }} className={arCls(v.avg_ar)}>{fmtAr(v.avg_ar)}</td>
                        <td style={{ textAlign: 'right' }} className="dim">{fmtWr(v.win_rate)}</td>
                        <td style={{ textAlign: 'right' }} className={tCls(v.t_stat)}>{fmtT(v.t_stat)}</td>
                      </tr>
                    );
                  });
                })}
              </tbody>
            </table>

            {horizons.length > 1 && (
              <div className="altdata-meta" style={{ marginTop: 10 }}>
                <span>Slice breakdowns for horizon:</span>
                {horizons.map(h => (
                  <button key={h} className={`altdata-hbtn ${String(hSel) === String(h) ? 'active' : ''}`} onClick={() => setHorizon(h)}>{h}d</button>
                ))}
              </div>
            )}
            {hData && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, marginTop: 8 }}>
                <Slice title="Buy · by chamber" obj={hData.buy_by_chamber} />
                <Slice title="Sell · by chamber" obj={hData.sell_by_chamber} />
                <Slice title="Buy · by amount bucket" obj={hData.buy_by_amount_bucket} />
                <Slice title="Buy · by transaction type" obj={hData.buy_by_transaction_type} />
                <Slice title="Buy · top members" obj={hData.buy_by_member_top15} limit={10} />
              </div>
            )}
            <p className="subtitle" style={{ marginTop: 8, fontStyle: 'italic' }}>
              AR green when positive; t green when |t| ≥ 2. {congress.note}
            </p>
          </>
        )}

        {/* ── Follow-the-trades backtest ─────────────────────────────── */}
        {congressBt && congressBt.computed !== false && congressBt.strategies && (() => {
          const pretty = { follow_all_buys: 'Follow all buys', senate_buys_only: 'Senate buys only', large_buys_50k: 'Large buys (≥$50k)', follow_sells: 'Follow sells' };
          const strats = Object.entries(congressBt.strategies)
            .filter(([, v]) => v && v.curve && v.curve.length > 1)
            .sort((a, b) => ((b[1].summary || {}).total_return ?? -Infinity) - ((a[1].summary || {}).total_return ?? -Infinity));
          if (!strats.length) return null;
          return (
            <div style={{ marginTop: 14 }}>
              <h4 style={{ margin: '10px 0 4px' }}>Follow-the-trades backtest (equity curve vs SPY)</h4>
              {strats.map(([name, v], i) => (
                <BacktestPanel key={name} title={pretty[name] || name} curve={v.curve} summary={v.summary} collapsible={i !== 0} />
              ))}
              <p className="subtitle" style={{ marginTop: 8, fontStyle: 'italic' }}>{congressBt.note}</p>
            </div>
          );
        })()}
      </div>

      {/* ── Card 2: Delisted-survivorship audit ──────────────────────── */}
      <div className="altdata-card">
        <div className="altdata-card-head">
          <h2>Delisted Survivorship Audit</h2>
          <LastUpdatedChip value={updated(delisted)} />
        </div>
        {dErr && <ErrorBanner message={dErr} onRetry={() => { setDErr(null); loadDelisted(); }} onDismiss={() => setDErr(null)} />}
        {!delisted && !dErr && <div className="loading">Loading survivorship audit...</div>}
        {delisted && delisted.computed === false && (
          <div className="empty-state" style={{ padding: '24px 0' }}>
            <p>{delisted.message || 'Not computed yet.'}</p>
            <button className="refresh-btn" onClick={runDelisted} disabled={dRunning}>{dRunning ? 'Running…' : 'Run audit'}</button>
          </div>
        )}
        {delisted && delisted.computed !== false && delisted.bias_magnitude && (
          <>
            <p className="subtitle" style={{ marginTop: 4 }}>
              How much of the historical opportunity set our survivor-only universe never sees — a rough upper bound on upward bias.
            </p>
            <div className="altdata-statgrid">
              <div className="altdata-stat">
                <div className="altdata-stat-v">{fmtN(delisted.universe?.universe_size)}</div>
                <div className="altdata-stat-l">Current study universe (survivors)</div>
              </div>
              <div className="altdata-stat">
                <div className="altdata-stat-v">{fmtN(delisted.delisted?.common_stock_count)}</div>
                <div className="altdata-stat-l">Delisted common stocks</div>
              </div>
              <div className="altdata-stat">
                <div className="altdata-stat-v">{fmtN(delisted.delisted?.total_rows)}</div>
                <div className="altdata-stat-l">Total delisted rows (all types)</div>
              </div>
              <div className="altdata-stat">
                <div className="altdata-stat-v bad">{delisted.bias_magnitude?.ratio_delisted_common_over_universe_plus_delisted_common != null
                  ? `${(delisted.bias_magnitude.ratio_delisted_common_over_universe_plus_delisted_common * 100).toFixed(1)}%` : '–'}</div>
                <div className="altdata-stat-l">Bias magnitude — share of dead common names never tracked</div>
              </div>
            </div>
            {delisted.bias_magnitude?.interpretation && (
              <p className="subtitle" style={{ fontStyle: 'italic' }}>{delisted.bias_magnitude.interpretation}</p>
            )}

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, marginTop: 8 }}>
              <div style={{ minWidth: 220, flex: '1 1 240px' }}>
                <h4 style={{ margin: '4px 0 6px' }}>Delisted by security type</h4>
                <table className="studies-table">
                  <thead><tr><th style={{ textAlign: 'left' }}>Type</th><th style={{ textAlign: 'right' }}>Count</th></tr></thead>
                  <tbody>
                    {topEntries(delisted.breakdown_by_type, 10).map(([k, v]) => (
                      <tr key={k}><td>{k}</td><td style={{ textAlign: 'right' }}>{fmtN(v)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ minWidth: 220, flex: '1 1 240px' }}>
                <h4 style={{ margin: '4px 0 6px' }}>Delisted by exchange (top 10)</h4>
                <table className="studies-table">
                  <thead><tr><th style={{ textAlign: 'left' }}>Exchange</th><th style={{ textAlign: 'right' }}>Count</th></tr></thead>
                  <tbody>
                    {topEntries(delisted.breakdown_by_exchange, 10).map(([k, v]) => (
                      <tr key={k}><td>{k}</td><td style={{ textAlign: 'right' }}>{fmtN(v)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {Array.isArray(delisted.implications) && delisted.implications.length > 0 && (
              <div style={{ marginTop: 10 }}>
                <h4 style={{ margin: '4px 0 6px' }}>Implications</h4>
                <ul className="altdata-implications">
                  {delisted.implications.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Dark Pool ─────────────────────────────────────────────────────────────
// Visualizes off-exchange / dark-pool activity. Two data cadences from the backend:
//   • Polygon "daily" = blended off-exchange proxy (ATS + internalizers), short recent series.
//   • FINRA  "weekly" = official ATS-only, long multi-year history but ~2-4wk publication lag.
// Endpoint: /dark-pool (market snapshot) and /dark-pool?ticker=X (per-ticker dual series).
function DarkPoolPage() {
  const [snap, setSnap] = useState(null);
  const [snapErr, setSnapErr] = useState(null);
  const [tickerInput, setTickerInput] = useState('CLF');
  const [detail, setDetail] = useState(null);
  const [detailErr, setDetailErr] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [bt, setBt] = useState(null);
  const snapSort = useSortedRows(snap && snap.snapshot, null, 'desc');

  const loadSnap = () => { setSnapErr(null); apiFetch('/dark-pool').then(setSnap).catch(e => setSnapErr(e.message)); };

  const loadTicker = (t) => {
    const sym = (t || tickerInput || '').trim().toUpperCase();
    if (!sym) return;
    setTickerInput(sym);
    setDetail(null);
    setDetailErr(null);
    setDetailLoading(true);
    apiFetch(`/dark-pool?ticker=${encodeURIComponent(sym)}`)
      .then(d => { setDetail(d); setDetailLoading(false); })
      .catch(e => { setDetailErr(e.message); setDetailLoading(false); });
  };

  // On mount: market snapshot + preload the default ticker overlay + historical backtest.
  useEffect(() => {
    loadSnap();
    loadTicker('CLF');
    apiFetch('/darkpool-backtest').then(setBt).catch(() => {});
    // eslint-disable-next-line
  }, []);

  const pct = (v) => (v == null ? '–' : `${(Number(v) * 100).toFixed(1)}%`);
  const fmtVol = (v) => {
    if (v == null) return '–';
    const n = Number(v);
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return String(n);
  };
  const fmtAr = (v) => (v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(2)}%`);
  const fmtWr = (v) => (v == null ? '–' : `${Number(v).toFixed(1)}%`);
  const fmtN = (v) => (v == null ? '–' : Number(v).toLocaleString());

  // Dual-cadence SVG: shared X = date span across BOTH series; Y = off_pct 0..maxY (auto-scaled
  // because FINRA ATS-only levels sit well below the blended Polygon proxy). Weekly = polyline
  // (long history), daily = circle markers (sparse, forward-only recent points).
  const Chart = ({ weekly, daily }) => {
    const W = 820, H = 340, padL = 44, padR = 16, padT = 16, padB = 34;
    const wk = (weekly || []).map(d => ({ t: new Date(d.week_start).getTime(), v: Number(d.off_pct) })).filter(p => !isNaN(p.t) && !isNaN(p.v));
    const dl = (daily || []).map(d => ({ t: new Date(d.date).getTime(), v: Number(d.off_pct) })).filter(p => !isNaN(p.t) && !isNaN(p.v));
    const all = wk.concat(dl);
    if (!all.length) return <div className="empty-state" style={{ padding: '20px 0' }}>No dark-pool series for this ticker.</div>;
    const tMin = Math.min(...all.map(p => p.t));
    const tMax = Math.max(...all.map(p => p.t));
    const tSpan = tMax - tMin || 1;
    const vMaxRaw = Math.max(...all.map(p => p.v), 0.01);
    const vMax = Math.min(1, vMaxRaw * 1.12);
    const x = (t) => padL + ((t - tMin) / tSpan) * (W - padL - padR);
    const y = (v) => padT + (1 - v / vMax) * (H - padT - padB);
    const wkPts = wk.sort((a, b) => a.t - b.t).map(p => `${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(' ');

    // Y gridlines (5) and year ticks along X.
    const yTicks = [0, 0.25, 0.5, 0.75, 1].map(f => f * vMax);
    const yearSet = [];
    const yr0 = new Date(tMin).getFullYear(), yr1 = new Date(tMax).getFullYear();
    for (let yr = yr0; yr <= yr1; yr++) yearSet.push(yr);

    return (
      <svg className="darkpool-chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label="Dark-pool off-exchange share over time">
        {yTicks.map((v, i) => (
          <g key={i}>
            <line className="darkpool-grid" x1={padL} y1={y(v)} x2={W - padR} y2={y(v)} />
            <text className="darkpool-axis" x={padL - 6} y={y(v) + 3} textAnchor="end">{(v * 100).toFixed(0)}%</text>
          </g>
        ))}
        {yearSet.map((yr) => {
          const t = new Date(`${yr}-01-01`).getTime();
          if (t < tMin || t > tMax) return null;
          return <text key={yr} className="darkpool-axis" x={x(t)} y={H - padB + 18} textAnchor="middle">{yr}</text>;
        })}
        {wkPts && <polyline className="darkpool-line-weekly" points={wkPts} fill="none" />}
        {dl.map((p, i) => <circle key={i} className="darkpool-dot-daily" cx={x(p.t)} cy={y(p.v)} r={3.5} />)}
      </svg>
    );
  };

  const amp = snap && snap.amplifier;

  return (
    <div className="darkpool-page">
      <h1>&#9899; Dark Pool</h1>
      <p className="subtitle">{(snap && snap.note) || 'Off-exchange / dark-pool activity. Polygon = daily blended off-exchange proxy (ATS + internalizers); FINRA = official weekly ATS only, ~2-4wk publication lag.'}</p>

      {snapErr && <ErrorBanner message={snapErr} onRetry={loadSnap} onDismiss={() => setSnapErr(null)} />}

      {/* ── Market snapshot ─────────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Market snapshot — highest off-exchange share</h2>
          <div className="darkpool-chips">
            {snap && snap.snapshot_date && <span className="last-updated-chip">Snapshot: {snap.snapshot_date}</span>}
            {snap && snap.finra_last_week && <span className="last-updated-chip">FINRA wk: {snap.finra_last_week}</span>}
          </div>
        </div>
        {!snap && !snapErr && <div className="loading">Loading dark-pool snapshot...</div>}
        {snap && Array.isArray(snap.snapshot) && (
          <table className="studies-table darkpool-snap-table">
            <thead><tr>
              <SortTh colKey="ticker" sort={snapSort} align="left">Ticker</SortTh>
              <SortTh colKey="off_pct" sort={snapSort}>Off-exchange %</SortTh>
              <SortTh colKey="total_vol" sort={snapSort} align="right">Total volume</SortTh>
              <SortTh colKey="date" sort={snapSort} align="right">Date</SortTh>
            </tr></thead>
            <tbody>
              {snapSort.rows.map((r) => {
                const p = Number(r.off_pct);
                const cls = p >= 0.6 ? 'bad' : p >= 0.45 ? 'good' : 'dim';
                return (
                  <tr key={r.ticker} className="darkpool-snap-row" onClick={() => loadTicker(r.ticker)} title={`Load ${r.ticker} overlay`}>
                    <td style={{ fontWeight: 600 }}>{r.ticker}</td>
                    <td>
                      <div className="darkpool-bar-wrap">
                        <div className={`darkpool-bar ${cls}`} style={{ width: `${Math.min(100, p * 100).toFixed(0)}%` }} />
                        <span className={`darkpool-bar-val ${cls}`}>{pct(p)}</span>
                      </div>
                    </td>
                    <td style={{ textAlign: 'right' }} className="dim">{fmtVol(r.total_vol)}</td>
                    <td style={{ textAlign: 'right' }} className="dim">{r.date}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* ── Per-ticker overlay ──────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Per-ticker overlay</h2>
          <div className="darkpool-loadbar">
            <input
              className="darkpool-input"
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value.toUpperCase())}
              onKeyDown={(e) => { if (e.key === 'Enter') loadTicker(); }}
              placeholder="Ticker (e.g. CLF)"
            />
            <button className="refresh-btn" onClick={() => loadTicker()} disabled={detailLoading}>{detailLoading ? 'Loading…' : 'Load'}</button>
          </div>
        </div>
        {detailErr && <ErrorBanner message={detailErr} onRetry={() => loadTicker()} onDismiss={() => setDetailErr(null)} />}
        {detailLoading && !detail && <div className="loading">Loading {tickerInput} dark-pool series...</div>}
        {detail && (
          <>
            <div className="darkpool-statgrid">
              <div className="darkpool-stat">
                <div className="darkpool-stat-v">{detail.daily_last ? pct(detail.daily_last.off_pct) : '–'}</div>
                <div className="darkpool-stat-l">Polygon daily (latest{detail.daily_last ? ` · ${detail.daily_last.date}` : ''})</div>
              </div>
              <div className="darkpool-stat">
                <div className="darkpool-stat-v">{detail.weekly_last ? pct(detail.weekly_last.off_pct) : '–'}</div>
                <div className="darkpool-stat-l">FINRA ATS weekly (latest{detail.weekly_last ? ` · wk ${detail.weekly_last.week_start}` : ''})</div>
              </div>
            </div>
            <div className="darkpool-legend">
              <span className="darkpool-legend-item"><span className="darkpool-swatch darkpool-swatch-weekly" /> FINRA ATS — weekly (official, lagged)</span>
              <span className="darkpool-legend-item"><span className="darkpool-swatch darkpool-swatch-daily" /> Polygon — daily blended proxy (recent)</span>
            </div>
            <Chart weekly={detail.weekly} daily={detail.daily} />
          </>
        )}
      </div>

      {/* ── Amplifier backtest ──────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Does high dark-pool share amplify edge?{amp ? ` (${amp.signal} → ${amp.exit})` : ''}</h2>
        </div>
        {!amp && (
          <p className="subtitle darkpool-muted">Amplifier backtest computing (stock-study sweep in progress) — check back after the nightly sweep.</p>
        )}
        {amp && (
          <table className="studies-table">
            <thead><tr>
              <th style={{ textAlign: 'left' }}>Dark-pool bucket</th>
              <th style={{ textAlign: 'right' }}>Avg return</th>
              <th style={{ textAlign: 'right' }}>Win %</th>
              <th style={{ textAlign: 'right' }}>Trades</th>
            </tr></thead>
            <tbody>
              <tr className="darkpool-base-row">
                <td style={{ fontWeight: 600 }}>All (base)</td>
                <td style={{ textAlign: 'right' }} className={amp.base_avg_return > 0 ? 'good' : 'bad'}>{fmtAr(amp.base_avg_return)}</td>
                <td style={{ textAlign: 'right' }} className="dim">{fmtWr(amp.base_win_rate)}</td>
                <td style={{ textAlign: 'right' }} className="dim">{fmtN(amp.base_trades)}</td>
              </tr>
              {Object.entries(amp.buckets || {}).map(([label, b]) => (
                <tr key={label}>
                  <td>{label}</td>
                  <td style={{ textAlign: 'right' }} className={b.avg_return > 0 ? 'good' : 'bad'}>{fmtAr(b.avg_return)}</td>
                  <td style={{ textAlign: 'right' }} className="dim">{fmtWr(b.win_rate)}</td>
                  <td style={{ textAlign: 'right' }} className="dim">{fmtN(b.total_trades)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* ── Historical dark-pool backtest ───────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Historical dark-pool backtest — equity curve vs SPY</h2>
          <LastUpdatedChip value={bt && (bt.last_updated || bt.computed_at)} />
        </div>
        {bt && bt.computed === false && (
          <p className="subtitle darkpool-muted">{bt.note || 'Dark-pool backtest not computed yet — check back after the nightly sweep.'}</p>
        )}
        {bt && bt.computed !== false && bt.strategies && (() => {
          const strats = Object.entries(bt.strategies)
            .filter(([, v]) => v && v.curve && v.curve.length > 1)
            .sort((a, b) => ((b[1].summary || {}).total_return ?? -Infinity) - ((a[1].summary || {}).total_return ?? -Infinity));
          if (!strats.length) return <p className="subtitle darkpool-muted">No dark-pool strategies to show yet.</p>;
          return (
            <>
              {strats.map(([name, v], i) => (
                <BacktestPanel key={name} title={name} curve={v.curve} summary={v.summary} collapsible={i !== 0} />
              ))}
              {bt.buckets && Object.keys(bt.buckets).length > 0 && (
                <table className="studies-table" style={{ marginTop: 12 }}>
                  <thead><tr>
                    <th style={{ textAlign: 'left' }}>Dark-pool bucket</th>
                    <th style={{ textAlign: 'right' }}>Avg next-month return</th>
                    <th style={{ textAlign: 'right' }}>n</th>
                  </tr></thead>
                  <tbody>
                    {Object.entries(bt.buckets).map(([label, b]) => (
                      <tr key={label}>
                        <td>{label}</td>
                        <td style={{ textAlign: 'right' }} className={b.avg_ret_pct > 0 ? 'good' : 'bad'}>{fmtAr(b.avg_ret_pct)}</td>
                        <td style={{ textAlign: 'right' }} className="dim">{fmtN(b.n)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {bt.note && <p className="subtitle" style={{ marginTop: 8, fontStyle: 'italic' }}>{bt.note}</p>}
            </>
          );
        })()}
      </div>
    </div>
  );
}

// Vol-normalized shock study: is a day whose move is large relative to the stock's own
// trailing volatility (z = return ÷ trailing-20d vol) followed by continuation or reversal?
// Reads GET /vol-shock-study; POST kicks a background recompute (poll until computed).
// One sortable backtest table for a single vol_shock signal (sig = { name, rows }).
// Extracted so useSortedRows (a hook) isn't called inside VolShockPage's .map loop.
// ── News Overreaction ────────────────────────────────────────────────────────
// "A material news event whose β-adjusted move CONTRADICTS its sentiment tends to revert —
// but only in a tail band." Leads with the crash-size TailStrip so the tail edge is the headline.
function NewsOverreactionPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  // by-category rows built up-front so the sortable-hook is unconditional (data may be null early).
  const rowAtH = (arr, H) => (Array.isArray(arr) ? arr.find(r => Number(r.H) === H) : null);
  const catRows = React.useMemo(() => {
    const bc = (data && data.deep_dive && data.deep_dive.by_category) || {};
    return Object.keys(bc).map(k => {
      const r5 = rowAtH(bc[k], 5), r20 = rowAtH(bc[k], 20);
      return { category: k, mean5: r5 ? r5.mean : null, mean20: r20 ? r20.mean : null, n: (r5 && r5.n != null) ? r5.n : (r20 ? r20.n : null) };
    });
  }, [data]);
  const catSort = useSortedRows(catRows, 'mean5', 'asc');

  const load = () => { setErr(null); apiFetch('/news-overreaction').then(setData).catch(e => setErr(e.message)); };
  useEffect(() => { load(); }, []);

  const run = () => {
    setRunning(true);
    apiFetch('/news-overreaction', { method: 'POST' }).then(() => {
      const t = setInterval(() => apiFetch('/news-overreaction').then(d => {
        if (d && d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(e => { setErr(e.message); setRunning(false); });
  };

  const sPct = (v) => (v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(2)}%`);
  const pctR = (v) => (v == null ? '–' : `${Number(v).toFixed(1)}%`);
  const fmtN = (v) => (v == null ? '–' : Number(v).toLocaleString());
  const signCls = (v) => (v == null ? 'dim' : v > 0 ? 'good' : 'bad');

  if (err) return (
    <div className="darkpool-page">
      <h1>News Overreaction</h1>
      <ErrorBanner message={err} onRetry={load} onDismiss={() => setErr(null)} />
    </div>
  );
  if (!data) return <div className="loading">Loading news-overreaction study...</div>;

  if (!data.computed) {
    return (
      <div className="darkpool-page">
        <h1>News Overreaction</h1>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.note || data.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Running…' : 'Run detector'}</button>
        </div>
      </div>
    );
  }

  const params = data.params || {};
  const dsize = data.downside_by_size || {};
  const deep = data.deep_dive || {};
  const gp = deep.gap_profile || {};
  const classes = data.classes || {};
  const HZ = params.horizons || [1, 3, 5, 10, 20];

  const DSIZE_ORDER = ['5-10%', '10-15%', '15-20%', '20-25%', '25%+'];
  const dsizeStrip = (H) => DSIZE_ORDER.map(k => {
    const r = rowAtH(dsize[k], H);
    return { label: k, value: r ? r.mean : null, n: r ? r.n : null };
  });

  // Classes × horizons compact reversion table (mean colored by sign).
  const classOrder = [
    ['over_dn', 'Overreaction ↓ (good news, crashed)'],
    ['conf_dn', 'Confirmation ↓ (bad news, fell)'],
    ['over_up', 'Overreaction ↑ (bad news, popped)'],
    ['baseline', 'Baseline (all events)'],
  ].filter(([k]) => Array.isArray(classes[k]) && classes[k].length);

  return (
    <div className="darkpool-page">
      <h1>News Overreaction <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle">{data.note || 'β-adjusted news-reaction reversion detector.'}</p>
      <p className="subtitle darkpool-muted">
        A material news event whose β-adjusted move <b>CONTRADICTS</b> its sentiment (good news that crashes / bad news
        that pops) tends to revert — but only in a tail band.
        {params.n_events != null ? ` ${fmtN(params.n_events)} events, threshold ${params.thr_pct != null ? params.thr_pct + '%' : '—'}.` : ''}
      </p>

      {/* ── LEAD: the tail ─────────────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Bounce by crash size (good news, 5d &amp; 10d)</h2>
        </div>
        <div style={{ display: 'flex', gap: 28, flexWrap: 'wrap' }}>
          <div style={{ minWidth: 240 }}>
            <div className="subtitle darkpool-muted" style={{ margin: '2px 0' }}>Forward mean at H=5</div>
            <TailStrip buckets={dsizeStrip(5)} />
          </div>
          <div style={{ minWidth: 240 }}>
            <div className="subtitle darkpool-muted" style={{ margin: '2px 0' }}>Forward mean at H=10</div>
            <TailStrip buckets={dsizeStrip(10)} />
          </div>
        </div>
        <p className="subtitle" style={{ marginTop: 10 }}>
          The edge concentrates at <b>10-15%</b>; <b>20-25% inverts</b> (keeps falling). Bigger is <b>not</b> better here —
          the bounce lives in a middle tail band, not at the extremes.
        </p>
      </div>

      {/* ── Gap profile ────────────────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Gap profile — {deep.band || '10-25% down on good news'}</h2>
        </div>
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', margin: '4px 0 8px' }}>
          {[
            ['Opened gap-down >2%', gp.pct_gap_down != null ? pctR(gp.pct_gap_down) : '–'],
            ['Avg overnight gap', sPct(gp.avg_overnight_gap)],
            ['Avg intraday open→close', sPct(gp.avg_intraday_open_to_close)],
            ['% recovered intraday', gp.pct_intraday_recovered != null ? pctR(gp.pct_intraday_recovered) : '–'],
            ['n', fmtN(gp.n)],
          ].map(([label, val]) => (
            <div key={label} style={{ border: '1px solid #333', borderRadius: 8, padding: '10px 14px', minWidth: 120 }}>
              <div className="dim" style={{ fontSize: 11 }}>{label}</div>
              <div style={{ fontSize: 20, fontWeight: 600 }}>{val}</div>
            </div>
          ))}
        </div>
        <p className="subtitle darkpool-muted">These are overnight / premarket earnings gaps — the crash is mostly gapped in before the open.</p>
      </div>

      {/* ── By category ────────────────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>By category — {deep.band || 'deep-dive band'}</h2>
        </div>
        <table className="studies-table">
          <thead><tr>
            <SortTh label="category" colKey="category" sort={catSort} align="left" />
            <SortTh label="5d mean" colKey="mean5" sort={catSort} align="right" />
            <SortTh label="20d mean" colKey="mean20" sort={catSort} align="right" />
            <SortTh label="n" colKey="n" sort={catSort} align="right" />
          </tr></thead>
          <tbody>
            {catSort.rows.map((r, i) => (
              <tr key={i}>
                <td>{CAT_LABELS[r.category] || r.category}</td>
                <td style={{ textAlign: 'right' }} className={signCls(r.mean5)}>{sPct(r.mean5)}</td>
                <td style={{ textAlign: 'right' }} className={signCls(r.mean20)}>{sPct(r.mean20)}</td>
                <td style={{ textAlign: 'right' }} className="dim">{fmtN(r.n)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── Classes × horizons ─────────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Forward reversion by class × horizon</h2>
        </div>
        <p className="subtitle darkpool-muted">Mean forward return (%) at each horizon. Overreaction classes are the reversion candidates.</p>
        <div style={{ overflowX: 'auto' }}>
          <table className="studies-table">
            <thead><tr>
              <th style={{ textAlign: 'left' }}>class</th>
              {HZ.map(h => <th key={h} style={{ textAlign: 'right' }}>{h}d</th>)}
              <th style={{ textAlign: 'right' }}>n</th>
            </tr></thead>
            <tbody>
              {classOrder.map(([k, label]) => {
                const arr = classes[k] || [];
                const nRow = rowAtH(arr, HZ[0]) || arr[0] || {};
                return (
                  <tr key={k}>
                    <td style={{ fontWeight: 600 }}>{label}</td>
                    {HZ.map(h => {
                      const r = rowAtH(arr, h);
                      return <td key={h} style={{ textAlign: 'right' }} className={signCls(r ? r.mean : null)}>{r ? sPct(r.mean) : '–'}</td>;
                    })}
                    <td style={{ textAlign: 'right' }} className="dim">{fmtN(nRow.n)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {data.note && <p className="subtitle" style={{ marginTop: 8, fontStyle: 'italic' }}>{data.note}</p>}
    </div>
  );
}

// ── RSI Intraday crossover ────────────────────────────────────────────────────
// RSI(14) crossing above its SMA(14) on an intraday timeframe. Averaged it's noise;
// bucketed by how oversold the cross was, the edge appears — led by the RSI-level TailStrip.
// Per-bucket sortable exit-ladder rendered by a child component (one useSortedRows per instance).
function RsiBucketTable({ bucket, rows }) {
  const sort = useSortedRows(rows || [], 'avg_pct', 'desc');
  const [open, setOpen] = useState(false);
  const fmtN = (v) => (v == null ? '–' : Number(v).toLocaleString());
  const sPct = (v) => (v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(2)}%`);
  const pctR = (v) => (v == null ? '–' : `${Number(v).toFixed(1)}%`);
  const num2 = (v) => (v == null ? '–' : Number(v).toFixed(2));
  const signCls = (v) => (v == null ? 'dim' : v > 0 ? 'good' : 'bad');
  return (
    <div style={{ marginBottom: 10 }}>
      <button className="refresh-btn" style={{ marginBottom: 4 }} onClick={() => setOpen(o => !o)}>
        {open ? '▾' : '▸'} RSI {bucket} <span className="dim">({(rows || []).length} exits)</span>
      </button>
      {open && (
        <table className="studies-table">
          <thead><tr>
            <SortTh label="exit" colKey="exit" sort={sort} align="left" />
            <SortTh label="name" colKey="name" sort={sort} align="left" />
            <SortTh label="trades" colKey="trades" sort={sort} align="right" />
            <SortTh label="avg %" colKey="avg_pct" sort={sort} align="right" />
            <SortTh label="win %" colKey="win_pct" sort={sort} align="right" />
            <SortTh label="t" colKey="t" sort={sort} align="right" />
          </tr></thead>
          <tbody>
            {sort.rows.map((r, i) => (
              <tr key={i}>
                <td style={{ fontWeight: 600 }}>{r.exit}</td>
                <td className="dim">{r.name}</td>
                <td style={{ textAlign: 'right' }} className="dim">{fmtN(r.trades)}</td>
                <td style={{ textAlign: 'right' }} className={signCls(r.avg_pct)}>{sPct(r.avg_pct)}</td>
                <td style={{ textAlign: 'right' }} className="dim">{pctR(r.win_pct)}</td>
                <td style={{ textAlign: 'right' }} className="dim">{num2(r.t)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// Plain sortable exit-ladder (all crossovers / daily benchmark).
function RsiLadderTable({ rows }) {
  const sort = useSortedRows(rows || [], 'avg_pct', 'desc');
  const fmtN = (v) => (v == null ? '–' : Number(v).toLocaleString());
  const sPct = (v) => (v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(2)}%`);
  const pctR = (v) => (v == null ? '–' : `${Number(v).toFixed(1)}%`);
  const num2 = (v) => (v == null ? '–' : Number(v).toFixed(2));
  const signCls = (v) => (v == null ? 'dim' : v > 0 ? 'good' : 'bad');
  return (
    <table className="studies-table">
      <thead><tr>
        <SortTh label="exit" colKey="exit" sort={sort} align="left" />
        <SortTh label="name" colKey="name" sort={sort} align="left" />
        <SortTh label="trades" colKey="trades" sort={sort} align="right" />
        <SortTh label="avg %" colKey="avg_pct" sort={sort} align="right" />
        <SortTh label="win %" colKey="win_pct" sort={sort} align="right" />
        <SortTh label="t" colKey="t" sort={sort} align="right" />
      </tr></thead>
      <tbody>
        {sort.rows.map((r, i) => (
          <tr key={i}>
            <td style={{ fontWeight: 600 }}>{r.exit}</td>
            <td className="dim">{r.name}</td>
            <td style={{ textAlign: 'right' }} className="dim">{fmtN(r.trades)}</td>
            <td style={{ textAlign: 'right' }} className={signCls(r.avg_pct)}>{sPct(r.avg_pct)}</td>
            <td style={{ textAlign: 'right' }} className="dim">{pctR(r.win_pct)}</td>
            <td style={{ textAlign: 'right' }} className="dim">{num2(r.t)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function RsiIntradayPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);
  const [tf, setTf] = useState('4h');

  const load = (which) => { setErr(null); apiFetch(`/rsi-intraday?tf=${which}`).then(setData).catch(e => setErr(e.message)); };
  useEffect(() => { load(tf); }, [tf]);

  const run = () => {
    setRunning(true);
    apiFetch(`/rsi-intraday?tf=${tf}`, { method: 'POST' }).then(() => {
      const t = setInterval(() => apiFetch(`/rsi-intraday?tf=${tf}`).then(d => {
        if (d && d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(e => { setErr(e.message); setRunning(false); });
  };

  // RSI-level lead strip: per bucket, the longest-hold row ('40b' exit if present, else last row).
  const RSI_ORDER = ['<25', '25-35', '35-45', '45-55', '55+'];
  const longestRow = (arr) => {
    const rows = Array.isArray(arr) ? arr : [];
    if (!rows.length) return null;
    return rows.find(r => r.exit === '40b') || rows[rows.length - 1];
  };

  if (err) return (
    <div className="darkpool-page">
      <h1>RSI Intraday Crossover</h1>
      <ErrorBanner message={err} onRetry={() => load(tf)} onDismiss={() => setErr(null)} />
    </div>
  );
  if (!data) return <div className="loading">Loading RSI intraday study...</div>;

  if (!data.computed) {
    return (
      <div className="darkpool-page">
        <h1>RSI Intraday Crossover</h1>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.note || data.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Running…' : 'Run study'}</button>
        </div>
      </div>
    );
  }

  const params = data.params || {};
  const availTf = data.available_tf || ['4h', '8h', '12h'];
  const byRsi = data.backtest_by_rsi || {};
  const hist = params.history || {};

  const rsiStrip = RSI_ORDER.map(k => {
    const r = longestRow(byRsi[k]);
    return { label: k, value: r ? r.avg_pct : null, n: r ? r.trades : null };
  });

  return (
    <div className="darkpool-page">
      <h1>RSI Intraday Crossover <LastUpdatedChip value={data.last_updated} /></h1>
      <div className="filters" style={{ margin: '4px 0 8px' }}>
        {availTf.map(t => (
          <button key={t} className={tf === t ? 'active' : ''} onClick={() => setTf(t)}>{t}</button>
        ))}
      </div>
      <p className="subtitle darkpool-muted">
        RSI({params.rsi_period ?? 14}) crossing above its SMA({params.sma_period ?? 14}). Averaged across all crossovers it
        looks like noise — bucket it by how oversold the cross was and the edge appears.
        {hist.from ? ` History ${hist.from} → ${hist.to}.` : ''}
        {params.n_with_data != null ? ` ${Number(params.n_with_data).toLocaleString()} names with ${data.tf || tf} data (${params.n_daily != null ? Number(params.n_daily).toLocaleString() : '—'} daily).` : ''}
      </p>

      {/* ── LEAD: the tail ─────────────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Edge by RSI level at the cross ({data.tf || tf})</h2>
        </div>
        <div style={{ maxWidth: 360 }}>
          <TailStrip buckets={rsiStrip} />
        </div>
        <p className="subtitle" style={{ marginTop: 10 }}>
          Only the oversold buckets (<b>&lt;35</b>) carry the edge — the deeper the RSI at the cross, the stronger the bounce.
          Crosses fired above 45 are noise.
        </p>
      </div>

      {/* ── Per-bucket exit ladders ────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Exit ladder by RSI bucket ({data.tf || tf})</h2>
        </div>
        {RSI_ORDER.filter(k => Array.isArray(byRsi[k]) && byRsi[k].length).map(k => (
          <RsiBucketTable key={k} bucket={k} rows={byRsi[k]} />
        ))}
      </div>

      {/* ── All crossovers + daily benchmark ───────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>All {data.tf || tf} crossovers (no RSI filter)</h2>
        </div>
        <RsiLadderTable rows={data.backtest_tf || []} />
      </div>

      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Daily benchmark (same signal, daily bars)</h2>
        </div>
        <RsiLadderTable rows={data.backtest_daily || []} />
      </div>

      {data.note && <p className="subtitle" style={{ marginTop: 8, fontStyle: 'italic' }}>{data.note}</p>}
    </div>
  );
}

function VolShockBacktestTable({ sig }) {
  const sort = useSortedRows((sig && sig.rows) || [], 'avg_pct', 'desc');
  const fmtN = (v) => (v == null ? '–' : Number(v).toLocaleString());
  const sPct = (v) => (v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(2)}%`);
  const pctR = (v) => (v == null ? '–' : `${Number(v).toFixed(1)}%`);
  const num2 = (v) => (v == null ? '–' : Number(v).toFixed(2));
  const signCls = (v) => (v == null ? 'dim' : v > 0 ? 'good' : 'bad');
  return (
    <div style={{ marginBottom: 12 }}>
      <h4 style={{ margin: '6px 0' }}>{(sig && sig.name) || ''}</h4>
      <table className="studies-table">
        <thead><tr>
          <SortTh label="exit" colKey="exit" sort={sort} align="left" />
          <SortTh label="name" colKey="name" sort={sort} align="left" />
          <SortTh label="trades" colKey="trades" sort={sort} align="right" />
          <SortTh label="avg %" colKey="avg_pct" sort={sort} align="right" />
          <SortTh label="win %" colKey="win_pct" sort={sort} align="right" />
          <SortTh label="t" colKey="t" sort={sort} align="right" />
        </tr></thead>
        <tbody>
          {sort.rows.map((r, i) => (
            <tr key={i} className={i === 0 ? 'darkpool-base-row' : ''}>
              <td style={{ fontWeight: i === 0 ? 700 : 600 }}>{r.exit}</td>
              <td className="dim">{r.name}</td>
              <td style={{ textAlign: 'right' }} className="dim">{fmtN(r.trades)}</td>
              <td style={{ textAlign: 'right' }} className={signCls(r.avg_pct)}>{sPct(r.avg_pct)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{pctR(r.win_pct)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{num2(r.t)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VolShockPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);
  const [contMode, setContMode] = useState('all'); // 'all' | 'hivol' — Continuation card source toggle

  const load = () => { setErr(null); apiFetch('/vol-shock-study').then(setData).catch(e => setErr(e.message)); };
  useEffect(() => { load(); }, []);

  const run = () => {
    setRunning(true);
    apiFetch('/vol-shock-study', { method: 'POST' }).then(() => {
      const t = setInterval(() => apiFetch('/vol-shock-study').then(d => {
        if (d && d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(e => { setErr(e.message); setRunning(false); });
  };

  const sPct = (v) => (v == null ? '–' : `${v > 0 ? '+' : ''}${Number(v).toFixed(2)}%`);
  const pctR = (v) => (v == null ? '–' : `${Number(v).toFixed(1)}%`);
  const num2 = (v) => (v == null ? '–' : Number(v).toFixed(2));
  const fmtN = (v) => (v == null ? '–' : Number(v).toLocaleString());
  const signCls = (v) => (v == null ? 'dim' : v > 0 ? 'good' : 'bad');

  if (err) return (
    <div className="darkpool-page">
      <h1>Vol-Shock Continuation</h1>
      <ErrorBanner message={err} onRetry={load} onDismiss={() => setErr(null)} />
    </div>
  );
  if (!data) return <div className="loading">Loading vol-shock study...</div>;

  if (!data.computed) {
    return (
      <div className="darkpool-page">
        <h1>Vol-Shock Continuation</h1>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.note || data.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={run} disabled={running}>{running ? 'Computing…' : 'Compute now'}</button>
        </div>
      </div>
    );
  }

  const params = data.params || {};
  const baseline = data.baseline || {};
  const cont = (contMode === 'hivol' ? data.continuation_hivol : data.continuation) || {};
  const slices = data.slices || {};
  const backtest = data.backtest || {};

  const baseHz = ['1', '5', '20'].filter(h => baseline[h] != null);
  const baseLine = baseHz.map(h => `${h}d ${sPct(baseline[h])}`).join(' · ');

  // ── Continuation table (up or dn) ────────────────────────────────
  const ContTable = ({ rows }) => (
    <table className="studies-table">
      <thead><tr>
        <th style={{ textAlign: 'left' }}>thr σ</th>
        <th style={{ textAlign: 'right' }}>H (d)</th>
        <th style={{ textAlign: 'right' }}>episodes</th>
        <th style={{ textAlign: 'right' }}>mean fwd %</th>
        <th style={{ textAlign: 'right' }}>cont %</th>
        <th style={{ textAlign: 'right' }}>edge %</th>
        <th style={{ textAlign: 'right' }}>t</th>
      </tr></thead>
      <tbody>
        {(rows || []).map((r, i) => (
          <tr key={i}>
            <td style={{ fontWeight: 600 }}>{num2(r.thr)}σ</td>
            <td style={{ textAlign: 'right' }} className="dim">{r.H}</td>
            <td style={{ textAlign: 'right' }} className="dim">{fmtN(r.episodes)}</td>
            <td style={{ textAlign: 'right' }} className={signCls(r.mean_pct)}>{sPct(r.mean_pct)}</td>
            <td style={{ textAlign: 'right' }} className="dim">{pctR(r.cont_pct)}</td>
            <td style={{ textAlign: 'right' }} className={signCls(r.edge_pct)}>{sPct(r.edge_pct)}</td>
            <td style={{ textAlign: 'right' }} className="dim">{num2(r.t)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  // ── Slice bucket table (one horizon list) ────────────────────────
  const SliceTable = ({ items }) => (
    <table className="studies-table">
      <thead><tr>
        <th style={{ textAlign: 'left' }}>bucket</th>
        <th style={{ textAlign: 'right' }}>n</th>
        <th style={{ textAlign: 'right' }}>mean %</th>
        <th style={{ textAlign: 'right' }}>cont %</th>
        <th style={{ textAlign: 'right' }}>t</th>
      </tr></thead>
      <tbody>
        {(items || []).map((r, i) => (
          <tr key={i}>
            <td>{r.bucket}</td>
            <td style={{ textAlign: 'right' }} className="dim">{fmtN(r.n)}</td>
            <td style={{ textAlign: 'right' }} className={signCls(r.mean)}>{sPct(r.mean)}</td>
            <td style={{ textAlign: 'right' }} className="dim">{pctR(r.wr)}</td>
            <td style={{ textAlign: 'right' }} className="dim">{num2(r.t)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  // For long slice lists (e.g. sector) keep only the top+bottom 6 by mean.
  const capList = (items) => {
    const arr = (items || []).slice();
    if (arr.length <= 12) return arr;
    const sorted = arr.slice().sort((a, b) => (b.mean ?? -Infinity) - (a.mean ?? -Infinity));
    return sorted.slice(0, 6).concat(sorted.slice(-6));
  };

  const sliceOrder = ['cap', 'vol', 'regime', 'sector'];
  const sliceLabels = { cap: 'Market cap', vol: 'Volatility', regime: 'Market regime', sector: 'Sector' };

  const DirBlock = ({ label, dir, capIt }) => {
    if (!dir) return null;
    return (
      <div className="darkpool-statgrid" style={{ display: 'block', marginBottom: 8 }}>
        <h4 style={{ margin: '6px 0' }}>{label}</h4>
        {['5', '10'].filter(h => dir[h]).map(h => (
          <div key={h} style={{ marginBottom: 8 }}>
            <div className="subtitle" style={{ margin: '2px 0' }}>H = {h} days</div>
            <SliceTable items={capIt ? capList(dir[h]) : dir[h]} />
          </div>
        ))}
      </div>
    );
  };

  const btOrder = ['vol_shock_up', 'vol_shock_dn', 'vol_shock_dn3', 'vol_shock_up_hivol', 'vol_shock_dn_hivol', 'vol_shock_dn3_hivol'];
  const btKeys = btOrder.filter(k => backtest[k]).concat(Object.keys(backtest).filter(k => !btOrder.includes(k)));

  // Shock-size gradient at H=10 for one direction: one bar per σ threshold (1.5σ, 2σ, 3σ),
  // value = mean fwd %. Shows the size ramp at a glance above the detailed table.
  const contStrip = (rows) => {
    const seen = new Set(), out = [];
    (rows || []).filter(r => Number(r.H) === 10).forEach(r => {
      if (r.thr == null || seen.has(r.thr)) return;
      seen.add(r.thr);
      out.push({ label: `${num2(r.thr)}σ`, value: r.mean_pct, n: r.episodes });
    });
    out.sort((a, b) => parseFloat(a.label) - parseFloat(b.label));
    return out;
  };

  return (
    <div className="darkpool-page">
      <h1>Vol-Shock Continuation <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle">{data.note || 'Vol-normalized shock study.'}</p>
      <p className="subtitle">
        Is a day whose move is large relative to the stock's own volatility (z = return ÷ trailing-20d vol) followed by
        continuation or reversal? Entry at the shock close.
        {params.universe && params.universe.names != null ? ` Universe: ${fmtN(params.universe.names)} names.` : ''}
      </p>

      {/* ── Continuation ────────────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Continuation vs reversal</h2>
          <div className="filters">
            <button className={contMode === 'all' ? 'active' : ''} onClick={() => setContMode('all')}>All shocks</button>
            <button className={contMode === 'hivol' ? 'active' : ''} onClick={() => setContMode('hivol')}>Volume-confirmed</button>
          </div>
        </div>
        {baseLine && <p className="subtitle">Baseline drift: {baseLine}</p>}
        {contMode === 'hivol' && <p className="subtitle darkpool-muted">Only shocks on &gt;1.5× average volume — fewer episodes, but the edge is far more reliable (higher t).</p>}
        <div className="darkpool-statgrid" style={{ display: 'block' }}>
          <div style={{ marginBottom: 12 }}>
            <h4 style={{ margin: '6px 0' }}>Good day (+σ) → keeps rising?</h4>
            <p className="subtitle darkpool-muted">For UP shocks, a positive edge % = momentum (kept rising above baseline).</p>
            <div style={{ maxWidth: 260, marginBottom: 6 }}>
              <div className="subtitle darkpool-muted" style={{ margin: '2px 0' }}>Shock-size gradient — mean fwd % at H=10</div>
              <TailStrip buckets={contStrip(cont.up)} />
            </div>
            <ContTable rows={cont.up} />
          </div>
          <div>
            <h4 style={{ margin: '6px 0' }}>Bad day (−σ) → keeps falling?</h4>
            <p className="subtitle darkpool-muted">For DN shocks, a negative edge % = it reversed (bounced back up).</p>
            <div style={{ maxWidth: 260, marginBottom: 6 }}>
              <div className="subtitle darkpool-muted" style={{ margin: '2px 0' }}>Shock-size gradient — mean fwd % at H=10</div>
              <TailStrip buckets={contStrip(cont.dn)} />
            </div>
            <ContTable rows={cont.dn} />
          </div>
        </div>
      </div>

      {/* ── Slices ──────────────────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>Slices — {params.slice_thr != null ? `${num2(params.slice_thr)}σ` : '2σ'} shocks</h2>
        </div>
        {sliceOrder.filter(s => slices[s]).map(s => (
          <div key={s} style={{ marginBottom: 14 }}>
            <h3 style={{ margin: '8px 0' }}>{sliceLabels[s] || s}</h3>
            <DirBlock label="Up shocks (+σ)" dir={slices[s].up} capIt={s === 'sector'} />
            <DirBlock label="Down shocks (−σ)" dir={slices[s].dn} capIt={s === 'sector'} />
          </div>
        ))}
      </div>

      {/* ── Backtest ────────────────────────────────────────────────── */}
      <div className="darkpool-card">
        <div className="darkpool-card-head">
          <h2>vol_shock signals × exit ladder (episode-deduped, no fees — directional)</h2>
        </div>
        {btKeys.map(k => (
          <VolShockBacktestTable key={k} sig={backtest[k]} />
        ))}
      </div>

      {data.note && <p className="subtitle" style={{ marginTop: 8, fontStyle: 'italic' }}>{data.note}</p>}
    </div>
  );
}

// Fundamentals table with a compact analyst rating-distribution column (surfaces the
// EODHD analyst consensus now imported per ticker). Lives in the Research hub.
function FundamentalsPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [sortCol, setSortCol] = useState('market_cap');
  const [sortDir, setSortDir] = useState('desc');
  const [search, setSearch] = useState('');

  useEffect(() => { apiFetch('/fundamentals').then(setData).catch(e => setErr(e.message)); }, []);

  if (err) return <div className="error">Error: {err}</div>;
  if (!data) return <div className="loading">Loading fundamentals...</div>;

  const capFmt = (v) => {
    if (v == null) return '–';
    if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
    if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
    if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`;
    return `$${Number(v).toLocaleString()}`;
  };
  const num = (v, d = 1) => (v == null ? '–' : Number(v).toFixed(d));
  // Compact analyst readout: SB·B·H·S·SS, strong-buy count in green. Full breakdown in title.
  const analystCell = (r) => {
    const parts = [r.analyst_strong_buy, r.analyst_buy, r.analyst_hold, r.analyst_sell, r.analyst_strong_sell];
    const hasDist = parts.some(p => p != null);
    const mean = r.analyst_rating_mean;
    if (!hasDist && mean == null) return <span className="dim">–</span>;
    const title = `Strong Buy ${r.analyst_strong_buy ?? '–'} · Buy ${r.analyst_buy ?? '–'} · Hold ${r.analyst_hold ?? '–'} · Sell ${r.analyst_sell ?? '–'} · Strong Sell ${r.analyst_strong_sell ?? '–'}${mean != null ? ` · mean ${Number(mean).toFixed(2)}/5 (higher = more bullish)` : ''}`;
    return (
      <span title={title}>
        {hasDist && (
          <span className="analyst-dist">
            <span className="good">{r.analyst_strong_buy ?? 0}</span>
            <span className="dim">·{r.analyst_buy ?? 0}·{r.analyst_hold ?? 0}·{r.analyst_sell ?? 0}·</span>
            <span className="bad">{r.analyst_strong_sell ?? 0}</span>
          </span>
        )}
        {mean != null && <span className="analyst-mean"> {Number(mean).toFixed(1)}</span>}
      </span>
    );
  };

  const q = search.trim().toLowerCase();
  let rows = (data.fundamentals || []).filter(r => !q || (r.ticker && r.ticker.toLowerCase().includes(q)) || (r.sector && r.sector.toLowerCase().includes(q)));
  rows = [...rows].sort((a, b) => {
    const av = a[sortCol], bv = b[sortCol];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;              // nulls sort last regardless of dir
    if (bv == null) return -1;
    if (typeof av === 'string') return sortDir === 'desc' ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
    return sortDir === 'desc' ? bv - av : av - bv;
  });

  const setSort = (col) => { if (sortCol === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc'); else { setSortCol(col); setSortDir('desc'); } };
  const arrow = (col) => sortCol === col ? (sortDir === 'desc' ? ' ▼' : ' ▲') : '';
  const cols = [
    ['ticker', 'Ticker', 'left'], ['sector', 'Sector', 'left'], ['market_cap', 'Mkt cap', 'right'],
    ['pe_ratio', 'P/E', 'right'], ['forward_pe', 'Fwd P/E', 'right'], ['revenue_growth', 'Rev growth', 'right'],
    ['profit_margin', 'Margin', 'right'], ['analyst_rating_mean', 'Analyst', 'right'],
  ];

  return (
    <div className="studies-page">
      <h1>Fundamentals <span className="dim">({data.total ?? rows.length})</span>
        <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle">Per-ticker fundamentals with EODHD analyst consensus. The "Analyst" column shows Strong-Buy·Buy·Hold·Sell·Strong-Sell counts and the mean rating (1–5, higher = more bullish); sortable by mean.</p>
      <input className="fund-search" placeholder="Filter by ticker or sector…" value={search} onChange={e => setSearch(e.target.value)} style={{ margin: '4px 0 10px', padding: '6px 10px', width: 260 }} />
      <table className="studies-table">
        <thead><tr>
          {cols.map(([k, l, align]) => (
            <th key={k} className="sortable" style={{ textAlign: align }} onClick={() => setSort(k)}>{l}{arrow(k)}</th>
          ))}
        </tr></thead>
        <tbody>
          {rows.slice(0, 500).map(r => (
            <tr key={r.ticker}>
              <td><b>{r.ticker}</b></td>
              <td className="dim" style={{ fontSize: 11 }}>{r.sector || '–'}</td>
              <td style={{ textAlign: 'right' }} className="dim">{capFmt(r.market_cap)}</td>
              <td style={{ textAlign: 'right' }}>{num(r.pe_ratio)}</td>
              <td style={{ textAlign: 'right' }}>{num(r.forward_pe)}</td>
              <td style={{ textAlign: 'right' }} className={r.revenue_growth > 0 ? 'good' : (r.revenue_growth < 0 ? 'bad' : 'dim')}>{r.revenue_growth == null ? '–' : `${(r.revenue_growth * 100).toFixed(1)}%`}</td>
              <td style={{ textAlign: 'right' }} className={r.profit_margin > 0 ? 'good' : (r.profit_margin < 0 ? 'bad' : 'dim')}>{r.profit_margin == null ? '–' : `${(r.profit_margin * 100).toFixed(1)}%`}</td>
              <td style={{ textAlign: 'right' }}>{analystCell(r)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > 500 && <p className="subtitle">Showing first 500 of {rows.length} matches — filter to narrow.</p>}
    </div>
  );
}

// Pretty-URL navigation via the History API (no '#'). nginx serves index.html for every route
// (SPA fallback already in nginx.conf), so deep links and refreshes work. Browser back/forward is
// handled by the popstate listener in App. Empty/'' means the home route ('/').
function navigate(to) {
  const path = to && to !== '' ? to : '/';
  if (window.location.pathname !== path) window.history.pushState({}, '', path);
}

// ---- Rotation Pick -----------------------------------------------------------
// Live version of the winning sector-rotation strategy. Reads GET /rotation-picks (a
// _StudyResultView-style endpoint); POST kicks a background recompute (poll every 8s until
// `computed`). The KEY insight is the explainer: rotating ETFs LOSES to SPY — the edge is
// using momentum rotation as a FILTER, then buying the cheapest positive-P/B (value) stock
// inside each top-momentum sector.
function RotationPicksPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const picks = (data && data.picks) || [];
  const sort = useSortedRows(picks, 'rank', 'asc');

  const load = () => apiFetch('/rotation-picks').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);

  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/rotation-picks`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/rotation-picks`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading rotation pick...</div>;
  if (data.computed === false) {
    return (
      <div className="studies-page">
        <h1>Rotation Pick</h1>
        <p className="subtitle">The live version of the winning sector-rotation strategy.</p>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.note || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Compute now'}</button>
        </div>
      </div>
    );
  }

  const params = data.params || {};
  const signCls = v => v == null ? 'dim' : v > 0 ? 'good' : v < 0 ? 'bad' : 'dim';
  const capFmt = v => v == null ? '–' : v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(0)}M` : `$${(v / 1e3).toFixed(0)}K`;
  const num = (v, d = 2) => v == null || isNaN(Number(v)) ? '–' : Number(v).toFixed(d);
  const pct = v => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(1)}%`;
  // How many metric cells the "no pick" placeholder should span (P/B, Price, Mkt cap, P/E, Fwd P/E).
  const METRIC_COLS = 5;

  return (
    <div className="studies-page">
      <h1>Rotation Pick <LastUpdatedChip value={data.last_updated} /></h1>

      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '4px 0 16px' }}>
        <p style={{ margin: 0 }}>
          <b>Rotating sector ETFs LOSES to SPY.</b> The edge is using rotation as a <b>FILTER</b>: rank sectors
          by 6-month momentum, take the top {params.top_n_sectors || 10}, and in each buy the <b>CHEAPEST positive-P/B (value) stock</b>.
        </p>
        <p style={{ margin: '8px 0 0' }} className="good">
          Backtest: <b>+154% vs SPY, t=2.09</b> — the only rotation arm with real alpha.
        </p>
        {params.backtest && (
          <p style={{ margin: '8px 0 0', fontFamily: 'ui-monospace, Menlo, monospace', fontSize: 12 }} className="dim">
            {params.backtest}
          </p>
        )}
        {params.rule && <p style={{ margin: '6px 0 0', fontSize: 12 }} className="dim">Rule: {params.rule}</p>}
        {data.note && <p style={{ margin: '6px 0 0', fontSize: 12 }} className="dim">{data.note}</p>}
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Recompute'}</button>
        </p>
      </div>

      <table className="studies-table">
        <thead><tr>
          <SortTh label="Rank" colKey="rank" sort={sort} align="right" />
          <SortTh label="Sector" colKey="sector" sort={sort} />
          <SortTh label="ETF" colKey="etf" sort={sort} />
          <SortTh label="6mo Mom %" colKey="momentum_6m" sort={sort} align="right" />
          <SortTh label="Pick" colKey="pick" sort={sort} />
          <SortTh label="P/B" colKey="pb_ratio" sort={sort} align="right" title="Value metric — lower is cheaper" />
          <SortTh label="Price" colKey="last_close" sort={sort} align="right" />
          <SortTh label="Mkt cap" colKey="market_cap" sort={sort} align="right" />
          <SortTh label="P/E" colKey="pe_ratio" sort={sort} align="right" />
          <SortTh label="Fwd P/E" colKey="forward_pe" sort={sort} align="right" />
          <SortTh label="Profit guard" colKey="guard_status" sort={sort} title="Profitability guard: profitable / turnaround (neg-EPS improving, kept) / ok / trap (excluded from the pick)" />
        </tr></thead>
        <tbody>
          {sort.rows.map(r => (
            <tr key={r.etf || r.rank} className="study-row">
              <td style={{ textAlign: 'right' }} className="dim">{r.rank}</td>
              <td>{r.sector}</td>
              <td className="dim">{r.etf}</td>
              <td style={{ textAlign: 'right' }} className={signCls(r.momentum_6m)}>{pct(r.momentum_6m)}</td>
              {r.pick ? (
                <>
                  <td><b>{r.pick}</b>{r.is_etf_proxy && <span title="Commodity/market ETF held as the position (trend sleeve, not a value pick)" style={{ marginLeft: 5, fontSize: 10, padding: '1px 5px', borderRadius: 4, background: 'rgba(210,153,34,0.16)', color: '#d29922' }}>ETF</span>}</td>
                  <td style={{ textAlign: 'right' }} className={r.pb_ratio != null && r.pb_ratio < 2 ? 'good' : ''}>{r.is_etf_proxy ? <span className="dim" title="no P/B — commodity/market sleeve">n/a</span> : num(r.pb_ratio)}</td>
                  <td style={{ textAlign: 'right' }} className="dim">{num(r.last_close)}</td>
                  <td style={{ textAlign: 'right' }} className="dim">{capFmt(r.market_cap)}</td>
                  <td style={{ textAlign: 'right' }} className="dim">{num(r.pe_ratio, 1)}</td>
                  <td style={{ textAlign: 'right' }} className="dim">{num(r.forward_pe, 1)}</td>
                  <td>{r.is_etf_proxy ? <span className="dim">–</span> : <GuardBadge status={r.guard_status} margin={r.margin_pct} />}</td>
                </>
              ) : (
                <>
                  <td className="dim">—</td>
                  <td className="dim" colSpan={METRIC_COLS + 1} style={{ fontStyle: 'italic' }}>— no data</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>

      <p className="subtitle" style={{ marginTop: 12 }}>
        {data.note || 'Directional, no fees; monthly-rebalance value basket; stock-universe survivorship applies.'}
      </p>
    </div>
  );
}

// ---- Rotation Call (flagship) -----------------------------------------------
// THE headline signal. Reads GET /rotation-call (BacktestResult[rotation_call]); POST recomputes.
// Assembled from the three things that survived the whole rotation research: regime-leader sectors
// (the only prospective sector angle with lift) ∩ cheapest-P/B value pick (the only t>2 equity alpha)
// ∩ an oversold-dip ENTRY on the pick's own price (dip adds, strength subtracts). Explicitly NOT the
// sector TREND TURN alert, which backtests as the worst beat-SPY signal.
function RegimePill({ label, value }) {
  const good = value === 'risk-on' || value === 'rising';
  return <span className="badge" style={{
    background: good ? 'rgba(63,185,80,0.14)' : 'rgba(210,153,34,0.12)',
    color: good ? '#3fb950' : '#d29922', padding: '2px 8px', borderRadius: 6, marginRight: 6, fontSize: 12,
  }}>{label}: <b>{value}</b></span>;
}

function EntryBadge({ txt, k }) {
  const map = {
    deep: ['#238636', '#fff'], enter: ['rgba(63,185,80,0.18)', '#3fb950'],
    wait: ['rgba(139,148,158,0.14)', '#8b949e'], extended: ['rgba(210,153,34,0.14)', '#d29922'],
  };
  const [bg, fg] = map[k] || map.wait;
  return <span style={{ background: bg, color: fg, padding: '2px 8px', borderRadius: 6, fontSize: 12, whiteSpace: 'nowrap' }}>{txt}</span>;
}

// Profitability-guard badge: profitable / turnaround (neg-EPS but improving, kept) / ok / trap (excluded).
function GuardBadge({ status, margin }) {
  if (!status || status === 'unknown') return <span className="dim">–</span>;
  const map = {
    profitable: ['rgba(63,185,80,0.16)', '#3fb950', 'profitable'],
    turnaround: ['rgba(88,166,255,0.16)', '#58a6ff', 'turnaround'],
    ok: ['rgba(139,148,158,0.14)', '#8b949e', 'ok'],
    trap: ['rgba(248,81,73,0.18)', '#f85149', 'TRAP'],
  };
  const [bg, fg, label] = map[status] || map.ok;
  const m = margin != null ? ` ${margin > 0 ? '+' : ''}${margin}%` : '';
  return <span title={`profit guard: ${status}${margin != null ? ` (net margin ${margin}%)` : ''}`}
    style={{ background: bg, color: fg, padding: '1px 6px', borderRadius: 4, fontSize: 11, whiteSpace: 'nowrap' }}>{label}{m}</span>;
}

function RotationCallPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const picks = (data && data.picks) || [];
  const sort = useSortedRows(picks, 'regime_score_pct', 'desc');

  const load = () => apiFetch('/rotation-call').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);

  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/rotation-call`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/rotation-call`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading rotation call...</div>;

  const rg = data.regime || {};
  const comp = data.components || {};
  const num = (v, d = 2) => v == null || isNaN(Number(v)) ? '–' : Number(v).toFixed(d);
  const capFmt = v => v == null ? '–' : v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(0)}M` : `$${(v / 1e3).toFixed(0)}K`;
  const pctS = v => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(2)}%`;
  const rsiCls = v => v == null ? 'dim' : v < 35 ? 'good' : v < 45 ? 'good' : v > 60 ? 'bad' : '';

  return (
    <div className="studies-page">
      <h1>🎯 Rotation Call <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle" style={{ marginTop: 2 }}>
        The flagship call — what actually beats SPY. Not the sector <b>TREND TURN</b> alert (that backtests worst).
      </p>

      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '8px 0 16px' }}>
        <div style={{ marginBottom: 8 }}>
          <b>Regime now ({rg.date}):</b>{' '}
          <RegimePill label="rates" value={rg.rates} />
          <RegimePill label="inflation" value={rg.inflation} />
          <RegimePill label="market" value={rg.market} />
        </div>
        <p style={{ margin: '4px 0 0', fontSize: 13 }}>
          <b>1. Sectors</b> — {comp.sectors || 'macro-regime leadership'}.{' '}
          <b>2. Stock</b> — {comp.stock || 'cheapest positive-P/B value pick'}.{' '}
          <b>3. Entry</b> — {comp.entry || 'oversold dip on the pick’s absolute price'}.
        </p>
        <p style={{ margin: '8px 0 0' }} className={data.ready_to_enter > 0 ? 'good' : 'dim'}>
          <b>{data.ready_to_enter || 0}</b> pick(s) at an oversold-dip entry now (RSI(10) &lt; 45).{' '}
          {data.ready_to_enter === 0 && 'The book is extended — the entry rule says wait for a pullback rather than chase.'}
        </p>
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Recompute'}</button>
        </p>
      </div>

      <table className="studies-table">
        <thead><tr>
          <SortTh label="Sector (regime leader)" colKey="sector" sort={sort} />
          <SortTh label="Regime score" colKey="regime_score_pct" sort={sort} align="right" title="Avg forward-rel lift in the current regime (in-sample)" />
          <SortTh label="Combo hit%" colKey="combo_hit_pct" sort={sort} align="right" title="Historical hit rate in this exact regime combo" />
          <SortTh label="Value pick" colKey="pick" sort={sort} />
          <SortTh label="P/B" colKey="pb_ratio" sort={sort} align="right" title="Lower = cheaper" />
          <SortTh label="Price" colKey="last_close" sort={sort} align="right" />
          <SortTh label="RSI(10)" colKey="rsi10" sort={sort} align="right" title="Entry timing — <45 = dip/enter, >60 = extended" />
          <SortTh label="Entry" colKey="entry_key" sort={sort} />
        </tr></thead>
        <tbody>
          {sort.rows.map(r => (
            <tr key={r.etf} className="study-row">
              <td>{r.sector} <span className="dim">{r.etf}</span></td>
              <td style={{ textAlign: 'right' }} className={r.regime_score_pct > 0 ? 'good' : 'bad'}>{pctS(r.regime_score_pct)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.combo_hit_pct == null ? '–' : `${r.combo_hit_pct}%`}</td>
              {r.pick ? (
                <>
                  <td><b>{r.pick}</b>{r.is_etf_proxy
                    ? <span title="Commodity/market ETF held as the position (trend sleeve, not a value pick)" style={{ marginLeft: 5, fontSize: 10, padding: '1px 5px', borderRadius: 4, background: 'rgba(210,153,34,0.16)', color: '#d29922' }}>ETF</span>
                    : (r.guard_status && r.guard_status !== 'unknown' && <span style={{ marginLeft: 6 }}><GuardBadge status={r.guard_status} margin={r.margin_pct} /></span>)}</td>
                  <td style={{ textAlign: 'right' }} className={r.pb_ratio != null && r.pb_ratio < 2 ? 'good' : ''}>{r.is_etf_proxy ? <span className="dim" title="no P/B — commodity/market sleeve">n/a</span> : num(r.pb_ratio)}</td>
                  <td style={{ textAlign: 'right' }} className="dim">{num(r.last_close)}</td>
                  <td style={{ textAlign: 'right' }} className={rsiCls(r.rsi10)}>{r.rsi10 == null ? '–' : r.rsi10}</td>
                  <td><EntryBadge txt={r.entry_state} k={r.entry_key} /></td>
                </>
              ) : (
                <><td className="dim">—</td><td className="dim" colSpan={4} style={{ fontStyle: 'italic' }}>— no data</td></>
              )}
            </tr>
          ))}
        </tbody>
      </table>

      <p className="subtitle" style={{ marginTop: 12, fontSize: 12 }}>{data.caveat}</p>
    </div>
  );
}

// ---- Entry Signal study -----------------------------------------------------
// Reads GET /entry-signal (BacktestResult[entry_signal]); POST recomputes. Holds the sector+stock
// selection fixed (arm3_lowpb) and varies ONLY the entry gate on the pick's absolute price, isolating
// entry timing. The conditional-lift column is the money read: dip entries add, strength entries subtract.
function EntrySignalPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const load = () => apiFetch('/entry-signal').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);

  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/entry-signal`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/entry-signal`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading entry-signal study...</div>;

  const entries = data.entries || {};
  const order = ['unconditional', ...(data.ranking || [])];
  const rec = data.recommendation || {};
  const rows = order.filter(k => entries[k]).map(k => {
    const e = entries[k], p = e.portfolio || {}, pf = e.portfolio_fallback || {};
    const lbh = e.cond_lift_by_hold || {};
    return { entry: k, lift1: lbh['1'] != null ? lbh['1'] : e.cond_lift_pct, lift2: lbh['2'],
      vs_spy: p.vs_spy, vs_spy_fb: pf.vs_spy, sharpe: p.sharpe, dd: p.max_drawdown, skip: e.skipped_frac,
      isBase: k === 'unconditional', isWinner: k === rec.winner };
  });
  const num = (v, d = 2) => v == null || isNaN(Number(v)) ? '–' : Number(v).toFixed(d);
  const pctS = (v, d = 1) => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(d)}%`;
  const liftCls = v => v == null ? 'dim' : v > 0 ? 'good' : 'bad';

  return (
    <div className="studies-page">
      <h1>⏳ Entry Signal <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle" style={{ marginTop: 2 }}>
        What is the right ENTRY for the value-pick basket? Selection is held fixed (arm3_lowpb); only the entry gate varies.
      </p>

      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '8px 0 16px' }}>
        <p style={{ margin: 0, fontSize: 13 }}>{rec.headline}</p>
        <p style={{ margin: '10px 0 0', fontSize: 13 }}>
          <span className="good">✓ Enter: <b>{rec.winner}</b></span> (best deployable dip entry) ·{' '}
          <span className="good">best risk-adj: <b>{rec.best_risk_adjusted}</b></span> ·{' '}
          <span className="bad">✗ Avoid: <b>{rec.worst}</b></span> (buying strength)
        </p>
        {rec.fallback_note && (
          <p style={{ margin: '10px 0 0', fontSize: 12 }} className="dim"><b>ETF fallback:</b> {rec.fallback_note}</p>
        )}
        {rec.hold_note && (
          <p style={{ margin: '8px 0 0', fontSize: 12 }} className="dim"><b>2-month hold:</b> {rec.hold_note}</p>
        )}
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Recompute'}</button>
        </p>
      </div>

      <table className="studies-table">
        <thead><tr>
          <th>Entry gate</th>
          <th style={{ textAlign: 'right' }} title="Fwd-1mo mean of picks with entry ON minus OFF — the per-pick lift (not confounded by selectivity). THE money column.">Lift @1mo</th>
          <th style={{ textAlign: 'right' }} title="Same per-pick lift at a 2-month hold. Deeper dips improve here — the reversal takes longer to mature.">Lift @2mo</th>
          <th style={{ textAlign: 'right' }} title="Pure value-pick basket (empty sectors dropped)">vs SPY</th>
          <th style={{ textAlign: 'right' }} title="With ETF fallback for empty/gate-failing sectors — the arm3_lowpb / +154% headline construction">vs SPY (fallback)</th>
          <th style={{ textAlign: 'right' }}>Sharpe</th>
          <th style={{ textAlign: 'right' }}>Max DD</th>
          <th style={{ textAlign: 'right' }} title="Fraction of months with zero passing picks (higher = less deployable; a 2-month hold offsets this)">Skip</th>
        </tr></thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.entry} className="study-row" style={r.isWinner ? { background: 'rgba(63,185,80,0.08)' } : r.isBase ? { background: 'rgba(139,148,158,0.06)' } : undefined}>
              <td>{r.isWinner && '✓ '}<b>{r.entry}</b>{r.isBase && <span className="dim"> (baseline)</span>}</td>
              <td style={{ textAlign: 'right' }} className={r.isBase ? 'dim' : liftCls(r.lift1)}>{r.isBase ? '—' : pctS(r.lift1, 2)}</td>
              <td style={{ textAlign: 'right' }} className={r.isBase ? 'dim' : liftCls(r.lift2)}>{r.isBase ? '—' : pctS(r.lift2, 2)}</td>
              <td style={{ textAlign: 'right' }} className={r.vs_spy > 0 ? 'good' : 'bad'}>{pctS(r.vs_spy)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{pctS(r.vs_spy_fb)}</td>
              <td style={{ textAlign: 'right' }}>{num(r.sharpe)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{pctS(r.dd)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.skip == null ? '–' : `${(r.skip * 100).toFixed(0)}%`}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="subtitle" style={{ marginTop: 12, fontSize: 12 }}>
        Baseline (unconditional) shown two ways: <b>vs SPY</b> = pure value picks; <b>vs SPY (fallback)</b> = all-10-slots with ETF fallback = the conservative +154% headline. {data.caveat}
      </p>
    </div>
  );
}

// ---- Profitability Guard study --------------------------------------------
// Reads GET /profitability-guard. Does excluding cheap-P/B value traps help? Blanket "profitable-only"
// HURTS (kills turnarounds); the nuanced ex_trap_turn (drop unprofitable+eroding-book+not-improving,
// keep turnarounds) beats the unguarded baseline on return, t, Sharpe AND drawdown. Now wired live.
function ProfitabilityGuardPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const load = () => apiFetch('/profitability-guard').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);
  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/profitability-guard`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/profitability-guard`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading profitability-guard study...</div>;

  const V = data.variants || {};
  const LABEL = {
    baseline: 'baseline (no guard)', profitable: 'profitable-only', book_stable: 'book not shrinking',
    profitable_book: 'profitable + book', ex_trap: 'exclude trap', ex_trap_turn: 'exclude trap, keep turnarounds',
  };
  const order = ['baseline', 'profitable', 'book_stable', 'profitable_book', 'ex_trap', 'ex_trap_turn'];
  const rows = order.filter(k => V[k]).map(k => {
    const nf = V[k].no_fallback || {}, fb = V[k].fallback || {};
    return { key: k, vs_spy: nf.vs_spy, t: nf.t_stat, sharpe: nf.sharpe, dd: nf.max_drawdown,
      fb_vs_spy: fb.vs_spy, isBase: k === 'baseline', isWinner: k === data.best_guard };
  });
  const pctS = (v, d = 1) => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(d)}%`;
  const num = (v, d = 2) => v == null || isNaN(Number(v)) ? '–' : Number(v).toFixed(d);

  return (
    <div className="studies-page">
      <h1>🛡️ Profitability Guard <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle" style={{ marginTop: 2 }}>
        Does excluding cheap-P/B <b>value traps</b> (unprofitable + eroding book) improve the value pick?
      </p>
      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '8px 0 16px' }}>
        <p style={{ margin: 0, fontSize: 13 }}>{data.recommendation}</p>
        <p style={{ margin: '8px 0 0', fontSize: 13 }}>
          <span className="good">✓ Winner: <b>{data.best_guard}</b></span> — drop only the true trap (unprofitable
          AND book shrinking AND not improving), <b>keep negative-EPS turnarounds</b>. Blanket "profitable-only"
          <span className="bad"> hurts</span> (throws out the turnaround winners). <b>Now wired into the live picks.</b>
        </p>
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Recompute'}</button>
        </p>
      </div>
      <table className="studies-table">
        <thead><tr>
          <th>Variant</th>
          <th style={{ textAlign: 'right' }} title="Pure value picks (empty sectors dropped)">vs SPY</th>
          <th style={{ textAlign: 'right' }}>t</th>
          <th style={{ textAlign: 'right' }}>Sharpe</th>
          <th style={{ textAlign: 'right' }}>Max DD</th>
          <th style={{ textAlign: 'right' }} title="With ETF fallback (realistic)">vs SPY (fallback)</th>
        </tr></thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.key} className="study-row" style={r.isWinner ? { background: 'rgba(63,185,80,0.10)' } : r.isBase ? { background: 'rgba(139,148,158,0.06)' } : undefined}>
              <td>{r.isWinner && '✓ '}<b>{LABEL[r.key] || r.key}</b></td>
              <td style={{ textAlign: 'right' }} className={r.vs_spy > 0 ? 'good' : 'bad'}>{pctS(r.vs_spy)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.t == null ? '–' : r.t}</td>
              <td style={{ textAlign: 'right' }}>{num(r.sharpe)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{pctS(r.dd)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{pctS(r.fb_vs_spy)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="subtitle" style={{ marginTop: 12, fontSize: 12 }}>
        Guard status on the live picks: <b>profitable</b> / <b>turnaround</b> (neg-EPS but improving — kept) /
        <b> ok</b> / <b>trap</b> (excluded). {data.caveat}
      </p>
    </div>
  );
}

// ---- Factor Lab -------------------------------------------------------------
// Reads GET /factor-lab. Sweeps many filters/tilts/combos on the value-pick baseline, ranked to find
// the best return (and best risk-adjusted). Multiple-comparisons caveat travels in the payload.
function FactorLabPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const load = () => apiFetch('/factor-lab').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);
  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/factor-lab`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/factor-lab`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading factor lab...</div>;

  const V = data.variants || {};
  const leg = data.factor_legend || {};
  const order = ['baseline', ...(data.ranking || [])];
  const rows = order.filter(k => V[k]).map(k => {
    const nf = V[k].no_fallback || {}, fb = V[k].fallback || {};
    return { key: k, vs_spy: nf.vs_spy, t: nf.t_stat, sharpe: nf.sharpe, dd: nf.max_drawdown,
      names: nf.avg_names, fb: fb.vs_spy, dbase: V[k].vs_baseline,
      isBase: k === 'baseline', isBest: k === data.best_return, isRA: k === data.best_risk_adjusted };
  });
  const pctS = (v, d = 1) => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(d)}%`;
  const num = (v, d = 2) => v == null || isNaN(Number(v)) ? '–' : Number(v).toFixed(d);

  return (
    <div className="studies-page">
      <h1>🧪 Factor Lab <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle" style={{ marginTop: 2 }}>
        Every filter / tilt / combo on the value pick, ranked to find the best return. Baseline = cheapest-P/B pick.
      </p>
      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '8px 0 16px' }}>
        <p style={{ margin: 0, fontSize: 13 }}>
          <span className="good">🥇 Best return: <b>{data.best_return}</b></span> ·{' '}
          <span className="good">⚖️ Best risk-adjusted: <b>{data.best_risk_adjusted}</b></span> ·{' '}
          baseline <b>{pctS(V.baseline?.no_fallback?.vs_spy)}</b> vs SPY.
        </p>
        <p style={{ margin: '8px 0 0', fontSize: 12 }} className="dim">{data.caveat}</p>
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Recompute'}</button>
        </p>
      </div>
      <table className="studies-table">
        <thead><tr>
          <th>Variant</th>
          <th style={{ textAlign: 'right' }} title="Pure value picks (empty sectors dropped)">vs SPY</th>
          <th style={{ textAlign: 'right' }}>Δ base</th>
          <th style={{ textAlign: 'right' }}>t</th>
          <th style={{ textAlign: 'right' }}>Sharpe</th>
          <th style={{ textAlign: 'right' }}>Max DD</th>
          <th style={{ textAlign: 'right' }} title="Avg names held/month (low = over-selective)">Names</th>
          <th style={{ textAlign: 'right' }} title="With ETF fallback (realistic)">vs SPY (fb)</th>
        </tr></thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.key} className="study-row" title={leg[r.key] || ''}
              style={r.isBest ? { background: 'rgba(63,185,80,0.12)' } : r.isBase ? { background: 'rgba(139,148,158,0.06)' } : undefined}>
              <td>{r.isBest && '🥇 '}{r.isRA && !r.isBest && '⚖️ '}<b>{r.key}</b></td>
              <td style={{ textAlign: 'right' }} className={r.vs_spy > 0 ? 'good' : 'bad'}>{pctS(r.vs_spy)}</td>
              <td style={{ textAlign: 'right' }} className={r.isBase ? 'dim' : (r.dbase > 0 ? 'good' : 'bad')}>{r.isBase ? '—' : pctS(r.dbase)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.t == null ? '–' : r.t}</td>
              <td style={{ textAlign: 'right' }}>{num(r.sharpe)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{pctS(r.dd)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.names == null ? '–' : r.names}</td>
              <td style={{ textAlign: 'right' }} className="dim">{pctS(r.fb)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="empty-state" style={{ textAlign: 'left', padding: '10px 14px', marginTop: 12, fontSize: 12 }}>
        <b>Factor legend:</b> {Object.entries(leg).map(([k, v]) => <div key={k} className="dim" style={{ marginTop: 3 }}><b>{k}</b> — {v}</div>)}
      </div>
    </div>
  );
}

// ---- Portfolio Blender ------------------------------------------------------
// Reads GET /portfolio-blender. Mix CORE value engine + CAPITULATION sleeve; measure correlation +
// crisis-alpha, sweep allocations, find the mix that maximizes return. (Honest result travels in the verdict.)
function PortfolioBlenderPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const load = () => apiFetch('/portfolio-blender').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);
  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/portfolio-blender`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/portfolio-blender`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading portfolio blender...</div>;

  const s = data.sleeves || {}, ca = data.crisis_alpha || {};
  const sweep = data.allocation_sweep || [];
  const bestRet = data.best_return_blend || {}, bestSh = data.best_sharpe_blend || {};
  const pctS = (v, d = 1) => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(d)}%`;
  const raw = (v, d = 1) => v == null || isNaN(Number(v)) ? '–' : `${Number(v).toFixed(d)}%`;
  const crisisBad = ca.cap_mean_when_core_down_pct != null && ca.cap_mean_when_core_down_pct < 0;

  return (
    <div className="studies-page">
      <h1>🔀 Portfolio Blender <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle" style={{ marginTop: 2 }}>
        Mix the CORE value engine with a CAPITULATION sleeve — does blending raise return-for-risk?
      </p>
      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '8px 0 16px' }}>
        <p style={{ margin: 0, fontSize: 13 }}>{data.verdict}</p>
        <p style={{ margin: '8px 0 0', fontSize: 12 }} className={crisisBad ? 'bad' : 'good'}>
          <b>Crisis-alpha check:</b> CAP returns <b>{raw(ca.cap_mean_when_core_down_pct, 2)}</b>/mo when CORE is
          DOWN ({ca.core_down_months} months) vs <b>{raw(ca.cap_mean_when_core_up_pct, 2)}</b>/mo when up —
          correlation <b>{data.correlation_core_cap}</b>. {crisisBad
            ? 'CAP LOSES when CORE loses → pro-cyclical, NOT a hedge → blending only dilutes.'
            : 'CAP pays when CORE bleeds → genuine crisis alpha.'}
        </p>
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Recompute'}</button>
        </p>
      </div>

      <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginBottom: 16 }}>
        {[['CORE (value engine)', s.core], ['CAP (capitulation)', s.cap]].map(([lab, sl]) => sl && (
          <div key={lab} className="empty-state" style={{ padding: '10px 14px', minWidth: 220 }}>
            <b>{lab}</b>
            <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>
              total {raw(sl.total_return)} · vs SPY {pctS(sl.vs_spy)} · Sharpe {sl.sharpe} · DD {raw(sl.max_drawdown)} · vol {raw(sl.vol)}
              {sl.avg_names != null && ` · ${sl.avg_names} names/mo`}
            </div>
          </div>
        ))}
      </div>

      <table className="studies-table">
        <thead><tr>
          <th>Capitulation weight (wA)</th>
          <th style={{ textAlign: 'right' }}>Total</th>
          <th style={{ textAlign: 'right' }}>vs SPY</th>
          <th style={{ textAlign: 'right' }}>Sharpe</th>
          <th style={{ textAlign: 'right' }}>Max DD</th>
          <th style={{ textAlign: 'right' }}>Vol</th>
        </tr></thead>
        <tbody>
          {sweep.map(r => (
            <tr key={r.wA} className="study-row"
              style={r.wA === bestSh.wA ? { background: 'rgba(63,185,80,0.10)' } : (r.wA === 0 ? { background: 'rgba(139,148,158,0.06)' } : undefined)}>
              <td><b>{(r.wA * 100).toFixed(0)}% CAP / {(100 - r.wA * 100).toFixed(0)}% CORE</b>{r.wA === 0 && <span className="dim"> (CORE only)</span>}{r.wA === bestSh.wA && r.wA !== 0 && ' ⭐ best Sharpe'}</td>
              <td style={{ textAlign: 'right' }} className="good">{raw(r.total_return)}</td>
              <td style={{ textAlign: 'right' }} className={r.vs_spy > 0 ? 'good' : 'bad'}>{pctS(r.vs_spy)}</td>
              <td style={{ textAlign: 'right' }}>{r.sharpe}</td>
              <td style={{ textAlign: 'right' }} className="dim">{raw(r.max_drawdown)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{raw(r.vol)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="subtitle" style={{ marginTop: 12, fontSize: 12 }}>
        Best return: <b>{(bestRet.wA * 100).toFixed(0)}% CAP</b> ({raw(bestRet.total_return)}). Regime-switched
        vol-matched: <b>{raw(data.regime_switched?.vol_matched?.total_return)}</b>. {data.caveat}
      </p>
    </div>
  );
}

// ---- Strategy Lab -----------------------------------------------------------
// Reads GET /strategy-lab. Can A/B beat C WITHOUT the sector rotation, and do C's rules travel?
function StrategyLabPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const load = () => apiFetch('/strategy-lab').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);
  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/strategy-lab`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/strategy-lab`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading strategy lab...</div>;

  const R = data.results || {};
  const LABEL = {
    C_ref: 'C — rotation + rules (the engine)', B_pure: 'B pure — dip in uptrend, no rotation',
    B_value: 'B value — cheap-P/B dips, no rotation', B_quality: "B + C's rules (all quality dips)",
    B_value_q: "B + cheapest book ratio + low debt + guard", A_pure: 'A pure — capitulation, no rotation',
    A_quality: "A + C's rules", A_value_q: 'A + cheapest book ratio + low debt + guard',
  };
  const order = ['C_ref', 'B_pure', 'B_value', 'B_quality', 'B_value_q', 'A_pure', 'A_quality', 'A_value_q'];
  const rows = order.filter(k => R[k]).map(k => ({ key: k, ...R[k] }));
  const pctS = (v) => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(1)}%`;
  const c = data.c_ref_vs_spy;

  return (
    <div className="studies-page">
      <h1>🔬 Strategy Lab <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle" style={{ marginTop: 2 }}>
        Can A or B beat C <b>without the sector rotation</b>, and do C's rules travel? Isolates the rotation filter vs the pick rules.
      </p>
      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '8px 0 16px' }}>
        <p style={{ margin: 0, fontSize: 13 }}>{data.verdict}</p>
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Recompute'}</button>
        </p>
      </div>
      <table className="studies-table">
        <thead><tr>
          <th>Strategy</th>
          <th style={{ textAlign: 'right' }}>vs SPY</th>
          <th style={{ textAlign: 'right' }}>t</th>
          <th style={{ textAlign: 'right' }}>Sharpe</th>
          <th style={{ textAlign: 'right' }}>Max DD</th>
          <th style={{ textAlign: 'right' }}>Names</th>
        </tr></thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.key} className="study-row"
              style={r.key === 'C_ref' ? { background: 'rgba(63,185,80,0.12)' } : (r.vs_spy > c ? { background: 'rgba(88,166,255,0.08)' } : undefined)}>
              <td><b>{LABEL[r.key] || r.key}</b></td>
              <td style={{ textAlign: 'right' }} className={r.vs_spy > 0 ? 'good' : 'bad'}>{pctS(r.vs_spy)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.t_stat == null ? '–' : r.t_stat}</td>
              <td style={{ textAlign: 'right' }}>{r.sharpe}</td>
              <td style={{ textAlign: 'right' }} className="dim">{pctS(r.max_drawdown)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.avg_names}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="subtitle" style={{ marginTop: 12, fontSize: 12 }}>
        <b>Read:</b> rotation is the RETURN amplifier (concentration into strong sectors); C's rules are a portable
        QUALITY edge (they rescue A/B) but can't replace the rotation. B+rules is the best low-drawdown alternative. {data.caveat}
      </p>
    </div>
  );
}

// ---- Value Ranking Lab ------------------------------------------------------
// Reads GET /value-ranking. Which value metric picks the best name? Selection (rotation+guard+low_debt)
// held fixed; only the ranking metric varies. Result: cheapest P/B beats EV/EBIT, FCF-yield, composite.
function ValueRankingPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const load = () => apiFetch('/value-ranking').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);
  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/value-ranking`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/value-ranking`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading value ranking lab...</div>;

  const RES = data.results || {}, leg = data.legend || {};
  const rows = (data.ranking || Object.keys(RES)).filter(k => RES[k]).map(k => ({ key: k, ...RES[k].no_fallback, dbase: RES[k].vs_baseline, fb: RES[k].fallback?.vs_spy }));
  const pctS = (v) => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(1)}%`;

  return (
    <div className="studies-page">
      <h1>💎 Value Ranking Lab <LastUpdatedChip value={data.last_updated} /></h1>
      <p className="subtitle" style={{ marginTop: 2 }}>
        Which value metric picks the best stock? Selection (rotation + guard + low_debt) fixed; only the ranking metric varies.
      </p>
      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '8px 0 16px' }}>
        <p style={{ margin: 0, fontSize: 13 }}>{data.verdict}</p>
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Recompute'}</button>
        </p>
      </div>
      <table className="studies-table">
        <thead><tr>
          <th>Value metric</th>
          <th style={{ textAlign: 'right' }}>vs SPY</th>
          <th style={{ textAlign: 'right' }}>Δ vs P/B</th>
          <th style={{ textAlign: 'right' }}>t</th>
          <th style={{ textAlign: 'right' }}>Sharpe</th>
          <th style={{ textAlign: 'right' }}>Max DD</th>
        </tr></thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.key} className="study-row"
              style={r.key === data.best_metric ? { background: 'rgba(63,185,80,0.12)' } : undefined}>
              <td title={leg[r.key] || ''}>{r.key === data.best_metric && '🏆 '}<b>{r.key}</b> <span className="dim" style={{ fontSize: 11 }}>{leg[r.key]}</span></td>
              <td style={{ textAlign: 'right' }} className={r.vs_spy > 0 ? 'good' : 'bad'}>{pctS(r.vs_spy)}</td>
              <td style={{ textAlign: 'right' }} className={r.key === 'pb' ? 'dim' : (r.dbase > 0 ? 'good' : 'bad')}>{r.key === 'pb' ? '—' : pctS(r.dbase)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.t_stat == null ? '–' : r.t_stat}</td>
              <td style={{ textAlign: 'right' }}>{r.sharpe}</td>
              <td style={{ textAlign: 'right' }} className="dim">{pctS(r.max_drawdown)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="subtitle" style={{ marginTop: 12, fontSize: 12 }}>
        <b>Why P/B wins:</b> book value is stable/slow-moving, so cheapest-P/B reliably finds the genuinely
        asset-cheap name; EV/EBIT and FCF-yield pick names on temporarily-inflated single-quarter earnings/cash
        flow (cyclical peaks) that mean-revert. {data.caveat}
      </p>
    </div>
  );
}

// RS-trend method sweep. Reads GET /rs-methods (BacktestResult[rs_methods]); POST recomputes (poll
// every 8s until `computed`). ~20 ways to read the ETF/SPY relative-strength bar, each feeding the
// SAME cheapest-P/B large-cap pick — so only the sector-SELECTION rule varies. The banner carries the
// multiple-comparisons caveat: rank by t-stat / Sharpe / drawdown, not by the single highest vs-SPY.
function RsMethodsPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const methods = (data && data.methods) || [];
  const sort = useSortedRows(methods, 'vs_spy', 'desc');

  const load = () => apiFetch('/rs-methods').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);

  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/rs-methods`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/rs-methods`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading RS methods...</div>;
  if (data.computed === false) {
    return (
      <div className="studies-page">
        <h1>RS Methods</h1>
        <p className="subtitle">~20 ways to read the ETF/SPY relative-strength bar, backtested side-by-side.</p>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing (~2-3 min)...' : 'Compute now'}</button>
        </div>
      </div>
    );
  }

  const params = data.params || {};
  const pct = v => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(1)}`;
  const num = (v, d = 2) => v == null || isNaN(Number(v)) ? '–' : Number(v).toFixed(d);
  const signCls = v => v == null ? 'dim' : v > 0 ? 'good' : v < 0 ? 'bad' : 'dim';
  const tCls = v => v == null ? 'dim' : v >= 2 ? 'good' : v >= 1.5 ? '' : 'dim';       // t>=2 is the bar
  const shCls = v => v == null ? 'dim' : v >= 1 ? 'good' : v >= 0.5 ? '' : 'dim';
  const FAM = { absolute: '#8892b0', 'rs-mom': '#64ffda', 'rs-trend': '#57cbff',
                'rs-osc': '#c792ea', breakout: '#ffcb6b', slope: '#f78c6c', 'risk-adj': '#82aaff',
                combo: '#addb67', weekly: '#ff5370', 'ma-cross': '#7fdbca', indicator: '#f07178',
                engine: '#ffd54f', reference: '#546e7a' };

  return (
    <div className="studies-page">
      <h1>RS Methods <LastUpdatedChip value={data.last_updated} /></h1>

      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '4px 0 16px' }}>
        <p style={{ margin: 0 }}>
          Every row uses the <b>same value pick</b> ({params.pick || 'cheapest positive-P/B large-cap'}) — only the
          sector-<b>selection</b> rule on the ETF/SPY bar changes. Benchmark: {params.benchmark || 'SPY buy-and-hold'}.
        </p>
        {data.caveat && (
          <p style={{ margin: '8px 0 0', fontSize: 12 }} className="bad">
            ⚠ {data.caveat}
          </p>
        )}
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Recomputing (~2-3 min)...' : 'Recompute'}</button>
        </p>
      </div>

      <table className="studies-table">
        <thead><tr>
          <SortTh label="Method" colKey="method" sort={sort} />
          <SortTh label="Family" colKey="family" sort={sort} />
          <SortTh label="Total %" colKey="total_return" sort={sort} align="right" />
          <SortTh label="vs SPY %" colKey="vs_spy" sort={sort} align="right" title="Excess return over SPY buy-and-hold" />
          <SortTh label="t-stat" colKey="t_stat" sort={sort} align="right" title="t>=2 = the significance bar; below that, treat as noise" />
          <SortTh label="Sharpe" colKey="sharpe" sort={sort} align="right" />
          <SortTh label="Max DD %" colKey="max_drawdown" sort={sort} align="right" title="Worst peak-to-trough" />
          <SortTh label="Avg sectors" colKey="avg_sectors" sort={sort} align="right" title="Avg # sectors selected/month — low = selective, ~10 = barely filtering" />
        </tr></thead>
        <tbody>
          {sort.rows.map((r, i) => (
            <tr key={r.method || i} className="study-row" style={r.family === 'reference' ? { opacity: 0.8 } : null}>
              <td title={r.description}>{r.method}</td>
              <td><span style={{ color: FAM[r.family] || '#8892b0', fontSize: 12 }}>{r.family}</span></td>
              <td style={{ textAlign: 'right' }} className={signCls(r.total_return)}>{pct(r.total_return)}</td>
              <td style={{ textAlign: 'right', fontWeight: 600 }} className={signCls(r.vs_spy)}>{pct(r.vs_spy)}</td>
              <td style={{ textAlign: 'right' }} className={tCls(r.t_stat)}>{num(r.t_stat, 2)}</td>
              <td style={{ textAlign: 'right' }} className={shCls(r.sharpe)}>{num(r.sharpe, 2)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{pct(r.max_drawdown)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.avg_sectors == null ? '–' : num(r.avg_sectors, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="subtitle" style={{ marginTop: 12 }}>
        {params.periods ? `${params.periods} monthly periods. ` : ''}
        Sorted by vs-SPY by default — but read the whole column: a rule is only trustworthy when its family agrees.
      </p>
    </div>
  );
}

// MA crossover run on all 93 synthetic RS candles. Reads GET /synthetic-ma-cross; POST recomputes.
// The finding it shows: a golden cross ON the relative-strength bar is MEAN-REVERTING (forward relative
// return is negative, t hugely negative) even though the ETF still rises in absolute terms — so it is
// NOT a buy signal. Only the coarse golden-vs-death state carries a thin edge.
function SyntheticMaCrossPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const load = () => apiFetch('/synthetic-ma-cross').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);

  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/synthetic-ma-cross`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/synthetic-ma-cross`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading synthetic MA-cross...</div>;
  if (data.computed === false) {
    return (
      <div className="studies-page">
        <h1>Synthetic MA-Cross</h1>
        <p className="subtitle">Moving-average crossover run on every synthetic ETF/SPY relative-strength candle.</p>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Compute now'}</button>
        </div>
      </div>
    );
  }

  const params = data.params || {};
  const pairs = data.pairs || [];
  const horizons = params.horizons_days || [21, 63, 126];
  const sgn = v => v == null ? 'dim' : v > 0 ? 'good' : v < 0 ? 'bad' : 'dim';
  const p1 = v => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(2)}%`;

  return (
    <div className="studies-page">
      <h1>Synthetic MA-Cross <LastUpdatedChip value={data.last_updated} /></h1>

      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '4px 0 16px' }}>
        <p style={{ margin: 0 }}>
          Golden/death crossovers run on all <b>{params.n_synthetic_candles || 93} synthetic {params.benchmark || 'SPY'}-relative candles</b> (Close = etf/spy).
          Forward <b>relative</b> return = ETF return minus {params.benchmark || 'SPY'} over the window (&gt;0 = beat the index).
        </p>
        <p style={{ margin: '8px 0 0' }} className="bad">
          ⚠ A golden cross <b>on the RS bar is mean-reverting, not a buy signal</b>: forward <b>relative</b> return is
          negative at every horizon (t hugely negative). The ETF still rises in <b>absolute</b> terms (beta) — it just
          lags {params.benchmark || 'SPY'}. Only the coarse golden-vs-death <b>state</b> carries a thin edge.
        </p>
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Recomputing...' : 'Recompute'}</button>
        </p>
      </div>

      {pairs.map(row => {
        const st = row.state_test || {};
        const gc = row.now_golden || [];
        return (
          <div key={row.pair} style={{ margin: '0 0 22px' }}>
            <h3 style={{ margin: '0 0 6px' }}>SMA {row.pair} on the RS bar</h3>
            <p className="subtitle" style={{ margin: '0 0 4px' }}>
              State drift: while <span className="good">golden</span> the sector out-drifts {params.benchmark || 'SPY'}{' '}
              <span className={sgn(st.golden_ann_rs_pct)}>{p1(st.golden_ann_rs_pct)}/yr</span>; while{' '}
              <span className="bad">death</span> <span className={sgn(st.death_ann_rs_pct)}>{p1(st.death_ann_rs_pct)}/yr</span>.
            </p>
            {row.pre_cross && (
              <p className="subtitle" style={{ margin: '0 0 8px' }}>
                <b>Run-up INTO the cross</b> (relative, the move that already happened):{' '}
                {(params.pre_windows_days || [5, 10, 20]).map((w, i) => {
                  const a = row.pre_cross[`${w}d`] || {};
                  return <span key={w} className={sgn(a.mean_pct)}>{i ? ' · ' : ''}{w}d {p1(a.mean_pct)}</span>;
                })}
                {' '}<span className="dim">— the golden cross is a lagging confirmation; forward returns are already reverting.</span>
              </p>
            )}
            <table className="studies-table">
              <thead><tr>
                <th style={{ textAlign: 'left' }}>Horizon</th>
                <th style={{ textAlign: 'right' }} title="Forward relative return after a golden cross">Golden: rel</th>
                <th style={{ textAlign: 'right' }}>% pos</th>
                <th style={{ textAlign: 'right' }}>t</th>
                <th style={{ textAlign: 'right' }} title="Forward absolute ETF return after a golden cross (beta)">Golden: abs ETF</th>
                <th style={{ textAlign: 'right' }} title="Forward relative return after a death cross">Death: rel</th>
                <th style={{ textAlign: 'right' }}>t</th>
                <th style={{ textAlign: 'right' }}>n</th>
              </tr></thead>
              <tbody>
                {horizons.map(h => {
                  const u = (row.cross_up || {})[`${h}d`] || {};
                  const a = (row.abs_after_golden || {})[`${h}d`] || {};
                  const d = (row.cross_dn || {})[`${h}d`] || {};
                  return (
                    <tr key={h} className="study-row">
                      <td>+{h}d</td>
                      <td style={{ textAlign: 'right' }} className={sgn(u.mean_pct)}>{p1(u.mean_pct)}</td>
                      <td style={{ textAlign: 'right' }} className="dim">{u.pos_pct == null ? '–' : `${u.pos_pct}%`}</td>
                      <td style={{ textAlign: 'right' }} className="dim">{u.t == null ? '–' : u.t}</td>
                      <td style={{ textAlign: 'right' }} className={sgn(a.mean_pct)}>{p1(a.mean_pct)}</td>
                      <td style={{ textAlign: 'right' }} className={sgn(d.mean_pct)}>{p1(d.mean_pct)}</td>
                      <td style={{ textAlign: 'right' }} className="dim">{d.t == null ? '–' : d.t}</td>
                      <td style={{ textAlign: 'right' }} className="dim">{u.n == null ? '–' : u.n.toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="subtitle" style={{ margin: '8px 0 0' }}>
              <b>Now:</b> {gc.length} sectors in a golden cross, {row.now_death_n} in a death cross.
              {gc.length ? ' Freshest: ' : ''}
              {gc.slice(0, 10).map((g, i) => (
                <span key={g.etf} className="good">{i ? ', ' : ''}{g.sector} ({g.days_since_cross}d)</span>
              ))}
              {' '}<span className="dim">— not a buy list on its own (see above).</span>
            </p>
          </div>
        );
      })}
      <p className="subtitle">{data.note}</p>
    </div>
  );
}

// Short-term absolute single-stock oversold-reversal entry. Reads GET /oversold-bounce; POST recomputes.
// The edge the RS bar LACKS: on absolute price, oversold RSI crosses give positive short-term forward
// return and deeper oversold pays more (the tail). Shows the entry backtest + a live "just crossed out of
// oversold, most-oversold-first" firing list.
function OversoldBouncePage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const firing = (data && data.firing) || [];
  const sort = useSortedRows(firing, 'min_rsi_7d', 'asc');

  const load = () => apiFetch('/oversold-bounce').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);
  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/oversold-bounce`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/oversold-bounce`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading oversold bounce...</div>;
  if (data.computed === false) {
    return (
      <div className="studies-page">
        <h1>Oversold Bounce</h1>
        <p className="subtitle">Short-term single-stock oversold-reversal entry.</p>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing (~3-4 min)...' : 'Compute now'}</button>
        </div>
      </div>
    );
  }

  const params = data.params || {};
  const horizons = params.horizons_days || [1, 3, 5, 10, 21];
  const sgn = v => v == null ? 'dim' : v > 0 ? 'good' : v < 0 ? 'bad' : 'dim';
  const p2 = v => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(2)}%`;
  const capFmt = v => v == null ? '–' : v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(0)}M` : `$${(v / 1e3).toFixed(0)}K`;

  return (
    <div className="studies-page">
      <h1>Oversold Bounce <LastUpdatedChip value={data.last_updated} /></h1>
      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '4px 0 16px' }}>
        <p style={{ margin: 0 }}>
          Absolute single-stock <b>oversold-reversal entry</b> across {params.universe_n || '~1000'} stocks —
          the edge the relative-strength bar <b>lacks</b>. On absolute price, RSI crossing up out of oversold
          gives positive short-term forward return, and <b>deeper oversold pays more</b> (the tail).
        </p>
        <p style={{ margin: '8px 0 0', fontSize: 12 }} className="dim">{data.note}</p>
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Recomputing...' : 'Recompute'}</button>
        </p>
      </div>

      <h3 style={{ margin: '0 0 6px' }}>Entry backtest — forward absolute return (mean / %pos / t)</h3>
      <table className="studies-table">
        <thead><tr><th style={{ textAlign: 'left' }}>Entry (RSI cross up)</th>{horizons.map(h => <th key={h} style={{ textAlign: 'right' }}>+{h}d</th>)}</tr></thead>
        <tbody>
          {(data.by_threshold || []).map(row => (
            <tr key={row.event} className="study-row">
              <td>{row.event}</td>
              {horizons.map(h => {
                const c = (row.fwd || {})[`${h}d`] || {};
                return <td key={h} style={{ textAlign: 'right' }} className={sgn(c.mean_pct)} title={`${c.pos_pct}% pos, t${c.t}, n${c.n}`}>{p2(c.mean_pct)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ margin: '16px 0 6px' }}>The tail — x30 crosses by oversold depth (min RSI, prior 5 bars)</h3>
      <table className="studies-table">
        <thead><tr><th style={{ textAlign: 'left' }}>Prior-min RSI</th><th style={{ textAlign: 'right' }}>+5d</th><th style={{ textAlign: 'right' }}>+10d</th><th style={{ textAlign: 'right' }}>n</th></tr></thead>
        <tbody>
          {(data.by_depth || []).map(row => (
            <tr key={row.bucket} className="study-row">
              <td>{row.bucket}</td>
              <td style={{ textAlign: 'right' }} className={sgn(row.fwd5?.mean_pct)} title={`${row.fwd5?.pos_pct}% pos, t${row.fwd5?.t}`}>{p2(row.fwd5?.mean_pct)}</td>
              <td style={{ textAlign: 'right' }} className={sgn(row.fwd10?.mean_pct)} title={`${row.fwd10?.pos_pct}% pos, t${row.fwd10?.t}`}>{p2(row.fwd10?.mean_pct)}</td>
              <td style={{ textAlign: 'right' }} className="dim">{row.fwd5?.n?.toLocaleString?.() ?? '–'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ margin: '16px 0 6px' }}>Firing now — crossed up out of oversold (most oversold first) <span className="dim" style={{ fontWeight: 400, fontSize: 13 }}>({firing.length})</span></h3>
      <table className="studies-table">
        <thead><tr>
          <SortTh label="Ticker" colKey="ticker" sort={sort} />
          <SortTh label="Sector" colKey="sectors" sort={sort} />
          <SortTh label="Min RSI 7d" colKey="min_rsi_7d" sort={sort} align="right" title="How oversold it got — lower = deeper = bigger expected bounce" />
          <SortTh label="RSI now" colKey="rsi_now" sort={sort} align="right" />
          <SortTh label="Cross" colKey="threshold" sort={sort} align="right" />
          <SortTh label="Bars ago" colKey="days_ago" sort={sort} align="right" />
          <SortTh label="Close" colKey="close" sort={sort} align="right" />
          <SortTh label="Mkt cap" colKey="market_cap" sort={sort} align="right" />
        </tr></thead>
        <tbody>
          {sort.rows.map(r => (
            <tr key={r.ticker} className="study-row">
              <td><b>{r.ticker}</b></td>
              <td className="dim">{(r.sectors || []).join(', ')}</td>
              <td style={{ textAlign: 'right' }} className={r.min_rsi_7d < 20 ? 'good' : ''}>{r.min_rsi_7d}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.rsi_now}</td>
              <td style={{ textAlign: 'right' }} className="dim">x{r.threshold}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.days_ago}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.close}</td>
              <td style={{ textAlign: 'right' }} className="dim">{capFmt(r.market_cap)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="subtitle" style={{ marginTop: 12 }}>Directional; forward windows, no fees; stock-universe survivorship. Not wired into risk rating.</p>
    </div>
  );
}

// Diversifiers: rank the 93 sleeves by correlation to SPY (low = diversifier), not by return. Reads
// GET /diversifier. Commodities/bonds don't beat SPY but the uncorrelated ones (Gold the standout) add
// drawdown control. good_diversifier = corr<0.35 AND positive 5y return.
function DiversifierPage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const rows = (data && data.rows) || [];
  const sort = useSortedRows(rows, 'corr', 'asc');

  const load = () => apiFetch('/diversifier').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);
  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/diversifier`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/diversifier`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading diversifiers...</div>;
  if (data.computed === false) {
    return (
      <div className="studies-page">
        <h1>Diversifiers</h1>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Compute now'}</button>
        </div>
      </div>
    );
  }

  const s = data.summary || {};
  const sgn = v => v == null ? 'dim' : v > 0 ? 'good' : v < 0 ? 'bad' : 'dim';
  const pc = v => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(1)}%`;

  return (
    <div className="studies-page">
      <h1>Diversifiers <LastUpdatedChip value={data.last_updated} /></h1>
      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '4px 0 16px' }}>
        <p style={{ margin: 0 }}>
          Sleeves ranked by <b>correlation to SPY</b> (low = diversifier), not by return. SPY 5y{' '}
          <span className="good">{pc(s.spy_5y)}</span>; only <b>{s.n_beat_spy}/{s.n_sleeves}</b> sleeves beat it.
          Commodities don't reliably beat SPY — their value is being <b>uncorrelated</b> (drawdown control).
        </p>
        <p style={{ margin: '8px 0 0' }} className="good">
          Best diversifiers (corr &lt; {data.params?.lowcorr_threshold ?? 0.35}, positive return): {(s.good_diversifiers || []).join(', ') || '–'}.
          {' '}<span className="dim">Gold is the standout — beat SPY <b>and</b> corr ~0.15.</span>
        </p>
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Recomputing...' : 'Recompute'}</button>
        </p>
      </div>

      <table className="studies-table">
        <thead><tr>
          <SortTh label="Sleeve" colKey="name" sort={sort} />
          <SortTh label="ETF" colKey="etf" sort={sort} />
          <SortTh label="5y ret" colKey="ret5y" sort={sort} align="right" />
          <SortTh label="vs SPY" colKey="vs_spy" sort={sort} align="right" />
          <SortTh label="Corr SPY" colKey="corr" sort={sort} align="right" title="Daily-return correlation to SPY — lower = better diversifier" />
          <SortTh label="Beta" colKey="beta" sort={sort} align="right" />
          <SortTh label="Ann vol" colKey="ann_vol" sort={sort} align="right" />
          <SortTh label="Diversifier" colKey="good_diversifier" sort={sort} align="right" />
        </tr></thead>
        <tbody>
          {sort.rows.map(r => (
            <tr key={r.etf} className="study-row" style={r.good_diversifier ? { background: 'rgba(100,255,218,0.06)' } : null}>
              <td>{r.name}{r.commodity ? <span className="dim" style={{ fontSize: 11 }}> ·comm</span> : ''}</td>
              <td className="dim">{r.etf}</td>
              <td style={{ textAlign: 'right' }} className={sgn(r.ret5y)}>{pc(r.ret5y)}</td>
              <td style={{ textAlign: 'right' }} className={sgn(r.vs_spy)}>{pc(r.vs_spy)}</td>
              <td style={{ textAlign: 'right' }} className={r.corr < 0.35 ? 'good' : ''}>{r.corr}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.beta}</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.ann_vol}%</td>
              <td style={{ textAlign: 'right' }}>{r.good_diversifier ? <span className="good">✓</span> : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="subtitle" style={{ marginTop: 12 }}>{data.note}</p>
    </div>
  );
}

// Macro regime -> sector leadership. Reads GET /regime. Classifies the month by rates/inflation/market and
// shows which sectors historically led in the CURRENT regime (the honest angle on "which sectors beat SPY
// at the right time" — momentum had ~0 lift; regime conditioning has real, if in-sample, lift).
function RegimePage() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [running, setRunning] = useState(false);

  const load = () => apiFetch('/regime').then(setData).catch(e => setErr(e.message));
  useEffect(() => { load(); }, []);
  const runCompute = () => {
    setRunning(true);
    fetch(`${API}/regime`, { method: 'POST' }).then(r => r.json()).then(() => {
      const t = setInterval(() => fetch(`${API}/regime`).then(r => r.json()).then(d => {
        if (d.computed) { clearInterval(t); setRunning(false); setData(d); }
      }).catch(() => {}), 8000);
    }).catch(() => setRunning(false));
  };

  if (err) return <div className="studies-page"><ErrorBanner message={err} onRetry={() => { setErr(null); load(); }} /></div>;
  if (!data) return <div className="loading">Loading regime...</div>;
  if (data.computed === false) {
    return (
      <div className="studies-page">
        <h1>Regime</h1>
        <div className="empty-state" style={{ padding: '40px 0' }}>
          <p>{data.message || 'Not computed yet.'}</p>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Computing...' : 'Compute now'}</button>
        </div>
      </div>
    );
  }

  const nl = data.now_labels || {};
  const base = data.base_rate || {};
  const pc = v => v == null || isNaN(Number(v)) ? '–' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(2)}%`;
  const hitCls = v => v == null ? 'dim' : v >= 55 ? 'good' : v >= 45 ? '' : 'bad';
  const Pill = ({ label, on, onColor }) => (
    <span style={{ display: 'inline-block', padding: '4px 12px', borderRadius: 14, margin: '0 8px 0 0',
      fontWeight: 600, fontSize: 13, background: on ? onColor : 'rgba(120,120,120,0.15)',
      color: on ? '#0b1220' : '#8892b0' }}>{label}</span>
  );

  const miniTable = (rows, activeCol) => (
    <table className="studies-table" style={{ fontSize: 13 }}>
      <tbody>
        {(rows || []).slice(0, 6).map(r => (
          <tr key={r.etf} className="study-row">
            <td>{r.sector}</td>
            <td className="dim">{r.etf}</td>
            <td style={{ textAlign: 'right' }} className={r.mean_pct > 0 ? 'good' : 'bad'}>{pc(r.mean_pct)}</td>
            <td style={{ textAlign: 'right' }} className={hitCls(r.hit_pct)}>{r.hit_pct}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  const axisBlock = (axisKey, states) => {
    const ax = (data.by_axis || {})[axisKey] || {};
    const cur = nl[axisKey];
    return (
      <div style={{ margin: '0 0 18px' }}>
        <h3 style={{ margin: '0 0 6px', textTransform: 'capitalize' }}>{axisKey}</h3>
        <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
          {states.map(st => (
            <div key={st} style={{ flex: '1 1 320px', border: cur === st ? '1px solid #64ffda' : '1px solid rgba(120,120,120,0.2)',
              borderRadius: 6, padding: '8px 10px' }}>
              <div style={{ fontWeight: 600, marginBottom: 4 }}>
                {axisKey} {st} {cur === st && <span className="good" style={{ fontSize: 12 }}>· now</span>}
                <span className="dim" style={{ fontWeight: 400, fontSize: 11 }}> — top leaders (fwd 3mo rel / hit)</span>
              </div>
              {miniTable(ax[st])}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="studies-page">
      <h1>Regime <LastUpdatedChip value={data.last_updated} /></h1>
      <div className="empty-state" style={{ textAlign: 'left', padding: '14px 16px', margin: '4px 0 16px' }}>
        <div style={{ marginBottom: 8 }}>
          <b style={{ marginRight: 10 }}>Now ({data.now?.date}):</b>
          <Pill label={`Rates ${nl.rates}`} on={true} onColor={nl.rates === 'rising' ? '#ffcb6b' : '#82aaff'} />
          <Pill label={`Inflation ${nl.inflation}`} on={true} onColor={nl.inflation === 'rising' ? '#ff5370' : '#addb67'} />
          <Pill label={nl.market} on={true} onColor={nl.market === 'risk-on' ? '#64ffda' : '#ff5370'} />
        </div>
        <p style={{ margin: 0 }}>
          The honest angle on <b>which sectors beat SPY at the right time</b>: price-momentum has ~0 lift, but
          conditioning on <b>macro regime</b> does. Base rate (all sectors, fwd 3mo vs SPY):{' '}
          <span className="bad">hit {base.hit_pct}% / mean {pc(base.mean_pct)}</span> — most sectors lag SPY.
        </p>
        {data.caveat && <p style={{ margin: '8px 0 0', fontSize: 12 }} className="bad">⚠ {data.caveat}</p>}
        <p style={{ margin: '10px 0 0' }}>
          <button className="refresh-btn" onClick={runCompute} disabled={running}>{running ? 'Recomputing...' : 'Recompute'}</button>
        </p>
      </div>

      <h3 style={{ margin: '0 0 6px' }}>Leaders in the CURRENT regime combo</h3>
      <table className="studies-table">
        <thead><tr>
          <th style={{ textAlign: 'left' }}>Sector</th><th style={{ textAlign: 'left' }}>ETF</th>
          <th style={{ textAlign: 'right' }} title="Avg of the 3 conditional means">Regime score</th>
          <th style={{ textAlign: 'right' }} title="Fwd 3mo relative return in the exact current combo">Combo mean</th>
          <th style={{ textAlign: 'right' }}>Combo hit%</th>
          <th style={{ textAlign: 'right' }}>n</th>
        </tr></thead>
        <tbody>
          {(data.leaders_now || []).map(r => (
            <tr key={r.etf} className="study-row">
              <td><b>{r.sector}</b></td>
              <td className="dim">{r.etf}</td>
              <td style={{ textAlign: 'right' }} className={r.regime_score_pct > 0 ? 'good' : 'bad'}>{pc(r.regime_score_pct)}</td>
              <td style={{ textAlign: 'right' }} className={r.combo_mean_pct > 0 ? 'good' : 'bad'}>{pc(r.combo_mean_pct)}</td>
              <td style={{ textAlign: 'right' }} className={hitCls(r.combo_hit_pct)}>{r.combo_hit_pct}%</td>
              <td style={{ textAlign: 'right' }} className="dim">{r.combo_n}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ margin: '18px 0 8px' }}>Leadership by regime axis</h3>
      {axisBlock('rates', ['rising', 'falling'])}
      {axisBlock('inflation', ['rising', 'falling'])}
      {axisBlock('market', ['risk-on', 'risk-off'])}
    </div>
  );
}

// NOTE: name kept as parseHash for minimal churn; it now reads window.location.pathname (pretty URLs).
function parseHash() {
  const h = window.location.pathname;
  if (h.startsWith('/settings')) return { view: 'settings' };
  if (h.startsWith('/live')) return { view: 'live' };
  if (h.startsWith('/alt-data')) return { view: 'altdata' };
  if (h.startsWith('/dark-pool')) return { view: 'darkpool' };
  if (h.startsWith('/regime')) return { view: 'regime' };
  if (h.startsWith('/oversold-bounce')) return { view: 'oversoldbounce' };
  if (h.startsWith('/diversifier')) return { view: 'diversifier' };
  if (h.startsWith('/synthetic-ma-cross')) return { view: 'synthmacross' };
  if (h.startsWith('/rotation-call')) return { view: 'rotationcall' };
  if (h.startsWith('/entry-signal')) return { view: 'entrysignal' };
  if (h.startsWith('/profitability-guard')) return { view: 'profitguard' };
  if (h.startsWith('/factor-lab')) return { view: 'factorlab' };
  if (h.startsWith('/portfolio-blender')) return { view: 'blender' };
  if (h.startsWith('/strategy-lab')) return { view: 'strategylab' };
  if (h.startsWith('/value-ranking')) return { view: 'valuerank' };
  if (h.startsWith('/rs-methods')) return { view: 'rsmethods' };
  if (h.startsWith('/rotation')) return { view: 'rotationpick' };
  if (h.startsWith('/vol-shock')) return { view: 'volshock' };
  if (h.startsWith('/backtest')) return { view: 'backtestlab' };
  if (h.startsWith('/news')) return { view: 'news' };
  if (h.startsWith('/research')) return { view: 'research' };
  // back-compat: the old single Studies hub is now split; land in Research.
  if (h.startsWith('/studies')) return { view: 'research' };
  if (h.startsWith('/journal')) return { view: 'journal' };
  if (h.startsWith('/trends')) return { view: 'research' };
  if (h.startsWith('/drilldown')) return { view: 'drilldown' };
  if (h.startsWith('/docs')) return { view: 'docs' };
  if (h.startsWith('/study/')) {
    const parts = h.slice(7).split('/');
    if (parts.length >= 3) {
      return { view: 'study_chart', signal: decodeURIComponent(parts[0]), exit: decodeURIComponent(parts[1]), sector: decodeURIComponent(parts[2]) };
    }
  }
  if (h.startsWith('/sector/')) return { view: 'sector', sector: decodeURIComponent(h.slice(8)) };
  if (h.startsWith('/chart/')) return { view: 'chart', ticker: h.slice(7) };
  return { view: 'home' };
}

export default function App() {
  const [page, setPage] = useState('dashboard');
  const [scanData, setScanData] = useState(null);
  const [drilldown, setDrilldown] = useState(null);
  const [selectedSector, setSelectedSector] = useState(null);
  const [chartTicker, setChartTicker] = useState(null);
  const [chartSectorEtf, setChartSectorEtf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortCol, setSortCol] = useState('rsi_spread');
  const [sortDir, setSortDir] = useState('desc');
  const [filter, setFilter] = useState('all');
  const [dashInterval, setDashInterval] = useState('1d');
  const [peMap, setPeMap] = useState({});

  const [backendDown, setBackendDown] = useState(false);

  // Fetch fundamentals ONCE and build a { ticker: forward_pe ?? pe_ratio } map so the
  // dashboard board (SectorTable, keyed by ETF ticker) can show a P/E column.
  useEffect(() => {
    apiFetch('/fundamentals')
      .then(d => {
        const rows = (d && d.fundamentals) || [];
        const m = {};
        rows.forEach(r => {
          const v = r.forward_pe != null ? r.forward_pe : r.pe_ratio;
          if (r.ticker != null && v != null) m[r.ticker] = v;
        });
        setPeMap(m);
      })
      .catch(() => {});
  }, []);

  const loadScan = (force = false, interval = dashInterval) => {
    setLoading(true);
    setError(null);
    const p = new URLSearchParams();
    if (force) p.set('force', 'true');
    if (interval !== '1d') p.set('interval', interval);
    apiFetch(`/scan?${p.toString()}`)
      .then(d => { setScanData(d); setBackendDown(false); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  };

  // Global backend-health poll — drives a persistent banner on EVERY page when the API is
  // unreachable (so a 502/restart shows a clear message + retry instead of dead/blank pages).
  useEffect(() => {
    let stop = false;
    const ping = () => apiFetch('/sectors')
      .then(() => { if (!stop) setBackendDown(false); })
      .catch(() => { if (!stop) setBackendDown(true); });
    const t = setInterval(ping, 20000);
    return () => { stop = true; clearInterval(t); };
  }, []);

  const handleIntervalChange = (iv) => {
    setDashInterval(iv);
    loadScan(false, iv);
  };

  useEffect(() => {
    loadScan();
    // Apply the current URL to app state — runs on mount (deep links) and on browser back/forward
    // (popstate). Pretty URLs come from navigate()'s pushState; this keeps state in sync.
    const applyRoute = () => {
      const r = parseHash();
      const simple = { settings: 'settings', live: 'live', news: 'news', research: 'research',
                       journal: 'journal', drilldown: 'drilldown', docs: 'docs', backtestlab: 'backtestlab',
                       altdata: 'altdata', darkpool: 'darkpool', rotationpick: 'rotationpick',
                       rotationcall: 'rotationcall', entrysignal: 'entrysignal', profitguard: 'profitguard', factorlab: 'factorlab', blender: 'blender', strategylab: 'strategylab', valuerank: 'valuerank',
                       rsmethods: 'rsmethods', synthmacross: 'synthmacross', oversoldbounce: 'oversoldbounce',
                       diversifier: 'diversifier', regime: 'regime', volshock: 'volshock' };
      if (simple[r.view]) { setPage(simple[r.view]); setSelectedSector(null); setChartTicker(null); }
      else if (r.view === 'sector') { setPage('dashboard'); handleSectorClick(r.sector); }
      else if (r.view === 'chart') { setPage('dashboard'); setChartTicker(r.ticker); }
      else { setPage('dashboard'); setSelectedSector(null); setChartTicker(null); }
    };
    applyRoute();
    window.addEventListener('popstate', applyRoute);
    return () => window.removeEventListener('popstate', applyRoute);
    // eslint-disable-next-line
  }, []);

  const handleRefresh = () => {
    fetch(`${API}/refresh`).then(() => loadScan(true));
  };

  const handleSectorClick = (sector) => {
    const sectorData = scanData?.sectors?.find(s => s.sector === sector);
    const etf = sectorData?.etf;

    setSelectedSector(sector);
    setDrilldown(null);
    setChartTicker(null);
    navigate(`/sector/${encodeURIComponent(sector)}`);
    fetch(`${API}/drilldown/${encodeURIComponent(sector)}`)
      .then(r => r.json())
      .then(d => {
        if (!d.stocks || d.stocks.length === 0) {
          setSelectedSector(null);
          setChartTicker(etf);
          navigate(`/chart/${etf}`);
        } else {
          setDrilldown(d);
        }
      })
      .catch(e => setError(e.message));
  };

  const handleChartClick = (ticker, sectorEtf = null) => {
    setChartTicker(ticker);
    setChartSectorEtf(sectorEtf);
    navigate(`/chart/${ticker}`);
  };

  const handleBack = () => {
    setSelectedSector(null);
    setChartTicker(null);
    navigate('');
  };

  const handleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortCol(col); setSortDir('desc'); }
  };

  const filteredData = scanData ? {
    ...scanData,
    sectors: scanData.sectors.filter(s => {
      if (filter === 'bullish') return s.bullish;
      if (filter === 'rotate') return s.rsi_crossover && s.bullish;
      if (filter === 'bearish') return s.signal === 'BEARISH';
      return true;
    }).map(s => ({ ...s, pe: peMap[s.etf] != null ? peMap[s.etf] : null }))
  } : null;

  return (
    <div className="layout">
      {backendDown && (
        <div className="backend-down-banner" role="alert">
          <span>&#9888; Backend unavailable — the server may be restarting or busy. Data may be stale.</span>
          <button onClick={() => { loadScan(true); }}>Retry now</button>
        </div>
      )}
      <nav className="sidebar">
        <div className="sidebar-logo">Rotation</div>
        <ul className="sidebar-menu">
          <li className={`sidebar-item ${page === 'dashboard' ? 'active' : ''}`} onClick={() => { setPage('dashboard'); setSelectedSector(null); setChartTicker(null); setChartSectorEtf(null); navigate(''); }}>
            <span className="sidebar-icon">&#9678;</span>
            Rotation Dashboard
          </li>
          <li className={`sidebar-item ${page === 'live' ? 'active' : ''}`} onClick={() => { setPage('live'); navigate('/live'); }}>
            <span className="sidebar-icon">&#9889;</span>
            Live Signals
          </li>
          <li className={`sidebar-item ${page === 'news' ? 'active' : ''}`} onClick={() => { setPage('news'); navigate('/news'); }}>
            <span className="sidebar-icon">&#128240;</span>
            News
          </li>
          <li className={`sidebar-item ${page === 'research' ? 'active' : ''}`} onClick={() => { setPage('research'); navigate('/research'); }}>
            <span className="sidebar-icon">&#9733;</span>
            Research
          </li>
          <li className={`sidebar-item ${page === 'backtestlab' ? 'active' : ''}`} onClick={() => { setPage('backtestlab'); navigate('/backtest'); }}>
            <span className="sidebar-icon">&#128200;</span>
            Backtest
          </li>
          <li className={`sidebar-item ${page === 'drilldown' ? 'active' : ''}`} onClick={() => { setPage('drilldown'); navigate('/drilldown'); }}>
            <span className="sidebar-icon">&#9660;</span>
            Stock Drilldown
          </li>
          <li className={`sidebar-item ${page === 'altdata' ? 'active' : ''}`} onClick={() => { setPage('altdata'); navigate('/alt-data'); }}>
            <span className="sidebar-icon">&#127963;</span>
            Alt-Data
          </li>
          <li className={`sidebar-item ${page === 'darkpool' ? 'active' : ''}`} onClick={() => { setPage('darkpool'); navigate('/dark-pool'); }}>
            <span className="sidebar-icon">&#9899;</span>
            Dark Pool
          </li>
          <li className={`sidebar-item ${page === 'rotationcall' ? 'active' : ''}`} onClick={() => { setPage('rotationcall'); navigate('/rotation-call'); }}>
            <span className="sidebar-icon">&#127919;</span>
            Rotation Call
          </li>
          <li className={`sidebar-item ${page === 'rotationpick' ? 'active' : ''}`} onClick={() => { setPage('rotationpick'); navigate('/rotation'); }}>
            <span className="sidebar-icon">&#128260;</span>
            Rotation Pick
          </li>
          <li className={`sidebar-item ${page === 'entrysignal' ? 'active' : ''}`} onClick={() => { setPage('entrysignal'); navigate('/entry-signal'); }}>
            <span className="sidebar-icon">&#9203;</span>
            Entry Signal
          </li>
          <li className={`sidebar-item ${page === 'profitguard' ? 'active' : ''}`} onClick={() => { setPage('profitguard'); navigate('/profitability-guard'); }}>
            <span className="sidebar-icon">&#128737;</span>
            Profit Guard
          </li>
          <li className={`sidebar-item ${page === 'factorlab' ? 'active' : ''}`} onClick={() => { setPage('factorlab'); navigate('/factor-lab'); }}>
            <span className="sidebar-icon">&#129514;</span>
            Factor Lab
          </li>
          <li className={`sidebar-item ${page === 'blender' ? 'active' : ''}`} onClick={() => { setPage('blender'); navigate('/portfolio-blender'); }}>
            <span className="sidebar-icon">&#128256;</span>
            Blender
          </li>
          <li className={`sidebar-item ${page === 'strategylab' ? 'active' : ''}`} onClick={() => { setPage('strategylab'); navigate('/strategy-lab'); }}>
            <span className="sidebar-icon">&#128300;</span>
            Strategy Lab
          </li>
          <li className={`sidebar-item ${page === 'valuerank' ? 'active' : ''}`} onClick={() => { setPage('valuerank'); navigate('/value-ranking'); }}>
            <span className="sidebar-icon">&#128142;</span>
            Value Ranking
          </li>
          <li className={`sidebar-item ${page === 'rsmethods' ? 'active' : ''}`} onClick={() => { setPage('rsmethods'); navigate('/rs-methods'); }}>
            <span className="sidebar-icon">&#128202;</span>
            RS Methods
          </li>
          <li className={`sidebar-item ${page === 'synthmacross' ? 'active' : ''}`} onClick={() => { setPage('synthmacross'); navigate('/synthetic-ma-cross'); }}>
            <span className="sidebar-icon">&#10005;</span>
            Synthetic MA-Cross
          </li>
          <li className={`sidebar-item ${page === 'oversoldbounce' ? 'active' : ''}`} onClick={() => { setPage('oversoldbounce'); navigate('/oversold-bounce'); }}>
            <span className="sidebar-icon">&#8623;</span>
            Oversold Bounce
          </li>
          <li className={`sidebar-item ${page === 'diversifier' ? 'active' : ''}`} onClick={() => { setPage('diversifier'); navigate('/diversifier'); }}>
            <span className="sidebar-icon">&#9878;</span>
            Diversifiers
          </li>
          <li className={`sidebar-item ${page === 'regime' ? 'active' : ''}`} onClick={() => { setPage('regime'); navigate('/regime'); }}>
            <span className="sidebar-icon">&#127760;</span>
            Regime
          </li>
          <li className={`sidebar-item ${page === 'volshock' ? 'active' : ''}`} onClick={() => { setPage('volshock'); navigate('/vol-shock'); }}>
            <span className="sidebar-icon">&#9889;</span>
            Vol-Shock
          </li>
          <li className={`sidebar-item ${page === 'journal' ? 'active' : ''}`} onClick={() => { setPage('journal'); navigate('/journal'); }}>
            <span className="sidebar-icon">&#9998;</span>
            Trade Journal
          </li>
          <li className={`sidebar-item ${page === 'docs' ? 'active' : ''}`} onClick={() => { setPage('docs'); navigate('/docs'); }}>
            <span className="sidebar-icon">&#9776;</span>
            Docs
          </li>
          <li className={`sidebar-item ${page === 'settings' ? 'active' : ''}`} onClick={() => { setPage('settings'); navigate('/settings'); }}>
            <span className="sidebar-icon">&#9881;</span>
            Settings
          </li>
        </ul>
      </nav>
      <div className="main">
        {page === 'settings' ? <SettingsPage /> : page === 'docs' ? <DocsPage /> : page === 'drilldown' ? <StockDrilldownPage /> : page === 'live' ? <LiveSignalsHub /> : page === 'backtestlab' ? <BacktestLabPage /> : page === 'news' ? <NewsHub /> : page === 'research' ? <ResearchHub /> : page === 'altdata' ? <AltDataPage /> : page === 'darkpool' ? <DarkPoolPage /> : page === 'rotationcall' ? <RotationCallPage /> : page === 'entrysignal' ? <EntrySignalPage /> : page === 'profitguard' ? <ProfitabilityGuardPage /> : page === 'factorlab' ? <FactorLabPage /> : page === 'blender' ? <PortfolioBlenderPage /> : page === 'strategylab' ? <StrategyLabPage /> : page === 'valuerank' ? <ValueRankingPage /> : page === 'rotationpick' ? <RotationPicksPage /> : page === 'rsmethods' ? <RsMethodsPage /> : page === 'synthmacross' ? <SyntheticMaCrossPage /> : page === 'oversoldbounce' ? <OversoldBouncePage /> : page === 'diversifier' ? <DiversifierPage /> : page === 'regime' ? <RegimePage /> : page === 'volshock' ? <VolShockPage /> : page === 'journal' ? <TradeJournalPage /> : <>
        <header>
          <h1>Sector Rotation Dashboard</h1>
          <div className="header-right">
            <p className="subtitle">RSI(10) &gt; SMA(10) + Omega(10) &gt; 1</p>
            <div className="header-actions">
              <button className="refresh-btn" onClick={handleRefresh} disabled={loading}>
                {loading ? 'Scanning...' : 'Refresh Data'}
              </button>
              {scanData?.cached_at && <span className="cached-at">Cached: {new Date(scanData.cached_at).toLocaleString()}</span>}
            </div>
          </div>
        </header>
        {error && <ErrorBanner message={error} onRetry={() => loadScan(true)} onDismiss={() => setError(null)} />}
        {chartTicker && <ChartView ticker={chartTicker} sectorEtf={chartSectorEtf} onClose={() => { setChartTicker(null); setChartSectorEtf(null); if (!selectedSector) navigate(''); }} />}
        {selectedSector ? (
          <StockTable sector={selectedSector} data={drilldown} onBack={handleBack} onTickerClick={(t, etf) => handleChartClick(t, etf)} />
        ) : (
          <>
            <DashboardBacktestPanel />
            <div className="filters">
              <button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>All ({scanData?.total || 0})</button>
              <button className={filter === 'bullish' ? 'active' : ''} onClick={() => setFilter('bullish')}>Bullish ({scanData?.bullish || 0})</button>
              <button className={filter === 'rotate' ? 'active' : ''} onClick={() => setFilter('rotate')}>Trend Turn ({scanData?.sectors?.filter(s => s.rsi_crossover && s.bullish).length || 0})</button>
              <button className={filter === 'bearish' ? 'active' : ''} onClick={() => setFilter('bearish')}>Bearish ({scanData?.sectors?.filter(s => s.signal === 'BEARISH').length || 0})</button>
              <span className="filter-sep">|</span>
              <button className={dashInterval === '1d' ? 'active' : ''} onClick={() => handleIntervalChange('1d')}>Daily</button>
              <button className={dashInterval === '1wk' ? 'active' : ''} onClick={() => handleIntervalChange('1wk')}>Weekly</button>
            </div>
            {loading ? <div className="loading">Scanning 94 sectors...</div> : <SectorTable data={filteredData} onSectorClick={handleSectorClick} onEtfClick={handleChartClick} sortCol={sortCol} sortDir={sortDir} onSort={handleSort} />}
          </>
        )}
        </>}
      </div>
    </div>
  );
}
