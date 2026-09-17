/**
 * Find TradingView indicator IDs (built-in + public/community Pine scripts) to feed tv_fetch.js.
 * RUNS ONLY IN DOCKER.  Usage:  docker compose run --rm tv tv_search.js "<query>"
 * Examples: tv_search.js LuxAlgo | tv_search.js SMC | tv_search.js RSI
 * NOTE: search matches single tokens — "Smart Money" (2 words) returns nothing; use "SMC" or "LuxAlgo".
 * Copy the printed id (e.g. PUB;6daa...) into: tv_fetch.js <SYMBOL> <TF> <NBARS> "<id>"
 */
const TradingView = require('@mathieuc/tradingview');
const query = process.argv[2] || 'RSI';
TradingView.searchIndicator(query)
  .then((list) => {
    console.log(`found ${list.length} for "${query}":`);
    list.slice(0, 25).forEach((i) => console.log(`  ${i.id}   v${i.version || '?'}   ${i.name}   [by ${i.author}]`));
    process.exit(0);
  })
  .catch((e) => {
    console.error('search failed:', e && e.message ? e.message : e);
    process.exit(1);
  });
