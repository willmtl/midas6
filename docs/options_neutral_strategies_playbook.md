# Non-Directional Options Strategies — Strategic Playbook

*Saved 2026-08-22. Reference cheat sheet for the four core neutral (non-directional) option
structures: when to use each, by target-price precision, capital size, and risk tolerance.*

> **Ties into our own data:** the earnings vol-risk-premium study (`BacktestResult[preearn_straddle]`,
> 948 liquid events 2022–2026) found **implied move 8.8% vs realized 6.85%, straddle buyer wins only
> ~31%** → the *seller* wins ~69%, +1.95pp/event average edge. That is precisely the "high IV you
> expect to fall / IV crush" regime these sell-premium structures target. Caveats from that study:
> the richest premium is in smaller/less-liquid names (widest spreads), mega-caps are near-efficient,
> and the short-vol tail is exactly the "big pop" we were originally hunting — negative skew.

---

## 1. Iron Condor
- **Best for:** Small-to-medium retail accounts wanting a reliable, high-probability income stream with strict protection.
- **Ideal environment:** High IV expected to fall (IV crush) + stock stuck in a broad, established horizontal channel.
- **The case:** You want a wide margin of error. Stock at $100 → short strikes at $90 / $110; it may wiggle but is unlikely to move >10% either way.
- **Why here:** **Defined risk.** A surprise breakout is capped completely by the long "wings" — no catastrophic loss.

## 2. Short Strangle
- **Best for:** Advanced traders with large margin accounts who prioritize highest win probability over safety nets.
- **Ideal environment:** High-priced, highly liquid stocks/indices (SPY, QQQ) with bloated premium after an overreaction to noise.
- **The case:** Widest possible profitable range — no money spent on protective wings, so break-evens are pushed far out. Bet the asset settles somewhere in that broad window.
- **Why here:** **Maximum efficiency** — decays faster (higher theta) and profits quicker from falling vol (vega) than an iron condor. **Requires margin to back unlimited risk** on a black swan.

## 3. Iron Butterfly
- **Best for:** Budget-conscious traders precisely targeting an exact price for a huge return relative to risk.
- **Ideal environment:** Overpriced, highly anticipated events where you expect the asset to "pin" a specific number or see a massive vol drop right after — **earnings reports, Fed announcements**.
- **The case:** High confidence the stock barely budges from its current price post-event. Selling ATM options right at the current price pulls in a massive premium credit.
- **Why here:** **Asymmetric risk/reward** — protective wings cap risk, but upside can be several times the risk if the stock finishes exactly on the center strike.

## 4. Short Straddle
- **Best for:** Institutional/professional traders squeezing the absolute maximum premium from a stock expected to stay stagnant.
- **Ideal environment:** Extremely quiet, low-beta, range-bound tape with VIX steadily compressing.
- **The case:** Target a specific price point (like the iron butterfly) but refuse to give up any premium buying protection. Belief: stock stays frozen and both ATM options erode fast.
- **Why here:** **Highest possible upfront premium** of any neutral strategy; the big credit builds directional break-even buffers — but **exposed to unlimited, aggressive loss** the moment the stock trends away from the center strike.

---

## Cheat Sheet

| Situation | Best Strategy | Key Benefit | Major Drawback |
|---|---|---|---|
| "I want a safe, wide range." | **Iron Condor** | Capped losses ✅ | Smaller profits |
| "I want the widest win-range possible." | **Short Strangle** | Fastest decay | Unlimited risk ⚠️ |
| "It will hit a specific target price safely." | **Iron Butterfly** | High payout potential | Hard to pinpoint exact center |
| "It will hit a specific target with maximum payout." | **Short Straddle** | Massive credit collected | Extreme (undefined) risk |

---

### Quick mental model
- **Defined risk** (has wings): Iron Condor, Iron Butterfly — retail/budget, capped tail.
- **Undefined risk** (no wings): Short Strangle, Short Straddle — need margin, unlimited tail.
- **Wide range** (OTM strikes): Iron Condor, Short Strangle — bet on "stays in a broad zone."
- **Pinpoint** (ATM strikes): Iron Butterfly, Short Straddle — bet on "pins this exact price," max premium.
