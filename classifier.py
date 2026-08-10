import json
import re

def classify_headline(title, ticker):
    """
    Classify a single headline returning:
    - dir: -1 (bearish), 0 (neutral), +1 (bullish)
    - impact: 0 (noise), 1 (minor), 2 (moderate), 3 (major)
    - cat: category string
    - horizon: day, week, month, or 3mo
    """
    t = title.lower()

    # Detect category first
    cat = None
    dir_val = 0
    impact_val = 0
    horizon = "month"

    # Earnings patterns (major impact)
    if re.search(r'earnings|q[1-4]\s*\d+', t) or 'financial results' in t:
        if any(w in t for w in ['beat', 'above', 'exceed', 'surge', 'jumps', 'profit', 'strong', 'record']):
            cat = "earnings_beat"
            dir_val = 1
            impact_val = 3
            horizon = "3mo"
        elif any(w in t for w in ['miss', 'below', 'disappointing', 'decline', 'loss', 'slump', 'slips', 'down']):
            cat = "earnings_miss"
            dir_val = -1
            impact_val = 3
            horizon = "3mo"
        else:
            cat = "earnings_beat"
            impact_val = 3
            horizon = "3mo"

    # Guidance changes
    elif any(w in t for w in ['guidance', 'outlook']) or 'raises' in t or 'raised' in t or 'hiking' in t or 'trims' in t:
        if any(w in t for w in ['up', 'raise', 'hike', 'increases', 'boost', 'exceed']):
            cat = "guidance_up"
            dir_val = 1
            impact_val = 2
            horizon = "month"
        elif any(w in t for w in ['cut', 'lower', 'down', 'reduce', 'trim', 'decrease']):
            cat = "guidance_down"
            dir_val = -1
            impact_val = 2
            horizon = "month"

    # Analyst upgrades/downgrades
    elif 'analyst' in t and any(w in t for w in ['upgrade', 'upgraded', 'initiates', 'initiated', 'raise']):
        cat = "upgrade"
        dir_val = 1
        impact_val = 2
        horizon = "week"
    elif 'analyst' in t and any(w in t for w in ['downgrade', 'downgraded', 'neutral', 'hold']):
        cat = "downgrade"
        dir_val = -1
        impact_val = 2
        horizon = "week"

    # M&A patterns (major impact)
    elif any(w in t for w in ['merger', 'acquisition', 'acquired', 'acquire', 'acquires', 'takeover']):
        cat = "ma"
        dir_val = 1
        impact_val = 3
        horizon = "day"
    elif 'to buy' in t or 'to acquire' in t or '$' in t and 'billion' in t and any(w in t for w in ['deal', 'investment', 'invests']):
        cat = "ma"
        dir_val = 1
        impact_val = 3
        horizon = "day"

    # Product launches & awards (not major unless very significant)
    elif any(w in t for w in ['launches', 'launched', 'unveil', 'unveiled', 'introduces', 'announced']) and any(w in t for w in ['product', 'model', 'vehicle', 'car', 'ev', 'chip', 'service', 'device']):
        if 'concept' in t:
            cat = "product"
            impact_val = 1
            horizon = "week"
        else:
            cat = "product"
            impact_val = 2
            horizon = "month"
        dir_val = 1
    elif any(w in t for w in ['award', 'awards', 'named', 'honored', 'earns', 'wins', 'recognize']):
        cat = "product"
        dir_val = 1
        impact_val = 1
        horizon = "week"

    # Contracts/partnerships
    elif any(w in t for w in ['partner', 'partnership', 'deal', 'contract', 'agreement', 'collaborate', 'alliance', 'mou', 'signs', 'expand']) or 'to expand' in t:
        cat = "contract"
        dir_val = 1
        impact_val = 2 if ('billion' in t or 'expand' in t or 'strategic' in t) else 1
        horizon = "month"

    # Legal/regulatory/investigations
    elif any(w in t for w in ['legal', 'lawsuit', 'sued', 'settlement', 'investigation', 'audit', 'probe', 'banned', 'fraud', 'criminal', 'bankrupt', 'bankruptcy', 'nhtsa opens', 'opens investigation']):
        cat = "legal"
        dir_val = -1
        if any(w in t for w in ['bankrupt', 'bankruptcy', 'fraud', 'criminal']):
            impact_val = 3
            horizon = "3mo"
        else:
            impact_val = 2
            horizon = "month"

    # Management changes
    elif any(w in t for w in ['ceo', 'chairman', 'executive', 'leader', 'chief']):
        if any(w in t for w in ['steps down', 'resigns', 'exit', 'replace', 'new', 'appoint', 'transition']):
            cat = "mgmt"
            dir_val = -1 if any(w in t for w in ['steps down', 'resigns', 'exit']) else 0
            impact_val = 2
            horizon = "3mo"

    # Capital actions
    elif any(w in t for w in ['dividend', 'buyback', 'share offering', 'repurchase', 'stock split', 'debt', 'raises capital']):
        if 'dividend' in t:
            cat = "dividend"
            dir_val = 1 if any(w in t for w in ['raises', 'increases']) else 0
            impact_val = 1
            horizon = "week"
        else:
            cat = "capital"
            dir_val = -1 if 'offering' in t else 1
            impact_val = 1
            horizon = "week"

    # Clinical/FDA (medical/pharma)
    elif any(w in t for w in ['fda', 'approval', 'clinical', 'trial', 'pharma', 'drug', 'vaccine', 'biosimilar']):
        cat = "clinical"
        dir_val = 1 if 'approval' in t or 'fda' in t else 0
        impact_val = 3 if 'approval' in t or 'fda' in t else 2
        horizon = "3mo"

    # Tariff/macro
    elif 'tariff' in t or 'trade deal' in t or 'trade' in t and 'hit' in t:
        cat = "macro"
        dir_val = -1 if 'tariff' in t or any(w in t for w in ['hit', 'impact', 'threatens']) else 0
        impact_val = 2
        horizon = "week"

    # Generic market moves (noise)
    elif any(w in t for w in ['stocks', 'rally', 'decline', 'sell-off', 'recovery', 'market', 'share price']) and not any(w in t for w in ['announce', 'report', 'beat', 'miss']):
        cat = "macro"
        impact_val = 0
        horizon = "day"

    # List articles (noise)
    elif any(w in t for w in ['roundup', '3 stocks', 'stocks to watch', 'best stocks', 'top stocks', 'trending', 'highlights', 'watch', 'to watch', 'recap']):
        cat = "other"
        impact_val = 0
        horizon = "day"

    # Default fallback
    if cat is None:
        cat = "other"
        impact_val = 0
        horizon = "day"

    # Refine direction if not set
    if dir_val == 0 and impact_val > 0:
        if any(w in t for w in ['record', 'best', 'strong', 'surge', 'jump', 'award', 'exceed', 'growth', 'boost']):
            dir_val = 1
        elif any(w in t for w in ['decline', 'down', 'loss', 'weak', 'threat', 'risk', 'drop', 'slips', 'slump']):
            dir_val = -1

    return {
        "dir": dir_val,
        "impact": impact_val,
        "cat": cat,
        "horizon": horizon
    }

# Load and classify
with open(r'C:\workspace\rotation\.data\news_batches\batch_000.json', 'r', encoding='utf-8') as f:
    batch = json.load(f)

results = []
for item in batch:
    classification = classify_headline(item['title'], item['ticker'])
    results.append({
        "id": item['id'],
        "dir": classification['dir'],
        "impact": classification['impact'],
        "cat": classification['cat'],
        "horizon": classification['horizon']
    })

# Write results
with open(r'C:\workspace\rotation\.data\news_batches\rated_000.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, separators=(',', ':'), ensure_ascii=True)

# Count impact distribution
impact_counts = {0: 0, 1: 0, 2: 0, 3: 0}
for r in results:
    impact_counts[r['impact']] += 1

print(f"400 items; impact 0:{impact_counts[0]} 1:{impact_counts[1]} 2:{impact_counts[2]} 3:{impact_counts[3]}")
