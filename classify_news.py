import json
import re

def classify_headline(headline):
    title = headline['title'].lower()
    direction = 0
    impact = 0
    category = 'other'
    horizon = 'day'

    if re.search(r'earnings (beat|exceed|top|surpass)', title):
        direction = 1; impact = 3; category = 'earnings_beat'; horizon = '3mo'
    elif re.search(r'(beat|exceed|top|surpass).*estimates', title):
        direction = 1; impact = 3; category = 'earnings_beat'; horizon = '3mo'
    elif re.search(r'earnings (miss|lag)', title) or re.search(r'miss.*estimates', title):
        direction = -1; impact = 3; category = 'earnings_miss'; horizon = '3mo'
    elif re.search(r'\b(q[1-4]|fourth quarter).*(earnings|results|report)', title):
        if any(w in title for w in ['beat','surpass','top','exceed']):
            direction = 1; impact = 2; category = 'earnings_beat'
        elif any(w in title for w in ['miss','lag','fall']):
            direction = -1; impact = 2; category = 'earnings_miss'
        else:
            impact = 1; category = 'earnings_beat'
        horizon = '3mo'
    elif 'earnings' in title and 'call highlight' in title:
        impact = 1; category = 'earnings_beat'; horizon = 'month'
    elif re.search(r'earnings.*(preview|expected|set to|gears|what.*expect|what.*look)', title):
        impact = 0; category = 'earnings_beat'; horizon = 'day'

    elif re.search(r'guidance.*(raise|hike|increase|boost|improve)', title):
        direction = 1; impact = 2; category = 'guidance_up'; horizon = '3mo'
    elif re.search(r'guidance.*(cut|lower|reduce)', title):
        direction = -1; impact = 2; category = 'guidance_down'; horizon = '3mo'
    elif re.search(r'outlook.*(raise|improve|boost|positive|growth)', title):
        direction = 1; impact = 2; category = 'guidance_up'; horizon = '3mo'
    elif re.search(r'outlook.*(cut|lower|negative|weak)', title):
        direction = -1; impact = 2; category = 'guidance_down'; horizon = '3mo'

    elif re.search(r'upgrade|upgraded|initiat.*outperform|pounding the table', title):
        direction = 1; impact = 2; category = 'upgrade'; horizon = 'week'
    elif re.search(r'downgrade|downgraded', title):
        direction = -1; impact = 2; category = 'downgrade'; horizon = 'week'
    elif re.search(r'initiates coverage', title):
        direction = 1; impact = 2; category = 'upgrade'; horizon = 'week'

    elif re.search(r'(acquisition|acquire|merger|merge|buyout|takeover)', title):
        if any(w in title for w in ['closes','closed','complete','successfully']):
            direction = 1 if 'success' in title else 0; impact = 3; category = 'ma'; horizon = 'day'
        else:
            impact = 3 if 'billion' in title else 2; category = 'ma'; horizon = 'month'

    elif re.search(r'(product launch|launches.*product|expand.*availability|new.*product)', title):
        direction = 1; impact = 2; category = 'product'; horizon = 'month'
    elif re.search(r'partner|partnership|collaboration|extend.*partnership', title):
        direction = 1 if 'extend' in title else 0; impact = 2 if 'multi' in title else 1; category = 'contract'; horizon = 'month'

    elif re.search(r'(class action|lawsuit|fraud|investigation|bankruptcy)', title):
        direction = -1; impact = 3 if 'fraud' in title or 'bankruptcy' in title else 2; category = 'legal'; horizon = 'month'
    elif re.search(r'(strike|union|labor)', title):
        direction = -1; impact = 2; category = 'legal'; horizon = 'month'

    elif re.search(r'(appoint|new ceo|ceo.*succession|board.*appoint)', title):
        direction = 0; impact = 2; category = 'mgmt'; horizon = '3mo'

    elif re.search(r'(dividend|share repurchase|buyback)', title):
        if 'buyback' in title or 'repurchase' in title:
            direction = 1; impact = 1; category = 'capital'
        else:
            direction = 1; impact = 1; category = 'dividend'
        horizon = 'week'
    elif re.search(r'(share offering|secondary|raises.*million|raised.*million)', title):
        direction = -1 if 'secondary' in title else 0; impact = 2 if 'billion' in title else 1; category = 'capital'; horizon = 'week'

    elif re.search(r'(fda.*approv|approval|approved)', title):
        direction = 1; impact = 3; category = 'clinical'; horizon = '3mo'
    elif re.search(r'(clinical|trial|fda)', title):
        direction = 0; impact = 1; category = 'clinical'; horizon = '3mo'

    elif re.search(r'(stock market|market futures|dow|s&p 500|nasdaq|tariff|iran|war|oil prices)', title):
        direction = 1 if re.search(r'(gain|rally|pop|surge|soar|boost)', title) else (-1 if re.search(r'(fall|plunge|drop|decline|slip)', title) else 0)
        impact = 0; category = 'macro'; horizon = 'day'

    elif re.search(r'(\d+.*stock|stocks to buy|stocks to watch|best.*stock|top.*stock)', title):
        impact = 0; category = 'other'; horizon = 'day'
    elif re.search(r'(valuation|undervalue|overvalue|fair value)', title):
        direction = 1 if 'upside' in title or 'undervalue' in title else (-1 if 'downside' in title else 0); impact = 0; category = 'other'; horizon = 'day'
    elif re.search(r'(bull case|bear case)', title):
        direction = 1 if 'bull' in title else -1; impact = 1; category = 'other'; horizon = 'month'

    return {'id': headline['id'], 'dir': direction, 'impact': impact, 'cat': category, 'horizon': horizon}

with open('.data/news_batches/batch_028.json', 'r', encoding='utf-8') as f:
    headlines = json.load(f)

classified = [classify_headline(h) for h in headlines]

with open('.data/news_batches/rated_028.json', 'w', encoding='utf-8') as f:
    json.dump(classified, f, separators=(',', ':'), ensure_ascii=False)

impact_dist = {0: 0, 1: 0, 2: 0, 3: 0}
for c in classified:
    impact_dist[c['impact']] += 1

print(f"{len(classified)} items; impact 0:{impact_dist[0]} 1:{impact_dist[1]} 2:{impact_dist[2]} 3:{impact_dist[3]}")
