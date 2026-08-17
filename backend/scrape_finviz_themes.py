# -*- coding: utf-8 -*-
"""Scrape the Finviz THEMES map (finviz.com/map?t=themes) with Playwright. The treemap is JS-rendered and
pulls its data from a backend endpoint; we intercept every network response and keep the JSON payload(s)
that contain the theme -> subtheme -> ticker tree. Dumps raw payloads to .data/finviz_themes_raw.json and,
if the tree is recognizable, a normalized .data/finviz_themes.json.

Run in the celery worker (has egress + chromium):
  MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/scrape_finviz_themes.py
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

RAW = Path("/app/.data/finviz_themes_raw.json")
OUT = Path("/app/.data/finviz_themes.json")
URL = "https://finviz.com/map?t=themes"


def main():
    captured = []

    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        pg = b.new_page(user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"))

        def on_resp(resp):
            try:
                u = resp.url
                if "finviz.com" not in u:
                    return
                ct = (resp.headers or {}).get("content-type", "")
                if "json" in ct or "javascript" in ct or "ashx" in u or u.endswith(".json") or "/api/" in u or "/maps" in u:
                    body = resp.text()
                    if body and len(body) > 40:
                        captured.append({"url": u, "body": body, "ct": ct})
            except Exception:
                pass

        pg.on("response", on_resp)
        pg.goto(URL, wait_until="domcontentloaded", timeout=45000)
        pg.wait_for_timeout(7000)
        # directly probe candidate STRUCTURE endpoints from the page context (carries finviz cookies/headers)
        for cand in ["/api/map?t=themes", "/maps/themes.json", "/api/map_perf?t=themes&st=d1",
                     "/api/map?t=themes&st=d1", "/maps/sec_all.json"]:
            try:
                txt = pg.evaluate("u => fetch(u).then(r => r.ok ? r.text() : ('ERR'+r.status))",
                                  "https://finviz.com" + cand)
                if txt and not txt.startswith("ERR") and len(txt) > 40:
                    captured.append({"url": "PROBE " + cand, "body": txt, "ct": "probe"})
                    print(f"probe OK {cand}: {len(txt)}B", flush=True)
                else:
                    print(f"probe {cand}: {str(txt)[:20]}", flush=True)
            except Exception as e:
                print(f"probe {cand} failed: {e}", flush=True)
        # also try to lift any big theme tree the app stashed on window
        try:
            win = pg.evaluate("""() => {
              const out = {};
              for (const k in window) { try {
                const v = window[k];
                if (v && typeof v === 'object') { const s = JSON.stringify(v);
                  if (s && s.length > 2000 && (s.includes('children') || s.includes('ticker') || s.includes('nodes'))) out[k] = JSON.parse(s); }
              } catch(e){} }
              return out;
            }""")
            if win:
                captured.append({"url": "window://globals", "body": json.dumps(win), "ct": "window"})
        except Exception as e:
            print("window scan failed:", e, flush=True)
        b.close()

    print(f"captured {len(captured)} JSON-ish payloads", flush=True)
    for c in captured:
        print(f"  {len(c['body']):>8}B  {c['url'][:110]}", flush=True)
    RAW.write_text(json.dumps(captured)[:5_000_000], encoding="utf-8")
    print(f"raw -> {RAW}", flush=True)

    # try to normalize: find the payload that parses into a nested name/children tree with tickers
    best = None
    for c in captured:
        try:
            j = json.loads(c["body"])
        except Exception:
            continue
        s = json.dumps(j)
        if ("children" in s or "nodes" in s) and ("ticker" in s.lower() or '"name"' in s):
            best = j
            break
    if best is not None:
        OUT.write_text(json.dumps(best), encoding="utf-8")
        print(f"normalized tree -> {OUT}", flush=True)
    else:
        print("no obvious nested tree found — inspect the raw payloads", flush=True)
    print("DONE_FINVIZ", flush=True)


if __name__ == "__main__":
    main()
