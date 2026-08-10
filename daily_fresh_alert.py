#!/usr/bin/env python3
"""
Daily "fresh" watcher.

Runs the sector scan, finds which sectors are FRESH today
(Sortino>0 + RSI crossover + RSI-of-Sortino crossover + green MACD within 14d),
and alerts on any that became fresh SINCE THE LAST RUN ("suddenly fresh").

State is persisted to .data/fresh_state.json so each run only alerts on the
delta. Alert channels are pluggable via env vars (see _send_alert):
  - always: prints to stdout and appends to .data/fresh_alerts.log
  - FRESH_ALERT_WEBHOOK   : POST JSON to a Slack/Discord/generic incoming webhook
  - FRESH_ALERT_EMAIL_TO  : send email via SMTP (needs FRESH_SMTP_* vars below)

Exit code 0 = ran fine (regardless of whether anything was fresh).

Usage:
    python daily_fresh_alert.py
"""
import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
from datetime import datetime, date
from pathlib import Path

import trend_analyzer

STATE_DIR = Path(__file__).parent / ".data"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "fresh_state.json"
ALERT_LOG = STATE_DIR / "fresh_alerts.log"


def _load_prev_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _fresh_map(results):
    """etf -> details for sectors currently FRESH."""
    out = {}
    for r in results:
        if r.get("fresh_state") == "FRESH":
            out[r["etf"]] = {
                "sector": r["sector"],
                "fresh_days": r.get("fresh_days"),
                "fresh_since": r.get("fresh_since"),
                "macd_great": r.get("macd_great", False),
                "rsi": r.get("rsi"),
                "omega": r.get("omega"),
                "sortino": r.get("sortino"),
                "signal": r.get("signal"),
            }
    return out


def _send_alert(subject, body):
    """Print + log always; optionally webhook and/or email."""
    print(subject)
    print(body)
    with open(ALERT_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} =====\n{subject}\n{body}\n")

    # Optional: generic/Slack webhook
    hook = os.environ.get("FRESH_ALERT_WEBHOOK")
    if hook:
        try:
            import urllib.request
            payload = json.dumps({"text": f"*{subject}*\n{body}"}).encode()
            req = urllib.request.Request(hook, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"[webhook failed] {e}")

    # Optional: email via SMTP
    to = os.environ.get("FRESH_ALERT_EMAIL_TO")
    if to:
        try:
            import smtplib
            from email.mime.text import MIMEText
            host = os.environ.get("FRESH_SMTP_HOST", "localhost")
            port = int(os.environ.get("FRESH_SMTP_PORT", "587"))
            user = os.environ.get("FRESH_SMTP_USER")
            pwd = os.environ.get("FRESH_SMTP_PASS")
            sender = os.environ.get("FRESH_SMTP_FROM", user or "fresh-bot@localhost")
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = to
            with smtplib.SMTP(host, port, timeout=15) as s:
                s.starttls()
                if user and pwd:
                    s.login(user, pwd)
                s.sendmail(sender, [t.strip() for t in to.split(",")], msg.as_string())
        except Exception as e:
            print(f"[email failed] {e}")


def main():
    results = trend_analyzer.analyze()
    current = _fresh_map(results)

    prev = _load_prev_state()
    prev_fresh = set(prev.get("fresh", {}).keys())
    cur_fresh = set(current.keys())

    newly = sorted(cur_fresh - prev_fresh, key=lambda e: current[e].get("fresh_days") or 99)
    dropped = sorted(prev_fresh - cur_fresh)

    today = date.today().isoformat()

    if newly:
        lines = []
        for etf in newly:
            d = current[etf]
            star = " *(MACD>0)*" if d["macd_great"] else ""
            lines.append(
                f"  • {d['sector']} ({etf}) — fresh since {d['fresh_since']} "
                f"({d['fresh_days']}d ago){star}; Sortino {d['sortino']}, Omega {d['omega']}, signal {d['signal']}"
            )
        subject = f"[Sector Rotation] {len(newly)} newly FRESH ({today})"
        body = "Suddenly fresh since last run:\n" + "\n".join(lines)
        if dropped:
            body += "\n\nNo longer fresh: " + ", ".join(
                f"{prev['fresh'][e]['sector']} ({e})" for e in dropped if e in prev.get("fresh", {})
            )
        _send_alert(subject, body)
    else:
        print(f"[{today}] No newly fresh sectors. Currently fresh: {len(cur_fresh)}"
              + (f"; dropped: {len(dropped)}" if dropped else ""))

    _save_state({
        "as_of": today,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "fresh": current,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
