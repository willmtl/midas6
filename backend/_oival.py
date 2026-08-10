import django, os, datetime as dt, time
os.environ.setdefault("DJANGO_SETTINGS_MODULE","rotation.settings"); django.setup()
import numpy as np, pandas as pd
from seq_fundamental_study import build_universe, load_candles, load_financial_reports
from studies import SIGNALS
from all_on_all_study import _prepare_indicators
from pit_fundamentals import _ad_state
from core.models import Fundamental, OptionSnapshot
from api.tasks import GICS2ETF
from thetadata import ThetaClient
TC=ThetaClient(api_key=os.environ['THETADATA_API_KEY'])
# existing polygon_hist (pc_vol, iv) by (ticker,date)
poly={}
for o in OptionSnapshot.objects.filter(source='polygon_hist').values('ticker','date','atm_iv','pc_vol'):
    poly[(o['ticker'],o['date'])]=(o['atm_iv'],o['pc_vol'])
# generate Mode A/B events (4yr) with polygon data + fwd return
tks=build_universe()
sec={r["ticker"]:r["sector"] for r in Fundamental.objects.filter(ticker__in=tks).values("ticker","sector")}
t2etf={t:GICS2ETF[s] for t,s in sec.items() if s in GICS2ETF}
reports=load_financial_reports(tks)
mkt=load_candles(sorted(set(t2etf.values()))+["SPY"]); spy=mkt["SPY"]["Close"]; spy63=spy.pct_change(63)
H=63; W0=pd.Timestamp("2022-09-08"); W1=pd.Timestamp("2026-05-01")
events=[]
cd=load_candles([t for t in tks if t in t2etf])
for tk,df in cd.items():
    if len(df)<320: continue
    rep=reports.get(tk)
    if rep is None or not len(rep): continue
    r2=rep.dropna(subset=["avail_date","shares_outstanding"]).sort_values("avail_date")
    if not len(r2): continue
    pdd=pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]).to_numpy()); sh=r2["shares_outstanding"].to_numpy(float)
    _prepare_indicators(df)
    c=df["Close"].values; n=len(c); idx=df.index; rsi=df["_rsi"].values; st=_ad_state(df).values
    sma=pd.Series(c).rolling(200).mean().values
    capA=((SIGNALS["new_52low"][1](df).fillna(False).values|SIGNALS["rsi_oversold20"][1](df).fillna(False).values)&(st==2))
    B=(rsi<30)&(c>sma); e63=mkt[t2etf[tk]]["Close"].pct_change(63); last=-99
    for i in range(252,n-2):
        mode="A" if capA[i] else ("B" if B[i] else None)
        if not mode: continue
        if i-last<=10: last=i; continue
        last=i
        d=idx[i]; dd=d.date() if hasattr(d,'date') else d
        if d<W0 or d>W1 or c[i]<5 or i+H>=n: continue
        j=int(pdd.searchsorted(d,"right"))-1
        if j<0 or np.isnan(sh[j]) or sh[j]*c[i]<300e6: continue
        rv=e63.asof(d); sv=spy63.asof(d)
        if not (pd.notna(rv) and pd.notna(sv) and rv>sv): continue
        if (tk,dd) not in poly: continue  # only where we have volume/IV to compare
        events.append((mode,(c[i+H]/c[i]-1)*100, tk, dd, poly[(tk,dd)][1]))  # fwd, ticker, date, pc_vol
print('events to price OI for:', len(events))
# cache expirations per ticker
expcache={}
def get_exps(sym):
    if sym in expcache: return expcache[sym]
    try:
        e=TC.option_list_expirations(sym); ed=e.to_pandas() if hasattr(e,'to_pandas') else e
        col='expiration' if 'expiration' in ed.columns else ed.columns[-1]
        out=[]
        for x in ed[col].tolist():
            if isinstance(x,str):
                try: out.append(dt.date.fromisoformat(x[:10]))
                except Exception: pass
            elif hasattr(x,'date'): out.append(x.date())
            elif isinstance(x,dt.date): out.append(x)
        expcache[sym]=sorted(set(out))
    except Exception:
        expcache[sym]=[]
    return expcache[sym]
def pc_oi(sym, d):
    exps=get_exps(sym)
    near=[x for x in exps if isinstance(x,dt.date) and d < x <= d+dt.timedelta(days=60)][:6]
    coi=poii=0
    for e2 in near:
        try:
            df=TC.option_history_open_interest(symbol=sym, expiration=e2, strike='*', right='both', start_date=d, end_date=d)
            dd=df.to_pandas() if hasattr(df,'to_pandas') else df
            coi+=int(dd[dd['right']=='CALL']['open_interest'].sum()); poii+=int(dd[dd['right']=='PUT']['open_interest'].sum())
        except Exception: pass
    return (poii/coi) if coi else None
# pull pc_oi + store
done=0
res=[]  # (mode, fwd, pc_vol, pc_oi)
for mode,fwd,tk,dd,pcv in events:
    po=pc_oi(tk,dd)
    if po is not None:
        OptionSnapshot.objects.filter(ticker=tk,date=dd).update(pc_oi=round(po,3))
        res.append((mode,fwd,pcv,po))
    done+=1
    if done%50==0: print('  priced',done,'/',len(events),flush=True)
    time.sleep(0.03)
print('priced with OI:', len(res))
def stat(sub):
    if len(sub)<12: return None
    a=np.array([e[1] for e in sub]); return (len(a),round(np.median(a),1),round((a>0).mean()*100),round((a<-20).mean()*100))
def terc(sub, mi, lab):
    vals=sorted(e[mi] for e in sub if e[mi] is not None)
    if len(vals)<24: print('    too few'); return
    lo,hi=vals[len(vals)//3],vals[2*len(vals)//3]
    for nm,g in [('LOW ',[e for e in sub if e[mi] is not None and e[mi]<=lo]),('MID ',[e for e in sub if e[mi] is not None and lo<e[mi]<hi]),('HIGH',[e for e in sub if e[mi] is not None and e[mi]>=hi])]:
        s=stat(g)
        if s: print('      %s %s: n=%d med=%+.1f%% win=%d%% dis20=%d%%'%(lab,nm,s[0],s[1],s[2],s[3]))
for mode in ('A','B'):
    sub=[e for e in res if e[0]==mode]
    print('\n=== Mode %s (n=%d) — OI put/call vs VOLUME put/call ==='%(mode,len(sub)))
    print('  put/call OPEN INTEREST (contrarian: high=fear):'); terc(sub,3,'pc_OI')
    print('  put/call VOLUME (for comparison):'); terc(sub,2,'pc_VOL')
