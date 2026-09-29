from __future__ import annotations
import numpy as np, pandas as pd

def rsi(s,n):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); ad=dn.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return 100-100/(1+au/ad.replace(0,np.nan))

def atr(df,n):
    p=df['Close'].shift(1); tr=pd.concat([df['High']-df['Low'],(df['High']-p).abs(),(df['Low']-p).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def macd(s):
    e12=s.ewm(span=12,adjust=False).mean(); e26=s.ewm(span=26,adjust=False).mean(); line=e12-e26; sig=line.ewm(span=9,adjust=False).mean(); return line,sig,line-sig

def stochastic(df,n=14):
    lo=df['Low'].rolling(n).min(); hi=df['High'].rolling(n).max(); k=100*(df['Close']-lo)/(hi-lo).replace(0,np.nan); return k,k.rolling(3).mean()

def adx(df,n=14):
    up=df['High'].diff(); dn=-df['Low'].diff(); plus=np.where((up>dn)&(up>0),up,0); minus=np.where((dn>up)&(dn>0),dn,0); a=atr(df,n)
    p=100*pd.Series(plus,index=df.index).ewm(alpha=1/n,adjust=False,min_periods=n).mean()/a.replace(0,np.nan)
    m=100*pd.Series(minus,index=df.index).ewm(alpha=1/n,adjust=False,min_periods=n).mean()/a.replace(0,np.nan)
    dx=100*(p-m).abs()/(p+m).replace(0,np.nan); return dx.ewm(alpha=1/n,adjust=False,min_periods=n).mean(),p,m

def cci(df,n=20):
    tp=(df['High']+df['Low']+df['Close'])/3; ma=tp.rolling(n).mean(); md=tp.rolling(n).apply(lambda x:np.mean(np.abs(x-np.mean(x))),raw=True); return (tp-ma)/(0.015*md.replace(0,np.nan))

def mfi(df,n=14):
    tp=(df['High']+df['Low']+df['Close'])/3; flow=tp*df['Volume']; pos=np.where(tp.diff()>0,flow,0); neg=np.where(tp.diff()<0,flow,0)
    p=pd.Series(pos,index=df.index).rolling(n).sum(); q=pd.Series(neg,index=df.index).rolling(n).sum(); return 100-100/(1+p/q.replace(0,np.nan))

def cmf(df,n=20):
    mfm=((df['Close']-df['Low'])-(df['High']-df['Close']))/(df['High']-df['Low']).replace(0,np.nan); return (mfm*df['Volume']).rolling(n).sum()/df['Volume'].rolling(n).sum().replace(0,np.nan)

def zscore(s,n): return (s-s.rolling(n).mean())/s.rolling(n).std().replace(0,np.nan)
def slope(s,n):
    x=np.arange(n); return s.rolling(n).apply(lambda y:np.polyfit(x,y,1)[0] if np.isfinite(y).all() else np.nan,raw=True)

def make_feature_frame(df,bench=None):
    x=df.copy(); c=x['Close']; ret=c.pct_change()
    for n in [1,2,3,5,10,20,60,120,252]: x[f'ret{n}']=c.pct_change(n)
    for n in [5,10,20,30,60,120]: x[f'vol{n}']=ret.rolling(n).std()*np.sqrt(252)
    x['skew60']=ret.rolling(60).skew(); x['kurt60']=ret.rolling(60).kurt(); x['autocorr20']=ret.rolling(20).apply(lambda y:pd.Series(y).autocorr(),raw=False)
    x['rsi7']=rsi(c,7); x['rsi14']=rsi(c,14); x['rsi21']=rsi(c,21); x['stochK'],x['stochD']=stochastic(x,14); x['williamsR']=-100*(x['High'].rolling(14).max()-c)/(x['High'].rolling(14).max()-x['Low'].rolling(14).min()).replace(0,np.nan); x['cci20']=cci(x,20)
    for n in [5,10,20,60]: x[f'roc{n}']=c.pct_change(n)
    for n in [20,50,100,150,200]: x[f'sma{n}_dist']=c/c.rolling(n).mean()-1
    for n in [9,20,50]: x[f'ema{n}_dist']=c/c.ewm(span=n,adjust=False).mean()-1
    x['sma20_gt50']=(c.rolling(20).mean()>c.rolling(50).mean()).astype(int); x['sma50_gt200']=(c.rolling(50).mean()>c.rolling(200).mean()).astype(int)
    x['trend_slope20']=slope(np.log(c),20); x['trend_slope60']=slope(np.log(c),60)
    x['macd'],x['macd_signal'],x['macd_hist']=macd(c); e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); x['ppo']=(e12-e26)/e26.replace(0,np.nan)
    x['atr14_pct']=atr(x,14)/c; x['atr20_pct']=atr(x,20)/c; ma=c.rolling(20).mean(); sd=c.rolling(20).std(); x['bb_pct']=(c-(ma-2*sd))/(4*sd).replace(0,np.nan); x['bb_width']=4*sd/ma.replace(0,np.nan); kc=c.ewm(span=20,adjust=False).mean(); x['keltner_pct']=(c-(kc-2*atr(x,14)))/(4*atr(x,14)).replace(0,np.nan)
    x['adx14'],x['plusDI14'],x['minusDI14']=adx(x,14); vol=x['Volume'].replace(0,np.nan); x['volume_ratio20']=vol/vol.rolling(20).mean().replace(0,np.nan); x['volume_ratio60']=vol/vol.rolling(60).mean().replace(0,np.nan); x['volume_z20']=zscore(vol,20); obv=(np.sign(c.diff()).fillna(0)*vol.fillna(0)).cumsum(); x['obv_slope20']=slope(obv,20); x['obv_slope60']=slope(obv,60); x['mfi14']=mfi(x,14); x['cmf20']=cmf(x,20); x['vwap20_dist_pct']=(c-(c*vol).rolling(20).sum()/vol.rolling(20).sum().replace(0,np.nan))/c
    x['donchian20_pos']=(c-c.rolling(20).min())/(c.rolling(20).max()-c.rolling(20).min()).replace(0,np.nan); x['donchian55_pos']=(c-c.rolling(55).min())/(c.rolling(55).max()-c.rolling(55).min()).replace(0,np.nan); x['z20']=zscore(c,20); x['z60']=zscore(c,60); x['range_pct']=(x['High']-x['Low'])/c; x['gap']=x['Open']/x['Close'].shift(1)-1; peak=c.rolling(252,min_periods=60).max(); x['drawdown']=c/peak-1
    if bench is not None and not bench.empty:
        bm=bench['Close'].reindex(x.index).ffill(); br=bm.pct_change(); x['beta60']=ret.rolling(60).cov(br)/br.rolling(60).var().replace(0,np.nan); x['relative20']=c.pct_change(20)-bm.pct_change(20); x['relative60']=c.pct_change(60)-bm.pct_change(60)
    else: x['beta60']=np.nan; x['relative20']=x['ret20']; x['relative60']=x['ret60']
    x['regime_bull']=((c.rolling(50).mean()>c.rolling(200).mean())&(c>c.rolling(200).mean())).astype(int); x['regime_highvol']=(x['vol20']>x['vol60']).astype(int)
    return x

TECHNICAL_FEATURES=[
'ret1','ret2','ret3','ret5','ret10','ret20','ret60','ret120','ret252','vol5','vol10','vol20','vol30','vol60','vol120','skew60','kurt60','autocorr20','rsi7','rsi14','rsi21','stochK','stochD','williamsR','cci20','roc5','roc10','roc20','roc60','sma20_dist','sma50_dist','sma100_dist','sma150_dist','sma200_dist','ema9_dist','ema20_dist','ema50_dist','sma20_gt50','sma50_gt200','trend_slope20','trend_slope60','macd','macd_signal','macd_hist','ppo','atr14_pct','atr20_pct','bb_pct','bb_width','keltner_pct','adx14','plusDI14','minusDI14','volume_ratio20','volume_ratio60','volume_z20','obv_slope20','obv_slope60','mfi14','cmf20','vwap20_dist_pct','donchian20_pos','donchian55_pos','z20','z60','range_pct','gap','drawdown','beta60','relative20','relative60','regime_bull','regime_highvol']
