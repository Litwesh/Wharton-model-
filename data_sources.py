from __future__ import annotations
import os, re, time
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache'; CACHE.mkdir(exist_ok=True)
load_dotenv(ROOT/'.env')

class DataHub:
    def __init__(self):
        self.av_key=os.getenv('ALPHAVANTAGE_API_KEY','').strip()

    @staticmethod
    def symbols(text):
        raw=re.split(r'[\s,;]+', str(text).upper())
        return list(dict.fromkeys(x for x in raw if re.match(r'^[A-Z0-9.^=-]{1,16}$',x)))

    def price(self,symbol,period='10y'):
        pth=CACHE/f'{symbol}_yf.csv'
        try:
            p=yf.Ticker(symbol).history(period=period,interval='1d',auto_adjust=True)
            if p is not None and not p.empty:
                p=p[~p.index.duplicated(keep='last')]
                p.to_csv(pth); return p
        except Exception: pass
        if self.av_key:
            try:
                r=requests.get('https://www.alphavantage.co/query',params={'function':'TIME_SERIES_DAILY_ADJUSTED','symbol':symbol,'outputsize':'full','apikey':self.av_key},timeout=20)
                ts=r.json().get('Time Series (Daily)',{})
                if ts:
                    rows=[]
                    for d,v in ts.items():
                        rows.append([pd.Timestamp(d),float(v['1. open']),float(v['2. high']),float(v['3. low']),float(v['4. close']),float(v['6. volume'])])
                    p=pd.DataFrame(rows,columns=['Date','Open','High','Low','Close','Volume']).set_index('Date').sort_index()
                    p.to_csv(CACHE/f'{symbol}_av.csv'); return p
            except Exception: pass
        if pth.exists():
            try: return pd.read_csv(pth,index_col=0,parse_dates=True)
            except Exception: pass
        return pd.DataFrame()

    @staticmethod
    def _f(x):
        try:
            v=float(x); return v if np.isfinite(v) else np.nan
        except Exception: return np.nan

    def fundamentals(self,symbol):
        try: t=yf.Ticker(symbol); info=t.get_info()
        except Exception: t=None; info={}
        av={}
        if self.av_key:
            try:
                av=requests.get('https://www.alphavantage.co/query',params={'function':'OVERVIEW','symbol':symbol,'apikey':self.av_key},timeout=20).json()
            except Exception: av={}
        def pick(a,b=None):
            v=info.get(a)
            return v if v not in (None,'') else av.get(b) if b else None
        out={
            'Name':pick('longName') or pick('shortName') or symbol,
            'AssetClass':self.asset_class(info,av),'Sector':pick('sector','Sector') or '',
            'Industry':pick('industry','Industry') or '', 'MarketCap':self._f(pick('marketCap','MarketCapitalization')),
            'ForwardPE':self._f(pick('forwardPE','ForwardPE')),'TrailingPE':self._f(pick('trailingPE','PERatio')),
            'PEG':self._f(pick('pegRatio','PEGRatio')),'EVEBITDA':self._f(pick('enterpriseToEbitda','EVToEBITDA')),
            'ROE':self._f(pick('returnOnEquity','ReturnOnEquityTTM')),'OperatingMargin':self._f(pick('operatingMargins','OperatingMarginTTM')),
            'ProfitMargin':self._f(pick('profitMargins','ProfitMargin')),'DebtEquity':self._f(pick('debtToEquity','DebtToEquity')),
            'CurrentRatio':self._f(pick('currentRatio','CurrentRatio')),'RevenueGrowth':self._f(pick('revenueGrowth','QuarterlyRevenueGrowthYOY')),
            'EarningsGrowth':self._f(pick('earningsGrowth','QuarterlyEarningsGrowthYOY')),'Beta':self._f(pick('beta','Beta')),
            'DividendYield':self._f(pick('dividendYield','DividendYield')),'EPS':self._f(pick('trailingEps','DilutedEPS')),
            'FCF':self._f(pick('freeCashflow')),'BusinessSummary':pick('longBusinessSummary') or '',
            'Website':pick('website') or '',
        }
        if t is not None:
            try: inc=t.ttm_income_stmt
            except Exception: inc=pd.DataFrame()
            try: bs=t.ttm_balance_sheet
            except Exception: bs=pd.DataFrame()
            try: cf=t.ttm_cashflow
            except Exception: cf=pd.DataFrame()
        else: inc=bs=cf=pd.DataFrame()
        def row(df,names):
            if df.empty:return np.nan
            for n in names:
                if n in df.index:
                    s=df.loc[n].dropna()
                    if len(s): return self._f(s.iloc[0])
            return np.nan
        rev=row(inc,['TotalRevenue','OperatingRevenue']); net=row(inc,['NetIncome','NetIncomeCommonStockholders'])
        op=row(inc,['OperatingIncome']); ocf=row(cf,['OperatingCashFlow','CashFlowFromContinuingOperatingActivities']); capex=row(cf,['CapitalExpenditure','CapitalExpenditures'])
        if not np.isfinite(out['FCF']): out['FCF']=ocf+capex if np.isfinite(ocf) and np.isfinite(capex) else np.nan
        if not np.isfinite(out['OperatingMargin']) and np.isfinite(op) and np.isfinite(rev) and rev: out['OperatingMargin']=op/rev
        if not np.isfinite(out['ProfitMargin']) and np.isfinite(net) and np.isfinite(rev) and rev: out['ProfitMargin']=net/rev
        out['FCFMargin']=out['FCF']/rev if np.isfinite(out['FCF']) and np.isfinite(rev) and rev else np.nan
        out['FCFYield']=out['FCF']/out['MarketCap'] if np.isfinite(out['FCF']) and np.isfinite(out['MarketCap']) and out['MarketCap'] else np.nan
        out['EarningsQuality']=ocf/net if np.isfinite(ocf) and np.isfinite(net) and net else np.nan
        available=sum(pd.notna(v) for v in out.values())
        out['DataQualityScore']=min(100.,40.+4.*available)
        return out

    @staticmethod
    def asset_class(info,av):
        qt=str(info.get('quoteType','')).upper(); text=f"{info.get('category','')} {info.get('industry','')} {info.get('longName','')}".lower()
        if 'ETF' in qt or av.get('AssetType')=='ETF': return 'ETF'
        if 'BOND' in qt or any(x in text for x in ['bond','treasury','fixed income']): return 'Bond / Fixed-Income'
        if qt in ('EQUITY','STOCK') or av.get('AssetType')=='Common Stock': return 'Stock'
        return qt.title() if qt else 'Other'

    def health(self):
        return [
            {'Source':'yfinance','Enabled':True,'Role':'Prices, statements, company information, ETF data'},
            {'Source':'Alpha Vantage','Enabled':bool(self.av_key),'Role':'Secondary prices/fundamentals; add ALPHAVANTAGE_API_KEY to .env'},
        ]
