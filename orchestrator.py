from __future__ import annotations
import numpy as np,pandas as pd
from data_sources import DataHub
from indicators import make_feature_frame,TECHNICAL_FEATURES
from agents import FundamentalAgent,TechnicalAgent,QuantAgent,PortfolioAgent,ClientFitAgent

CLIENT={'initial_2027':300000,'contribution_2028':150000,'payment':50000,'payment_years':list(range(2033,2043))}

def parse_holdings(text):
    out={}
    for line in str(text).splitlines():
        p=[x.strip() for x in line.split(',')]
        try:out[p[0].upper()]={'shares':float(p[1]),'cost_basis':float(p[2]) if len(p)>2 and p[2] else np.nan}
        except Exception:pass
    return out

def analyze_universe(universe,horizon=20,min_history=750,force_update=False):
    hub=DataHub(); stocks=hub.symbols(universe); bench=hub.price('SPY','10y'); price_map={};feat_map={};raw={}
    for s in stocks:
        p=hub.price(s,'10y')
        if p.empty or len(p)<min_history:continue
        f=make_feature_frame(p,bench); price_map[s]=p;feat_map[s]=f;raw[s]=hub.fundamentals(s)
    if not raw:return pd.DataFrame(),{}
    fa=FundamentalAgent();fs=fa.score(pd.DataFrame(raw).T); ta=TechnicalAgent();qa=QuantAgent();ca=ClientFitAgent(CLIENT); rows=[];detail={}
    for s in price_map:
        f=feat_map[s]; model=qa.fit_or_load(s,f,horizon,force_update); pred=qa.predict(model,f); tw=model.get('indicator_weights',qa.indicator_weights(f,horizon)); ts=ta.score(f,tw); fund=fs.loc[s].to_dict(); fit=ca.evaluate(raw[s],fund,f); asset=raw[s].get('AssetClass','Other')
        fundamental=np.nan if asset!='Stock' else np.nanmean([fund.get('Quality',np.nan),fund.get('Valuation',np.nan),fund.get('Growth',np.nan)])
        overall=.30*pred['model_score']+.24*ts['score']+.14*pred['risk_adjusted_score']+.20*(fundamental if np.isfinite(fundamental) else 50)+.12*fit['score']
        chart=f[['Close']].copy();chart['SMA50']=f.Close.rolling(50).mean();chart['SMA200']=f.Close.rolling(200).mean();
        row={'Stock':s,'Name':raw[s].get('Name',s),'AssetClass':asset,'OverallScore':float(np.clip(overall,0,100)),'ModelProbabilityUp':pred['probability_up'],'ExpectedReturn':pred['expected_return'],'ExpectedVolatility':pred['expected_volatility'],'Quality':fund.get('Quality',np.nan) if asset=='Stock' else np.nan,'Valuation':fund.get('Valuation',np.nan) if asset=='Stock' else np.nan,'Growth':fund.get('Growth',np.nan) if asset=='Stock' else np.nan,'TechnicalScore':ts['score'],'RiskScore':pred['risk_score'],'ClientFit':fit['score'],'OOSAccuracy':pred['oos_accuracy'],'OOSAUC':pred['oos_auc'],'OOSBrier':pred['oos_brier'],'PredictionCoverage':pred['prediction_coverage'],'DataQualityScore':raw[s].get('DataQualityScore',50),'DataDate':str(f.index[-1].date())}
        rows.append(row); techrows=[{'Indicator':k,'Value':float(f[k].iloc[-1]) if np.isfinite(f[k].iloc[-1]) else np.nan,'LearnedWeight':tw.get(k,0)} for k in TECHNICAL_FEATURES]
        detail[s]={**row,'ExecutiveSummary':f"{fit['headline']} Quantitative model probability of a positive {horizon}-day return: {pred['probability_up']:.1%}.",'FundamentalMetrics':fa.display_metrics(s,fund,raw[s]),'FundamentalNarrative':fa.narrative(fund,raw[s]),'TechnicalNarrative':ta.narrative(ts,f),'TechnicalIndicators':techrows,'ClientFitDetails':fit['details'],'PriceChart':chart,'RawFeatures':f,'Model':model}
    return pd.DataFrame(rows).sort_values(['OverallScore','ModelProbabilityUp'],ascending=False).reset_index(drop=True),detail

def portfolio_report(h,rf=.04):return PortfolioAgent().report(h,rf)
def efficient_frontier(h,rf=.04):return PortfolioAgent().frontier(h,rf)
def run_walk_forward_backtest(universe,horizon=20,min_history=750):
    hub=DataHub(); bench=hub.price('SPY','10y'); fmap={}
    for s in hub.symbols(universe):
        p=hub.price(s,'10y')
        if not p.empty and len(p)>=min_history:fmap[s]=make_feature_frame(p,bench)
    return QuantAgent().full_backtest(fmap,horizon)
def data_health():return DataHub().health()
