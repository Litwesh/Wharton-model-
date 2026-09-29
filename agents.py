from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, joblib
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor
from indicators import TECHNICAL_FEATURES
from data_sources import DataHub

ROOT = Path(__file__).resolve().parent; MODEL_DIR = ROOT / 'models'; MODEL_DIR.mkdir(exist_ok=True)

def safe(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except: return np.nan

def pct(s, higher=True):
    s = pd.to_numeric(s, errors='coerce')
    if s.notna().sum() < 2: return pd.Series(np.nan, index=s.index)
    r = s.rank(pct=True)
    return r if higher else 1 - r

class FundamentalAgent:
    SPECS = {
        'Quality': [('ROE', .24, 1), ('OperatingMargin', .15, 1), ('ProfitMargin', .12, 1), ('FCFMargin', .24, 1), ('EarningsQuality', .15, 1), ('CurrentRatio', .10, 1)],
        'Growth': [('RevenueGrowth', .55, 1), ('EarningsGrowth', .45, 1)],
        'Valuation': [('ForwardPE', .18, 0), ('TrailingPE', .12, 0), ('PEG', .15, 0), ('EVEBITDA', .15, 0), ('FCFYield', .40, 1)],
        'BalanceSheet': [('DebtEquity', .55, 0), ('CurrentRatio', .45, 1)]
    }
    
    def score(self, df):
        d = df.copy()
        for name, spec in self.SPECS.items():
            total = pd.Series(0., index=d.index)
            wsum = pd.Series(0., index=d.index)
            for col, w, high in spec:
                q = pct(d.get(col, pd.Series(np.nan, index=d.index)), bool(high))
                valid = q.notna()
                total += q.fillna(0) * w
                wsum += valid.astype(float) * w
            d[name] = np.where(wsum > 0, 100 * total / wsum, np.nan)
            d[name + 'Coverage'] = 100 * wsum / sum(w for _, w, _ in spec)
        return d
        
    def display_metrics(self, stock, fund, raw):
        out = []
        for name, spec in self.SPECS.items():
            if np.isfinite(fund.get(name, np.nan)):
                out.append({'Metric': f'{name} Score', 'Value': float(fund[name])})
        for k in ['ForwardPE', 'TrailingPE', 'PEG', 'EVEBITDA', 'CurrentRatio', 'ROE', 'OperatingMargin', 'ProfitMargin', 'FCFMargin', 'EarningsQuality', 'RevenueGrowth', 'EarningsGrowth', 'DebtEquity']:
            if np.isfinite(raw.get(k, np.nan)):
                out.append({'Metric': k, 'Value': float(raw[k])})
        return out

    def narrative(self, score, raw):
        vals = []
        for k, label in [('Quality', 'quality'), ('Valuation', 'valuation'), ('Growth', 'growth')]:
            if np.isfinite(score.get(k, np.nan)): vals.append(f'{label} peer score {score[k]:.1f}/100')
        for k, label in [('FCFMargin', 'FCF margin'), ('DebtEquity', 'debt/equity'), ('RevenueGrowth', 'revenue growth')]:
            if np.isfinite(raw.get(k, np.nan)): 
                vals.append(f'{label} {raw[k]:.1%}' if 'Growth' in k or 'Margin' in k else f'{label} {raw[k]:.2f}')
        return '; '.join(vals) if vals else 'Fundamental data insufficient.'

class TechnicalAgent:
    def indicator_table(self, feat, weights):
        z = feat.iloc[-1]
        return [{'Indicator': f, 'Value': safe(z.get(f)), 'LearnedWeight': weights.get(f, 0.)} for f in TECHNICAL_FEATURES]
        
    def score(self, feat, weights):
        z = feat.iloc[-1]; vals = []
        for f, w in weights.items():
            v = safe(z.get(f))
            if np.isfinite(v): vals.append(w * np.tanh(v / 2.5))
        learned = np.sum(vals) if vals else 0
        trend = np.mean([np.clip(50 + 250 * safe(z.get('sma50_dist', 0)), 0, 100), np.clip(50 + 250 * safe(z.get('sma200_dist', 0)), 0, 100), 100 if safe(z.get('sma50_gt200', 0)) == 1 else 0])
        mom = np.mean([np.clip(50 + 220 * safe(z.get('ret20', 0)), 0, 100), np.clip(50 + 150 * safe(z.get('relative20', 0)), 0, 100), np.clip(50 + 120 * safe(z.get('roc60', 0)), 0, 100)])
        risk = np.clip(100 - 100 * safe(z.get('vol20', .3)) - 60 * abs(safe(z.get('drawdown', 0))), 0, 100)
        score = np.clip(.40 * (50 + 45 * learned) + .25 * trend + .20 * mom + .15 * risk, 0, 100)
        return {'score': float(score), 'learned_component': float(np.clip(50 + 45 * learned, 0, 100)), 'trend': float(trend), 'momentum': float(mom), 'risk': float(risk)}
        
    def narrative(self, score, feat):
        z = feat.iloc[-1]
        return f"Learned technical composite {score['learned_component']:.1f}/100; trend {score['trend']:.1f}; momentum {score['momentum']:.1f}; risk {score['risk']:.1f}; RSI-14 {safe(z.get('rsi14')):.1f}; ADX-14 {safe(z.get('adx14')):.1f}; MACD histogram {safe(z.get('macd_hist')):.4g}."

class QuantAgent:
    def __init__(self): 
        self.dir = ROOT / 'models'
        
    def targets(self, x, h):
        z = x.copy(); z['yret'] = x['Close'].shift(-h) / x['Close'] - 1; z['yup'] = (z['yret'] > 0).astype(int); return z
        
    def indicator_weights(self, feat, h, end=None):
        z = self.targets(feat, h); z = z[z.index < end] if end is not None else z; z = z.replace([np.inf, -np.inf], np.nan).dropna(subset=['yret'])
        if len(z) < 350: return {f: 1 / len(TECHNICAL_FEATURES) for f in TECHNICAL_FEATURES}
        out = []
        for f in TECHNICAL_FEATURES:
            q = z[[f, 'yret']].dropna()
            if len(q) < 250: continue
            split_idx = np.array_split(np.arange(len(q)), 10)
            ics = []
            for idx in split_idx:
                if len(idx) >= 30:
                    b = q.iloc[idx]
                    ic = spearmanr(np.asarray(b[f], dtype=float), np.asarray(b['yret'], dtype=float), nan_policy='omit').statistic
                    if np.isfinite(ic): ics.append(float(ic))
            if not ics: continue
            m = np.mean(ics); s = np.std(ics, ddof=1) if len(ics) > 1 else .25; w = m / (1 + s) * min(max(abs(m) / (s + .03), 0), 1)
            out.append((f, w, m, s))
        if not out: return {f: 1 / len(TECHNICAL_FEATURES) for f in TECHNICAL_FEATURES}
        d = pd.DataFrame(out, columns=['Indicator', 'Weight', 'MeanIC', 'ICStd'])
        cap = d.Weight.abs().quantile(.90); d['Weight'] = d.Weight.clip(-cap, cap); norm = d.Weight.abs().sum()
        return (d.set_index('Indicator').Weight / norm).to_dict()
        
    def fit(self, stock, feat, h):
        z = self.targets(feat, h).replace([np.inf, -np.inf], np.nan).dropna(subset=TECHNICAL_FEATURES + ['yret', 'yup'])
        if len(z) < 500: return None
        X = z[TECHNICAL_FEATURES].fillna(0); yc = z.yup; yr = z.yret; split = int(.78 * len(z))
        Xtr, Xte = X.iloc[:split], X.iloc[split:]; ytr, yte = yc.iloc[:split], yc.iloc[split:]; ytrr, yter = yr.iloc[:split], yr.iloc[split:]
        
        clfs = {
            'logit': Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=3000, class_weight='balanced'))]),
            'extra': ExtraTreesClassifier(n_estimators=450, min_samples_leaf=8, max_features='sqrt', class_weight='balanced', random_state=7, n_jobs=-1),
            'rf': RandomForestClassifier(n_estimators=350, min_samples_leaf=8, max_features='sqrt', class_weight='balanced', random_state=7, n_jobs=-1),
            'hist': HistGradientBoostingClassifier(max_depth=3, learning_rate=.04, max_iter=300, l2_regularization=.5, random_state=7),
            'xgb': XGBClassifier(n_estimators=350, max_depth=3, learning_rate=.035, subsample=.8, colsample_bytree=.8, reg_lambda=2, eval_metric='logloss', random_state=7, n_jobs=4)
        }
        regs = {
            'extra': ExtraTreesRegressor(n_estimators=450, min_samples_leaf=8, max_features='sqrt', random_state=7, n_jobs=-1),
            'rf': RandomForestRegressor(n_estimators=350, min_samples_leaf=8, max_features='sqrt', random_state=7, n_jobs=-1),
            'hist': HistGradientBoostingRegressor(max_depth=3, learning_rate=.04, max_iter=300, l2_regularization=.5, random_state=7),
            'xgb': XGBRegressor(n_estimators=350, max_depth=3, learning_rate=.035, subsample=.8, colsample_bytree=.8, reg_lambda=2, objective='reg:squarederror', random_state=7, n_jobs=4)
        }
        
        auc = {}; mae = {}
        for k, m in clfs.items(): 
            m.fit(Xtr, ytr)
            prob = np.asarray(m.predict_proba(Xte), dtype=float)
            p = prob[:, 1] if prob.shape[1] > 1 else prob[:, 0]
            auc[k] = roc_auc_score(np.asarray(yte, dtype=int), p) if len(np.unique(yte)) > 1 else .5
        for k, m in regs.items(): 
            m.fit(Xtr, ytrr)
            mae[k] = mean_absolute_error(np.asarray(yter, dtype=float), np.asarray(m.predict(Xte), dtype=float))
            
        best_thr = .5; best_acc = .5; best_cov = 0
        prob_xgb = np.asarray(clfs['xgb'].predict_proba(Xte), dtype=float)
        p_xgb = prob_xgb[:, 1] if prob_xgb.shape[1] > 1 else prob_xgb[:, 0]
        yte_arr = np.asarray(yte, dtype=int)
        
        for thr in np.arange(.5, .86, .02):
            mask = np.asarray((p_xgb >= thr) | (p_xgb <= 1 - thr), dtype=bool)
            if mask.sum() < max(20, int(.1 * len(mask))): continue
            acc = float((((p_xgb[mask] >= .5).astype(int)) == yte_arr[mask]).mean())
            cov = float(mask.mean())
            if acc > best_acc or (abs(acc - best_acc) < 1e-6 and cov > best_cov): 
                best_acc, best_cov, best_thr = acc, cov, float(thr)
                
        wfa = []; wauc = []; wb = []; wc = []
        for tri, tei in TimeSeriesSplit(n_splits=5).split(X):
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=.04, max_iter=250, l2_regularization=.5, random_state=7)
            m.fit(X.iloc[tri], yc.iloc[tri])
            prob_cv = np.asarray(m.predict_proba(X.iloc[tei]), dtype=float)
            pp = prob_cv[:, 1] if prob_cv.shape[1] > 1 else prob_cv[:, 0]
            yy = yc.iloc[tei]
            yy_arr = np.asarray(yy, dtype=int)
            mask = np.asarray((pp >= best_thr) | (pp <= 1 - best_thr), dtype=bool)
            wauc.append(roc_auc_score(yy_arr, pp) if len(np.unique(yy_arr)) > 1 else .5)
            wb.append(brier_score_loss(yy_arr, pp))
            wc.append(float(mask.mean()))
            if mask.sum(): 
                wfa.append(float((((pp[mask] >= .5).astype(int)) == yy_arr[mask]).mean()))
                
        for m in clfs.values(): m.fit(X, yc)
        for m in regs.values(): m.fit(X, yr)
        
        cw = {k: max(v - .5, .01) for k, v in auc.items()}; s = sum(cw.values()); cw = {k: v / s for k, v in cw.items()}
        rw = {k: 1 / (v + 1e-8) for k, v in mae.items()}; s = sum(rw.values()); rw = {k: v / s for k, v in rw.items()}
        bundle = {'trained_through': str(feat.index[-1].date()), 'horizon': h, 'features': TECHNICAL_FEATURES, 'classifiers': clfs, 'regressors': regs, 'cw': cw, 'rw': rw, 'indicator_weights': self.indicator_weights(feat, h), 'validation': {'accuracy': float(np.mean(wfa)) if wfa else .5, 'auc': float(np.mean(wauc)), 'brier': float(np.mean(wb)), 'coverage': float(np.mean(wc)), 'threshold': best_thr}}
        return bundle
        
    def fit_or_load(self, stock, feat, h, force=False):
        p = self.dir / f'{stock}_{h}.joblib'; old = None
        if p.exists():
            try: old = joblib.load(p)
            except: old = None
        if old and not force and old.get('trained_through') == str(feat.index[-1].date()): return old
        new = self.fit(stock, feat, h)
        if new is None: return old
        if old:
            ov = old.get('validation', {}); nv = new['validation']
            worse = nv['auc'] + 0.02 < ov.get('auc', .5) and nv['accuracy'] + 0.03 < ov.get('accuracy', .5)
            if worse: return old
        joblib.dump(new, p); return new
        
    def predict(self, m, feat):
        if not m or not m.get('classifiers'):
            return {'probability_up': .5, 'expected_return': 0, 'expected_volatility': float(feat.vol20.iloc[-1]), 'model_score': 50, 'risk_adjusted_score': 50, 'risk_score': 50, 'oos_accuracy': .5, 'oos_auc': .5, 'oos_brier': .25, 'prediction_coverage': 1}
        X = feat.iloc[[-1]][TECHNICAL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        p = 0.0
        for k, v in m['classifiers'].items():
            prob = np.asarray(v.predict_proba(X), dtype=float)
            val = prob[:, 1][0] if prob.shape[1] > 1 else prob[:, 0][0]
            p += m['cw'][k] * float(val)
            
        er = sum(m['rw'][k] * float(np.asarray(v.predict(X), dtype=float).ravel()[0]) for k, v in m['regressors'].items())
        er = float(np.tanh(er / .2) * .2)
        vol = float(feat.vol20.iloc[-1])
        risk = np.clip(100 - 100 * vol - 60 * abs(float(feat.drawdown.iloc[-1])), 0, 100)
        ra = np.clip(50 + 300 * er / max(vol, .05), 0, 100)
        score = np.clip(100 * (.65 * p + .35 * np.clip(.5 + er / .2, 0, 1)), 0, 100)
        v = m['validation']
        return {'probability_up': float(p), 'expected_return': er, 'expected_volatility': vol, 'model_score': float(score), 'risk_adjusted_score': float(ra), 'risk_score': float(risk), 'oos_accuracy': v['accuracy'], 'oos_auc': v['auc'], 'oos_brier': v['brier'], 'prediction_coverage': v['coverage']}
        
    def full_backtest(self, feat_map, h):
        dates = sorted(set().union(*[set(v.index) for v in feat_map.values()])); dates = pd.DatetimeIndex(dates)
        train = 504; step = 20; equity = [(dates[train], 1.)]; rows = []
        for i in range(train, len(dates) - h, step):
            d = dates[i]; scored = []
            for stock, feat in feat_map.items():
                past = feat.loc[feat.index < d]
                if len(past) < train: continue
                w = self.indicator_weights(past.head(-h), h); z = feat.loc[d]
                s = sum(v * np.tanh((safe(z.get(k)) or 0) / 2.5) for k, v in w.items())
                scored.append((stock, s))
            if len(scored) < 5: continue
            sel = [x[0] for x in sorted(scored, key=lambda x: x[1], reverse=True)[:max(3, len(scored) // 5)]]
            fd = dates[i + h]; rs = []
            for s in sel:
                f = feat_map[s]
                if d in f.index and fd in f.index: rs.append(f.loc[fd, 'Close'] / f.loc[d, 'Close'] - 1)
            if not rs: continue
            rr = np.mean(rs); equity.append((fd, equity[-1][1] * (1 + rr))); rows.append({'Date': d, 'FutureDate': fd, 'Return': rr, 'N': len(rs)})
        eq = pd.Series(dict(equity)).sort_index(); out = {}
        if len(eq) > 1:
            r = eq.pct_change().dropna(); years = max((eq.index[-1] - eq.index[0]).days / 365.25, .1)
            out = {'CAGR': eq.iloc[-1]**(1 / years) - 1, 'AnnualizedVolatility': r.std() * np.sqrt(252), 'Sharpe': r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan, 'MaxDrawdown': (eq / eq.cummax() - 1).min(), 'HitRate': (r > 0).mean(), 'EndingMultiple': eq.iloc[-1], 'OOSAccuracy': (r > 0).mean(), 'Coverage': 1.}
        panel = pd.concat([v.assign(ForwardReturn=v.Close.shift(-h) / v.Close - 1) for v in feat_map.values()]); inds = []
        for f in TECHNICAL_FEATURES:
            q = panel[[f, 'ForwardReturn']].dropna()
            if len(q) < 300: continue
            ic = spearmanr(np.asarray(q[f], dtype=float), np.asarray(q.ForwardReturn, dtype=float), nan_policy='omit').statistic
            inds.append({'Indicator': f, 'MeanIC': ic, 'LearnedWeight': np.sign(ic) * min(abs(ic) * 3, 1), 'DirectionalHitRate': (((q[f] - q[f].median()) * q.ForwardReturn) > 0).mean()})
        return {'summary': out, 'equity': eq, 'indicator_table': pd.DataFrame(inds).sort_values('LearnedWeight', key=lambda x: x.abs(), ascending=False)}

class ClientFitAgent:
    KEYS = {
        'Creativity': ['creative', 'artist', 'design', 'illustration', 'publishing', 'story', 'content', 'media'],
        'Education': ['education', 'educator', 'learning', 'school', 'student', 'teacher'],
        'Entrepreneurship': ['entrepreneur', 'business', 'startup', 'innovation', 'founder'],
        'Technology': ['technology', 'software', 'digital', 'ai', 'platform', 'computing'],
        'Community': ['community', 'collaboration', 'culture', 'social'],
        'Sustainability': ['sustainability', 'sustainable', 'climate', 'energy', 'environment'],
        'Asia': ['asia', 'taiwan', 'china', 'japan', 'korea', 'singapore']
    }
    
    def __init__(self, client): 
        self.client = client
        
    def evaluate(self, raw, fund, feat):
        text = (raw.get('BusinessSummary') or '').lower()
        themes = [k for k, v in self.KEYS.items() if any(w in text for w in v)]
        financial = 0; notes = []
        if np.isfinite(fund.get('Growth', np.nan)) and fund['Growth'] >= 70: financial += 30; notes.append('strong relative growth')
        if np.isfinite(fund.get('Quality', np.nan)) and fund['Quality'] >= 70: financial += 30; notes.append('strong relative quality')
        if np.isfinite(fund.get('Valuation', np.nan)) and fund['Valuation'] >= 70: financial += 20; notes.append('supportive relative valuation')
        if feat.vol20.iloc[-1] < feat.vol60.iloc[-1]: financial += 10; notes.append('recent volatility below longer-term volatility')
        if abs(feat.drawdown.iloc[-1]) < .15: financial += 10; notes.append('moderate current drawdown')
        thematic = min(100, 20 + 12 * len(themes)); score = .6 * financial + .4 * thematic
        headline = f'Client-fit score {score:.1f}/100; themes: {", ".join(themes) if themes else "none detected"}.'
        details = [
            {'Item': 'Business themes', 'Assessment': ', '.join(themes) if themes else 'No strong thematic match found.'},
            {'Item': 'Financial alignment', 'Assessment': '; '.join(notes) if notes else 'No dominant positive factor alignment.'},
            {'Item': 'Client objective', 'Assessment': 'The client requires long-term growth while maintaining high certainty for the future operating commitment and preserving flexibility.'},
            {'Item': 'Source limitation', 'Assessment': 'Thematic fit uses public business-description text; it is not an independent verification of the company mission.'}
        ]
        return {'score': score, 'headline': headline, 'details': details}

class PortfolioAgent:
    def prices(self, holdings):
        hub = DataHub(); out = {}
        for s in holdings:
            p = hub.price(s, '3y')
            if not p.empty: out[s] = p.Close
        return pd.concat(out, axis=1).dropna(how='all')
        
    def report(self, holdings, rf=.04):
        hub = DataHub(); values = {}; positions = []
        for s, h in holdings.items():
            p = hub.price(s, '5d')
            price = float(p.Close.iloc[-1]) if not p.empty else np.nan
            value = price * h['shares'] if np.isfinite(price) else 0
            values[s] = value
        total = sum(values.values())
        for s, h in holdings.items():
            price = values[s] / h['shares'] if h['shares'] else np.nan
            pnl = (price - h['cost_basis']) * h['shares'] if np.isfinite(price) and np.isfinite(h['cost_basis']) else np.nan
            positions.append({'Stock': s, 'Shares': h['shares'], 'Price': price, 'MarketValue': values[s], 'Weight': values[s] / total if total else 0, 'PnL': pnl, 'PnLPercent': price / h['cost_basis'] - 1 if np.isfinite(price) and np.isfinite(h['cost_basis']) and h['cost_basis'] else np.nan})
        prices = self.prices(holdings); w = pd.Series({s: (v / total if total else 0) for s, v in values.items()}).reindex(prices.columns).fillna(0)
        rets = prices.pct_change().dropna().fillna(0); p = rets @ w; annret = (1 + p.mean())**252 - 1; vol = p.std() * np.sqrt(252); sharpe = (annret - rf) / vol if vol else np.nan; down = p[p < 0].std() * np.sqrt(252); sortino = (annret - rf) / down if down else np.nan; eq = (1 + p).cumprod(); dd = eq / eq.cummax() - 1; cov = pd.DataFrame(LedoitWolf().fit(rets.values).covariance_ * 252, index=rets.columns, columns=rets.columns); var = w.values @ cov.values @ w.values; rc = w.values * (cov.values @ w.values) / np.sqrt(max(var, 1e-12)); rc = rc / rc.sum() if rc.sum() else rc; corr = rets.corr()
        avg = np.nanmean([corr.iloc[i, j] for i in range(len(w)) for j in range(i + 1, len(w))]) if len(w) > 1 else np.nan
        hhi = (w[w > 0]**2).sum(); dr = (np.abs(w.values) * rets.std().values * np.sqrt(252)).sum() / vol if vol else np.nan
        metrics = [('Annualized return', annret), ('Annualized volatility', vol), ('Sharpe', sharpe), ('Sortino', sortino), ('Max drawdown', dd.min()), ('Average pairwise correlation', avg), ('Diversification ratio', dr), ('HHI', hhi), ('Effective number of positions', 1 / hhi if hhi else np.nan)]
        return {'portfolio_value': total, 'positions': positions, 'correlation': corr, 'risk_contribution': [{'Stock': s, 'RiskContribution': float(v)} for s, v in zip(w.index, rc)], 'metrics': {'AnnualizedReturn': annret, 'AnnualizedVolatility': vol, 'Sharpe': sharpe, 'Sortino': sortino, 'MaxDrawdown': dd.min()}, 'metrics_table': [{'Metric': a, 'Value': b} for a, b in metrics], 'correlation_interpretation': f'Average pairwise correlation is {avg:.2f}. Lower/negative correlations can reduce covariance terms and improve risk-adjusted efficiency; they do not create expected return automatically.', 'client_constraints': ['Protect the 2033-2042 operating commitment with a high degree of certainty.', 'De-risk the liability bucket as cash needs approach.', 'Only treat surplus capital after the operating reserve as facility capital; preserve flexibility.']}
        
    def frontier(self, holdings, rf=.04):
        prices = self.prices(holdings); rets = prices.pct_change().dropna().fillna(0); names = list(rets.columns); mu = rets.mean().values * 252; cov = LedoitWolf().fit(rets.values).covariance_ * 252; n = len(names); x = np.ones(n) / n; cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}; bounds = [(0, .5)] * n
        def ret(w): return float(w @ mu)
        def vol(w): return float(np.sqrt(max(w @ cov @ w, 1e-12)))
        def neg(w): return -(ret(w) - rf) / vol(w)
        a = minimize(neg, x, bounds=bounds, constraints=cons, method='SLSQP'); b = minimize(vol, x, bounds=bounds, constraints=cons, method='SLSQP'); wm = a.x if a.success else x; wv = b.x if b.success else x
        targets = np.linspace(max(0, ret(wv)), max(ret(wm), ret(wv)) + .08, 30); fr = []
        for t in targets:
            cc = [cons, {'type': 'eq', 'fun': lambda w, t=t: ret(w) - t}]; q = minimize(vol, x, bounds=bounds, constraints=cc, method='SLSQP')
            if q.success: fr.append({'ExpectedReturn': t, 'Volatility': vol(q.x), 'Sharpe': (t - rf) / vol(q.x)})
        return {'frontier': pd.DataFrame(fr).set_index('ExpectedReturn')['Volatility'] if fr else pd.Series(dtype=float), 'key': [{'Portfolio': 'Minimum variance', 'ExpectedReturn': ret(wv), 'Volatility': vol(wv), 'Sharpe': (ret(wv) - rf) / vol(wv)}, {'Portfolio': 'Maximum Sharpe', 'ExpectedReturn': ret(wm), 'Volatility': vol(wm), 'Sharpe': (ret(wm) - rf) / vol(wm)}], 'weights': [{'Stock': s, 'Weight': float(v)} for s, v in zip(names, wm)]}