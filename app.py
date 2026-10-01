import numpy as np, pandas as pd, streamlit as st
from orchestrator import analyze_universe, parse_holdings, portfolio_report, efficient_frontier, run_walk_forward_backtest, data_health, horizon_label

st.set_page_config(page_title='Quant Portfolio Intelligence', layout='wide')
st.title('Quant Portfolio Intelligence')
st.caption('Fundamental + technical + ML + portfolio construction')

with st.sidebar:
    st.header('Research universe')
    universe = st.text_area('Stocks / ETFs / bond ETFs', 'AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,TSLA,AMD,ORCL,CRM,LLY,SPY,QQQ,TLT,IEF,SHY,GLD', height=100)

    st.header('Forecast horizon')
    horizon_value = st.number_input('How far ahead?', min_value=1.0, max_value=2500.0, value=20.0, step=1.0)
    horizon_unit = st.selectbox('Unit', ['Trading days', 'Months', 'Years'], index=0)
    if horizon_unit == 'Trading days':
        horizon = int(round(horizon_value))
    elif horizon_unit == 'Months':
        horizon = int(round(horizon_value * 21))
    else:
        horizon = int(round(horizon_value * 252))
    st.caption(f'Model horizon: {horizon:,} trading days ({horizon_value:g} {horizon_unit.lower()}).')

    min_history = st.number_input('Minimum history', 500, 3000, 750, 50)
    update = st.button('Update data + train', type='primary')

    st.divider()
    st.header('Portfolio holdings')
    holdings_text = st.text_area('Current holdings: STOCK, SHARES, COST BASIS', 'AAPL,10,180\nMSFT,8,390\nNVDA,12,150', height=100)
    rf = st.number_input('Risk-free rate', 0.0, 0.2, 0.04, 0.005, format='%.3f')

try:
    table, detail = analyze_universe(universe, horizon, int(min_history), update)
except Exception as e:
    st.error(f'Model failed: {e}')
    st.stop()

if table.empty:
    st.error('No securities analyzed. Check symbols, data/API access, or whether the requested horizon is supported by the available history.')
    st.stop()

tabs = st.tabs(['Stock lab', 'Portfolio manager', 'Backtest lab', 'Efficient frontier', 'System health'])

with tabs[0]:
    cols = [c for c in ['Stock','Name','AssetClass','ForecastHorizonDays','OverallScore','ModelProbabilityUp','ExpectedReturn','ExpectedVolatility','Quality','Valuation','Growth','TechnicalScore','RiskScore','ClientFit','OOSAccuracy','OOSAUC','PredictionCoverage','DataQualityScore','DataDate'] if c in table.columns]
    fmt = {'ForecastHorizonDays':'{:.0f}','OverallScore':'{:.1f}','ModelProbabilityUp':'{:.1%}','ExpectedReturn':'{:.1%}','ExpectedVolatility':'{:.1%}','Quality':'{:.1f}','Valuation':'{:.1f}','Growth':'{:.1f}','TechnicalScore':'{:.1f}','RiskScore':'{:.1f}','ClientFit':'{:.1f}','OOSAccuracy':'{:.1%}','OOSAUC':'{:.3f}','PredictionCoverage':'{:.1%}','DataQualityScore':'{:.1f}'}
    st.dataframe(table[cols].style.format(fmt), use_container_width=True, height=520, hide_index=True)
    st.download_button('Download ranking', table.to_csv(index=False), 'ranking.csv', 'text/csv')

    s = st.selectbox('Inspect security', table.Stock.tolist())
    r = detail[s]
    a,b,c,d = st.columns(4)
    a.metric('Overall', f"{r['OverallScore']:.1f}")
    b.metric(f"P(up) — {horizon_label(horizon)}", f"{r['ModelProbabilityUp']:.1%}")
    c.metric(f"Expected return — {horizon_label(horizon)}", f"{r['ExpectedReturn']:.1%}")
    d.metric('OOS accuracy', f"{r['OOSAccuracy']:.1%}")

    st.write(r['ExecutiveSummary'])
    st.line_chart(r['PriceChart'], use_container_width=True)

    st.subheader('Fundamentals')
    st.dataframe(pd.DataFrame(r['FundamentalMetrics']).style.format({'Value':'{:.4g}'}), use_container_width=True, hide_index=True)
    st.write(r['FundamentalNarrative'])

    st.subheader('Technical analysis: 70+ signals')
    st.dataframe(pd.DataFrame(r['TechnicalIndicators']).style.format({'Value':'{:.6g}','LearnedWeight':'{:.5f}'}), use_container_width=True, height=600, hide_index=True)
    st.write(r['TechnicalNarrative'])

    st.subheader('Quant / ML diagnostics')
    st.dataframe(pd.DataFrame([
        {'Metric':'Forecast horizon','Value':f'{horizon:,} trading days'},
        {'Metric':'Walk-forward accuracy','Value':r['OOSAccuracy']},
        {'Metric':'ROC-AUC','Value':r['OOSAUC']},
        {'Metric':'Brier score','Value':r['OOSBrier']},
        {'Metric':'Prediction coverage','Value':r['PredictionCoverage']},
        {'Metric':'Confidence threshold','Value':r['ConfidenceThreshold']},
        {'Metric':'Expected volatility','Value':r['ExpectedVolatility']},
        {'Metric':'Learned indicators','Value':r['LearnedIndicatorCount']},
    ]).style.format({'Value':'{:.4g}'}), use_container_width=True, hide_index=True)

    st.subheader('Client-fit agent')
    st.dataframe(pd.DataFrame(r['ClientFitDetails']), use_container_width=True, hide_index=True)

with tabs[1]:
    h = parse_holdings(holdings_text)
    if h:
        rep = portfolio_report(h, rf)
        st.dataframe(pd.DataFrame(rep['positions']).style.format({'Price':'{:.2f}','MarketValue':'{:.2f}','Weight':'{:.1%}','PnL':'{:.2f}','PnLPercent':'{:.1%}'}), use_container_width=True, hide_index=True)
        a,b,c,d = st.columns(4)
        a.metric('Value', f"${rep['portfolio_value']:,.0f}")
        b.metric('Vol', f"{rep['metrics']['AnnualizedVolatility']:.1%}")
        c.metric('Sharpe', f"{rep['metrics']['Sharpe']:.2f}")
        d.metric('Max DD', f"{rep['metrics']['MaxDrawdown']:.1%}")
        st.subheader('Correlation')
        st.dataframe(rep['correlation'].style.format('{:.2f}'), use_container_width=True)
        st.subheader('Risk contributions')
        st.dataframe(pd.DataFrame(rep['risk_contribution']).style.format({'RiskContribution':'{:.1%}'}), use_container_width=True, hide_index=True)
        st.subheader('Portfolio diagnostics')
        st.dataframe(pd.DataFrame(rep['metrics_table']).style.format({'Value':'{:.4g}'}), use_container_width=True, hide_index=True)
        st.write(rep['correlation_interpretation'])
        st.write('• ' + '\n• '.join(rep['client_constraints']))
    else:
        st.info('Add holdings in the sidebar.')

with tabs[2]:
    st.subheader('Historical walk-forward backtest')
    st.write(f'This test uses the selected {horizon_label(horizon)} horizon and purges training observations whose future-return labels could overlap the test window.')
    if st.button('Run walk-forward backtest'):
        with st.spinner('Running historical out-of-sample tests...'):
            bt = run_walk_forward_backtest(universe, horizon, int(min_history))
        if bt['summary']:
            st.dataframe(pd.DataFrame([bt['summary']]).style.format({'CAGR':'{:.1%}','AnnualizedVolatility':'{:.1%}','Sharpe':'{:.2f}','Sortino':'{:.2f}','MaxDrawdown':'{:.1%}','HitRate':'{:.1%}','OOSAccuracy':'{:.1%}','EndingMultiple':'{:.2f}x'}), use_container_width=True, hide_index=True)
        if bt['equity'] is not None and not bt['equity'].empty:
            st.line_chart(bt['equity'], use_container_width=True)
        st.dataframe(bt['indicator_table'].style.format({'MeanIC':'{:.4f}','ICStd':'{:.4f}','LearnedWeight':'{:.4f}','DirectionalHitRate':'{:.1%}'}), use_container_width=True, hide_index=True)
        st.download_button('Download backtest', bt['indicator_table'].to_csv(index=False), 'walk_forward_indicators.csv', 'text/csv')

with tabs[3]:
    h = parse_holdings(holdings_text)
    if h:
        try:
            ef = efficient_frontier(h, rf)
            st.dataframe(pd.DataFrame(ef['key']).style.format({'ExpectedReturn':'{:.1%}','Volatility':'{:.1%}','Sharpe':'{:.2f}'}), use_container_width=True, hide_index=True)
            if ef['frontier'] is not None and not ef['frontier'].empty:
                st.line_chart(ef['frontier'], use_container_width=True)
            st.dataframe(pd.DataFrame(ef['weights']).style.format({'Weight':'{:.1%}'}), use_container_width=True, hide_index=True)
            st.info('Covariance is the key portfolio-level calculation: lower or negative correlations can reduce variance and improve Sharpe, but they do not guarantee higher expected returns.')
        except Exception as e:
            st.error(f'Efficient frontier failed: {e}')
    else:
        st.info('Add holdings first.')

with tabs[4]:
    st.dataframe(pd.DataFrame(data_health()), use_container_width=True, hide_index=True)
    st.subheader('Forecast configuration')
    st.dataframe(pd.DataFrame([
        {'Parameter':'Horizon entered','Value':f'{horizon_value:g} {horizon_unit.lower()}'},
        {'Parameter':'Converted horizon','Value':f'{horizon:,} trading days'},
        {'Parameter':'Minimum history requested','Value':int(min_history)},
    ]), use_container_width=True, hide_index=True)
    st.write('Client case encoded: $300k in 2027, +$150k in 2028, ten fixed $50k payments 2033-2042, with high funding certainty and preserved flexibility.')
    st.info('The system measures the 80% target; it does not guarantee it. Accuracy is reported with coverage, AUC and Brier score to expose low-coverage or poorly calibrated predictions.')
