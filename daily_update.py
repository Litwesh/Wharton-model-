from orchestrator import analyze_universe
UNIVERSE='AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA AMD ORCL CRM LLY JPM SPY QQQ TLT IEF SHY GLD'
if __name__=='__main__':
    table,_=analyze_universe(UNIVERSE,20,750,True)
    print(table[['Stock','OverallScore','ModelProbabilityUp','ExpectedReturn','OOSAccuracy','OOSAUC','ClientFit']].to_string(index=False) if not table.empty else 'No results')
