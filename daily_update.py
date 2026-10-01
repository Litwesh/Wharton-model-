import argparse
from orchestrator import analyze_universe,horizon_label

parser=argparse.ArgumentParser()
parser.add_argument('--horizon',type=int,default=20,help='Forecast horizon in trading days')
args=parser.parse_args()

universe='AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA AMD ORCL CRM LLY JPM SPY QQQ TLT IEF SHY GLD'

table,detail=analyze_universe(universe,args.horizon,750,True)
print(f'Forecast horizon: {horizon_label(args.horizon)}')
if table.empty:
    print('No securities analyzed.')
else:
    print(table[['Stock','OverallScore','ModelProbabilityUp','ExpectedReturn','ExpectedVolatility','OOSAccuracy','OOSAUC','PredictionCoverage','ClientFit','DataQualityScore']].to_string(index=False))