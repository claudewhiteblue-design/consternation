import pandas as pd
def load(p,name):
    d=pd.read_csv(p)[['TIME_PERIOD','OBS_VALUE']].rename(columns={'OBS_VALUE':name})
    d['TIME_PERIOD']=pd.to_datetime(d['TIME_PERIOD']); return d.set_index('TIME_PERIOD')[name]
fx=pd.concat([load('/tmp/fxl_RER_USD_ILS.csv','usd'),load('/tmp/fxl_RER_EUR_ILS.csv','eur')],axis=1).resample('MS').mean()
fx=fx.loc['2023-10-01':'2026-07-01']
fx['basket']=.5*fx.usd+.5*fx.eur
fx.to_csv('/tmp/fx_monthly_lagged.csv')
print(fx.head(5).round(4).to_string()); print('...'); print(fx.tail(3).round(4).to_string())
print('months available:',len(fx),'(need 31 + 3 lag =',34,')')
