import pandas as pd
def load(p,name):
    d=pd.read_csv(p)
    d=d[['TIME_PERIOD','OBS_VALUE']].rename(columns={'OBS_VALUE':name})
    d['TIME_PERIOD']=pd.to_datetime(d['TIME_PERIOD'])
    return d.set_index('TIME_PERIOD')[name]
usd=load('/tmp/fx_RER_USD_ILS.csv','usd'); eur=load('/tmp/fx_RER_EUR_ILS.csv','eur')
fx=pd.concat([usd,eur],axis=1).resample('MS').mean()
fx=fx.loc['2024-01-01':'2026-07-01']
fx['basket']=0.5*fx.usd+0.5*fx.eur
fx.to_csv('/tmp/fx_monthly.csv')
print(fx.round(4).to_string())
print()
print('months:',len(fx))
