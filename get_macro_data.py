import os
import requests
import pandas as pd
import datetime

API_KEY = os.environ.get('FRED_API_KEY') or open('.env').read().split('=')[1].strip()

start = '1990-01-01'
end   = datetime.datetime.today().strftime('%Y-%m-%d')

def fetch_fred(series_id, start, end, api_key):
    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id':       series_id,
        'api_key':         api_key,
        'file_type':       'json',
        'observation_start': start,
        'observation_end':   end,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    obs = r.json()['observations']
    df  = pd.DataFrame(obs)[['date', 'value']]
    df['date']  = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.set_index('date')['value']
    df.name = series_id
    return df

fedfunds   = fetch_fred('FEDFUNDS',  start, end, API_KEY)
yield_curve = fetch_fred('T10Y2Y',   start, end, API_KEY)

print(f"FEDFUNDS:   {len(fedfunds)} obs  ({fedfunds.index[0].date()} – {fedfunds.index[-1].date()})")
print(f"T10Y2Y:     {len(yield_curve)} obs  ({yield_curve.index[0].date()} – {yield_curve.index[-1].date()})")

raw = pd.read_csv('data/raw_data.csv', index_col=0, parse_dates=True)

daily_idx = raw.index

fedfunds_daily    = fedfunds.reindex(daily_idx).ffill()
yield_curve_daily = yield_curve.reindex(daily_idx).ffill()

raw['FEDFUNDS']    = fedfunds_daily
raw['YIELD_CURVE'] = yield_curve_daily

raw.to_csv('data/raw_data.csv')
print("\nMacro data merged into data/raw_data.csv")
print(f"Shape: {raw.shape}")
print(raw[['FEDFUNDS', 'YIELD_CURVE']].dropna().tail())
