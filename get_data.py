import yfinance as yf
import pandas as pd
import datetime

start = '1990-01-01'
end = datetime.datetime.today().strftime('%Y-%m-%d')

def download_close(ticker, name, start, end):
    raw = yf.download(ticker, start=start, end=end, progress=False)['Close']
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.name = name
    return raw

vix   = download_close('^VIX',     'VIX',   start, end)
vix3m = download_close('^VIX3M',   'VIX3M', start, end)
sp500 = download_close('^GSPC',    'SP500', start, end)
gold  = download_close('GC=F',     'GOLD',  start, end)
tny   = download_close('^TNX',     'TNY',   start, end)
dxy   = download_close('DX-Y.NYB', 'DXY',   start, end)
vix9d = download_close('^VIX9D',   'VIX9D', start, end)
skew  = download_close('^SKEW',    'SKEW',  start, end)

df = pd.concat([vix, vix3m, sp500, gold, tny, dxy, vix9d, skew], axis=1)
df = df.dropna(subset=['VIX'])
df = df.ffill()

df.to_csv('data/raw_data.csv')
print("Data downloaded successfully")
print(df.tail())
print(df.shape)
