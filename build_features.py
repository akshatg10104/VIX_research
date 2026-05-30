import pandas as pd
import numpy as np

df = pd.read_csv('data/raw_data.csv', index_col=0, parse_dates=True)

df['VIX_SMA10'] = df['VIX'].rolling(window=10).mean()
df['VIX_SMA20'] = df['VIX'].rolling(window=20).mean()

df['VIX_MOM10'] = df['VIX'] - df['VIX'].shift(10)
df['VIX_MOM20'] = df['VIX'] - df['VIX'].shift(20)

df['VIX_ROC10'] = df['VIX'].pct_change(periods=10)
df['VIX_ROC20'] = df['VIX'].pct_change(periods=20)

df['VIX_BB_MID']      = df['VIX'].rolling(20).mean()
df['VIX_BB_STD']      = df['VIX'].rolling(20).std()
df['VIX_BB_UPPER']    = df['VIX_BB_MID'] + 1.5 * df['VIX_BB_STD']
df['VIX_BB_LOWER']    = df['VIX_BB_MID'] - 1.5 * df['VIX_BB_STD']
df['VIX_BB_POSITION'] = (df['VIX'] - df['VIX_BB_LOWER']) / (df['VIX_BB_UPPER'] - df['VIX_BB_LOWER'])

def calculate_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

df['VIX_RSI14']   = calculate_rsi(df['VIX'], 14)
df['VIX_MA_CROSS'] = df['VIX_SMA10'] - df['VIX_SMA20']
df['VIX_STDDEV']   = df['VIX'].rolling(20).std()

df['VIX_TERM_SPREAD'] = df['VIX3M'] - df['VIX']

df['SP500_RET1']  = df['SP500'].pct_change(1)
df['SP500_RET5']  = df['SP500'].pct_change(5)
df['SP500_RET20'] = df['SP500'].pct_change(20)

df['SP500_MOM10'] = df['SP500'] - df['SP500'].shift(10)
df['SP500_MOM20'] = df['SP500'] - df['SP500'].shift(20)

df['SP500_RVOL'] = df['SP500_RET1'].rolling(20).std() * np.sqrt(252)

rolling_max          = df['SP500'].rolling(252).max()
df['SP500_DRAWDOWN'] = (df['SP500'] - rolling_max) / rolling_max

df['GOLD_RET1']  = df['GOLD'].pct_change(1)
df['GOLD_RET5']  = df['GOLD'].pct_change(5)
df['GOLD_RET20'] = df['GOLD'].pct_change(20)

df['TNY_CHANGE1']  = df['TNY'].diff(1)
df['TNY_CHANGE5']  = df['TNY'].diff(5)
df['TNY_CHANGE20'] = df['TNY'].diff(20)

df['DXY_RET1']  = df['DXY'].pct_change(1)
df['DXY_RET5']  = df['DXY'].pct_change(5)
df['DXY_RET20'] = df['DXY'].pct_change(20)

df['LABEL'] = (df['VIX'] >= 20).astype(int)

for n in [5, 10, 15, 20, 25]:
    shifted = df['VIX'].shift(-n)
    df[f'LABEL_{n}'] = (shifted >= 20).astype(float).where(shifted.notna())

df['FEDFUNDS_CHANGE1M'] = df['FEDFUNDS'].diff(21)
df['FEDFUNDS_CHANGE3M'] = df['FEDFUNDS'].diff(63)

df['YIELD_CURVE_CHANGE1M'] = df['YIELD_CURVE'].diff(21)
df['YIELD_CURVE_CHANGE3M'] = df['YIELD_CURVE'].diff(63)

raw_cols   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY', 'FEDFUNDS', 'YIELD_CURVE']
label_cols = ['LABEL', 'LABEL_5', 'LABEL_10', 'LABEL_15', 'LABEL_20', 'LABEL_25']
feature_cols = [c for c in df.columns if c not in raw_cols + label_cols]

df = df.dropna(subset=feature_cols)

df.to_csv('data/features.csv')
print("Features built successfully")
print(f"Shape: {df.shape}")
print(f"\nLabel distribution:\n{df['LABEL'].value_counts()}")
for n in [5, 10, 15, 20, 25]:
    valid = df[f'LABEL_{n}'].dropna()
    print(f"LABEL_{n:2d}: {int((valid==1).sum())} high-vol / {int((valid==0).sum())} low-vol ({len(valid)} rows)")
