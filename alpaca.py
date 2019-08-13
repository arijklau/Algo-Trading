import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np

'''
MODEL OUTLINE:
indicators at day [0:99] : ∆ price at day [106, 129] (i.e how much will stock rise/fall over the next week/month)
21 indicators x 100 days --> high dimensionality, figure out floydhub to train? use that free CPU/GPU

indicators
    *open
    *high
    *low
    *close
    *simple MA 50 (need 150 dp)
    *simple MA 200 (need 300 dp)
    *exp MA 50
    *exp MA 200
    *tri MA 50
    #tri MA 200
    MACD
    RSI (14 day)
    Bollinger band upper (sma20 + 2sigma(20-day close))
    bollinger band middle (sma20)
    bollinger band lower (sma20 - 2sigma(20-day close))
    parabolic sar
    time series forecast
    typical price
    weighted close
    median price
    volume
    open interest
    
    --> consider adding 10 day MA's 
'''

endpoint = 'https://paper-api.alpaca.markets'
key_id = 'PK35ENGBMQNOF6K1NCK7'
secret_key = 'd0zpeAM/y/n5XdMu3HX82iKNtKFPJeSfdb14pQyz'

api = tradeapi.REST(key_id, secret_key, base_url=endpoint, api_version='v2') # or use ENV Vars shown below
universe = ['#MMM','#AXP','#AAPL','#BA','#CVX','#CSCO','#KO','#DWDP',
        '#XOM','#GS','#HD','#IBM','#INTC','#JNJ','#JPM','#MCD','#MRK','#MSFT']

#returns a dataframe prepopulated with open, high, low, close, volume for symbol over last n trading days
def ohlcv(symbol, n):
    open, high, low, close, dates, vol = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), [0]*n, np.zeros(n)
    bar = api.get_barset(symbol, '1D', limit=n)
    b = bar[symbol]
    for i in range(len(b)):
        open[i] = b[i].o
        high[i] = b[i].h
        low[i] = b[i].l
        close[i] = b[i].c
        vol[i] = b[i].v
    
    df = pd.DataFrame({
        'open':pd.Series(open),
        'high':pd.Series(high),
        'low':pd.Series(low),
        'close':pd.Series(close),
        'vol':pd.Series(vol)
    })
    return df

def sma50(df):
    if(df.shape[0] < 150):
        raise ValueError("DataFrame must have 150 or more rows")

    sma50 = pd.Series(np.zeros(df.shape[0]))
    for i in range(50, sma50.shape[0]):
        sma50[i] = np.mean(df.loc[i-50:i, 'close'])

    df['sma50'] = sma50
    return df

def sma200(df):
    if(df.shape[0] < 300):
        raise ValueError("DataFrame must have 300 or more rows")

    sma200 = pd.Series(np.zeros(df.shape[0]))
    for i in range(200, sma200.shape[0]):
        sma200[i] = np.mean(df.loc[i-200:i, 'close'])

    df['sma200'] = sma200
    return df

def ema50(df):
    if(df.shape[0] < 150):
        raise ValueError("DataFrame must have 150 or more rows")

    k = 2/51
    ema50 = pd.Series(np.zeros(df.shape[0]))
    ema50[50] = df.loc[50, 'sma50'] #first period of calculation is SMA
    for i in range(51, ema50.shape[0]):
        ema50[i] = (k*df.loc[i, 'close']) + ((1-k)*ema50[i - 1])
        
    df['ema50'] = ema50
    return df

def ema200(df):
    if(df.shape[0] < 300):
        raise ValueError("DataFrame must have 300 or more rows")

    k = 2/201
    ema200 = pd.Series(np.zeros(df.shape[0]))
    ema200[200] = df.loc[200, 'sma200'] #first period of calculation is SMA
    for i in range(201, ema200.shape[0]):
        ema200[i] = (k*df.loc[i, 'close']) + ((1-k)*ema200[i - 1])
    return

def tma50(df):
    if(df.shape[0] < 150):
        raise ValueError("DataFrame must have 150 or more rows")

    tma50 = pd.Series(np.zeros(df.shape[0]))
    for i in range(50, tma50.shape[0]):
        tma50[i] = np.mean(df.loc[i-50:i, 'sma50'])
    
    df['tma50'] = tma50
    return df

def tma200(df):
    if(df.shape[0] < 300):
        raise ValueError("DataFrame must have 300 or more rows")

    tma200 = pd.Series(np.zeros(df.shape[0]))
    for i in range(200, tma200.shape[0]):
        tma200[i] = np.mean(df.loc[i-200:i, 'sma200'])
    
    df['tma200'] = tma200
    return df

def macd(df):
    return

def rsi(df):
    return





df = ohlcv('MMM', 300)
df = sma50(df)
df = sma200(df)
df = ema50(df)
print(df)






