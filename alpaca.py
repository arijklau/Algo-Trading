import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np

'''
indicators
    *open
    *high
    *low
    *close
    *volume
    *simple MA 50
    *simple MA 200
    *exp MA 50
    *exp MA 200
    *MACD
    *RSI (14 day)
    *Bollinger band upper (sma20 + 2sigma(20-day close))
    *bollinger band middle (sma20)
    *bollinger band lower (sma20 - 2sigma(20-day close))
    *bollinger band diff (bbu - bbl)
    *adx
'''

class AlpacaStreamer:

    def loadData(self, stock, days=100, date=False):
        if(date):
            df = ohlcv(stock, days, date)
        else:
            df = ohlcv(stock, 100)
        populate_df(df)
        return df


endpoint = 'https://paper-api.alpaca.markets'
key_id = 'PK35ENGBMQNOF6K1NCK7'
secret_key = 'd0zpeAM/y/n5XdMu3HX82iKNtKFPJeSfdb14pQyz'

api = tradeapi.REST(key_id, secret_key, base_url=endpoint, api_version='v2') # or use ENV Vars shown below
'''universe = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','KO','DIS','DOW','XOM','GS','HD','IBM','INTC',
'JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG','TRV','UTX','UNH','VZ','V','WMT','WBA']'''

#returns a dataframe prepopulated with open, high, low, close, volume for symbol over last n trading days
def ohlcv(symbol, n, date):
    open, high, low, close, dates, vol = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), [0]*n, np.zeros(n)
    bar = api.get_barset(symbol, '1D', limit=n, until=date)
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

def sma(df, n):
    if(df.shape[0] < (n + 100)):
        raise ValueError("DataFrame must have " + str(n + 100) + " or more rows")

    sma = pd.Series(np.zeros(df.shape[0]))
    for i in range(n, sma.shape[0]):
        sma[i] = np.mean(df.loc[i-n:i, 'close'])

    return sma

def ema(df, n):
    if(df.shape[0] < (n + 100)):
        raise ValueError("DataFrame must have " + str(n + 100) + " or more rows")

    k = 2/(n+1)
    ema = pd.Series(np.zeros(df.shape[0]))
    ema[n] = df.loc[n, 'sma' + str(n)]
    for i in range(n+1, ema.shape[0]):
        ema[i] = (k*df.loc[i, 'close']) + ((1-k)*ema[i - 1])

    return ema

def tma(df, n):
    if(df.shape[0] < n + 100):
        raise ValueError("DataFrame must have " + str(n + 100) + " or more rows")

    tma = pd.Series(np.zeros(df.shape[0]))
    for i in range(n, tma.shape[0]):
        tma[i] = np.mean(df.loc[i-n:i, 'sma' + str(n)])
    
    return tma

def macd(df):
    df['sma26'] = sma(df, 26)
    df['sma12'] = sma(df, 12)
    ema26 = ema(df, 26)
    ema12 = ema(df, 12)

    macd = ema26 - ema12
    df = df.drop(columns=['sma26', 'sma12'])
    return macd

def rsi(df):
    if(df.shape[0] < 14):
        raise ValueError("DataFrame must have " + str(14) + " or more rows")

    rsi = pd.Series(np.zeros(df.shape[0]))
    for i in range(14, rsi.shape[0]):
        temp = df.iloc[i-14:i]
        gains = temp[temp['close'] > temp['open']]
        losses = temp[temp['close'] <= temp['open']]
        avg_gain = np.mean((gains['close'] - gains['open'])/gains['open'])/14
        avg_loss = -1*np.mean((losses['close'] - losses['open'])/losses['open'])/14
        if(avg_loss == 0):
            avg_loss = 0.1/14
        rsi[i] = 100 - (100/(1 + (avg_gain/avg_loss)))

    return rsi

def bbu(df):
    sma_ = sma(df, 20)
    bbu = pd.Series(np.zeros(df.shape[0]))
    for i in range(20, bbu.shape[0]):
        bbu[i] = sma_[i] + np.std(df.loc[i-20:i, 'close'])

    return bbu

def bbm(df):
    return sma(df, 20)

def bbl(df):
    sma_ = sma(df, 20)
    bbl = pd.Series(np.zeros(df.shape[0]))
    for i in range(20, bbl.shape[0]):
        bbl[i] = sma_[i] - np.std(df.loc[i-20:i, 'close'])

    return bbl

def bbdiff(df):
    return df['bbu'] - df['bbl']

def adx(df, n):
    tr = np.zeros(df.shape[0])
    tr[0] = df.loc[0, 'high'] - df.loc[0, 'low']
    for i in range(1, df.shape[0]):
        hilo = df.loc[i, 'high'] - df.loc[i, 'low']
        hicl = np.abs(df.loc[i, 'high'] - df.loc[i - 1, 'close'])
        locl = np.abs(df.loc[i, 'low'] - df.loc[i - 1, 'close'])
        tr[i] = max(hilo, hicl, locl)

    atr = np.zeros(df.shape[0])
    for i in range(n, df.shape[0]):
        atr[i] = np.mean(tr[i-n:i])

    pos_dm, neg_dm = np.zeros(df.shape[0]), np.zeros(df.shape[0])
    for i in range(1, df.shape[0]):
        pos_dm[i] = df.loc[i, 'high'] - df.loc[i-1, 'high']
        neg_dm[i] = df.loc[i, 'low'] - df.loc[i-1, 'low']
    
    smooth_pos_dm, smooth_neg_dm = np.zeros(df.shape[0]), np.zeros(df.shape[0])
    for i in range(n, df.shape[0]):
        smooth_pos_dm[i] = np.sum(pos_dm[i-n:i-1]) - (np.sum(pos_dm[i-n:i-1])/(n - 1)) + pos_dm[i]
        smooth_neg_dm[i] = np.sum(neg_dm[i-n:i-1]) - (np.sum(neg_dm[i-n:i-1])/(n - 1)) + neg_dm[i]
    
    pos_di, neg_di, dx = np.zeros(df.shape[0]), np.zeros(df.shape[0]), np.zeros(df.shape[0])
    for i in range(n, df.shape[0]):
        pos_di[i] = 100*smooth_pos_dm[i]/atr[i]
        neg_di[i] = 100*smooth_neg_dm[i]/atr[i]
        dx[i] = 100*((pos_di[i] - neg_di[i])/(pos_di[i] + neg_di[i]))

    adx = np.zeros(df.shape[0])
    for i in range(n, df.shape[0]):
        adx[i] = np.mean(dx[i-n:i])

    return adx

def populate_df(df):
    df['sma50'] = sma(df, 50)
    df['sma200'] = sma(df, 200)
    df['ema50'] = ema(df, 50)
    df['ema200'] = ema(df, 200)
    df['macd'] = macd(df)
    df['rsi'] = rsi(df)
    df['bbu'] = bbu(df)
    df['bbm'] = bbm(df)
    df['bbl'] = bbl(df)
    df['bbdiff'] = bbdiff(df)
    df['adx50'] = adx(df, 50)
    df['adx200'] = adx(df, 200)

#
#   Save all this shit for retraining in the future
#

#date = pd.Timestamp(year = 2016, month = 1, day = 1, tz = 'US/Eastern').isoformat()

#   universe = ['MMM', 'AXP','AAPL','BA','CAT','CVX','CSCO','KO','DIS','DOW','XOM','GS','HD','IBM','INTC',
#   'JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG','TRV','UTX','UNH','VZ','V','WMT','WBA']

# Mid-caps
'''universe = ['AAN', 'ACM', 'ATR', 'BKH', 'BRO', 'BYD', 'CAKE', 'CBSH', 'CHE', 'CIEN', 
    'DDS', 'DNKN', 'ENR', 'ERI', 'EVR', 'FFIN', 'FLR', 'GATX', 'HWC', 'HOMB', 'IDA',
    'INGR', 'JBL', 'LECO', 'LM', 'MAN', 'MDP', 'NNN', 'NYT', 'OII', 'PACW', 'PB', 
    'R', 'RYN', 'SBNY', 'SFM', 'STLD', 'TOL', 'URBN', 'WERN', 'Y', 'ZBRA']


data = np.zeros((671, 1901))
for stock in universe:
    f = open('data/' + stock + '.csv', 'a')
    df = ohlcv(stock, 1000, date)
    populate_df(df)
    for i in range(299, 970):
        x = df.loc[i-99:i].values.flatten()
        y = 100*(df.loc[i + 30, 'close'] - df.loc[i, 'close'])/df.loc[i, 'close']
        data[i - 299] = np.append(x, y)
    data_df = pd.DataFrame(data, columns=None)
    f.write(data_df.to_csv(index=None, header=False))'''
'''df = ohlcv('MMM', 1000, date)
populate_df(df)
print(df.columns)'''
        

#XOM invalid value warning
    


'''f = open('data.csv', 'w+')
f.write(df.to_csv(index=None))
df = pd.read_csv('data.csv')'''





