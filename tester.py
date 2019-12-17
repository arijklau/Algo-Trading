import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from sklearn.preprocessing import StandardScaler
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

universe = ['MMM', 'AXP','AAPL','BA','CAT','CVX','CSCO','KO','DIS','DOW','XOM',
            'GS','HD','IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE',
            'PG','TRV','UTX','UNH','VZ','V','WMT','WBA', 'AAN', 'ACM', 'ATR',
            'BKH', 'BRO', 'BYD', 'CAKE', 'CBSH', 'CHE', 'CIEN', 'DDS', 'DNKN',
            'ENR', 'ERI', 'EVR', 'FFIN', 'FLR', 'GATX', 'HWC', 'HOMB', 'IDA',
            'INGR', 'JBL', 'LECO', 'LM', 'MAN', 'MDP', 'NNN', 'NYT', 'OII',
            'PACW', 'PB', 'R', 'RYN', 'SBNY', 'SFM', 'STLD', 'TOL', 'URBN',
            'WERN', 'Y', 'ZBRA']

s = 20
rand_universe = np.random.choice(len(universe), size=s, replace=False)
data = np.zeros((1343*s, 1901))
k = 0
for i in rand_universe:
    print("Processing " + str(k+1) + "/" + str(len(universe)))
    file = open('data/' + universe[i] + '.csv')
    file_length = len(file.read().split('\n'))
    file.seek(0)
    for j in range(file_length):
        f = file.readline().split(',')
        for e in range(len(f)):
            try:
                f[e] = float(f[e])
            except:
                f[e] = 0.0
        data[1343*k + j] = f
    k += 1

print("Data Loaded. Converting to DataFrame...")

df = pd.DataFrame(data, columns=None)
df = df[(df.T != 0).any()]
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

print("DataFrame Loaded. Size:", df.shape)
print("Scaling Data...")
y = df[1900]
df = df.drop(columns=[1900])
scaler = StandardScaler()
X = scaler.fit_transform(df)
dfScaled = pd.DataFrame(X, columns=None)

print("Scaled Data")

#X = X.reshape(-1, 100, 19)

def relabel(a):
    y = np.zeros((a.shape[0], 3))
    for i in range(a.shape[0]):
        if(a[i] <= -2): #sell if -2 or lower
            y[i] = [1, 0, 0]
        if(a[i] <= 4): #hold if -2 < f(x) <= 4
            y[i] = [0, 1, 0]
        if(a[i] > 4): #buy if 4 < f(x)
            y[i] = [0, 0, 1]
    return y

yLabeled = relabel(y.values)

lstm = Sequential()

lstm.add(LSTM(units = 100, return_sequences = True, input_shape = (100, 19), batch_size=1))
lstm.add(Dropout(0.2))

lstm.add(LSTM(units = 50, return_sequences = True))
lstm.add(Dropout(0.2))

lstm.add(LSTM(units = 50, return_sequences = True))
lstm.add(Dropout(0.2))
lstm.add(Flatten())
lstm.add(Dense(units = 3)) #sell, hold, buy

lstm.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['acc'])
lstm.load_weights('lstm_weights3.h5')

for j in range(10):
    modelChoice = 100000
    modelPass = 100000
    totalGainPct = 0
    totalGains = 0
    totalLossPct = 0
    totalLosses = 0
    totalBuys = 0
    for i in range(60):
        idx = np.random.randint(0, dfScaled.shape[0], 2)
        x1 = dfScaled.iloc[idx[0]]
        x2 = dfScaled.iloc[idx[1]]
        y1_change = y.values[idx[0]]
        y2_change = y.values[idx[1]]
        y1_pred = lstm.predict(x1.values.reshape(-1, 100, 19))[0]
        y2_pred = lstm.predict(x2.values.reshape(-1, 100, 19))[0]
        y1 = np.argmax(y1_pred)
        y2 = np.argmax(y2_pred)
        if((y1 == 2) or (y2 == 2)):
            totalBuys += 1
            if(y1_pred[2] >= y2_pred[2]):
                #print("Bought", x1['ticker'], "for a change of", str(y1_change) + '%')
                modelChoice *= 1 + (y1_change/100)
                modelPass *= 1 + (y2_change/100)
                if(y1_change > 0):
                    totalGainPct += y1_change
                    totalGains += 1
                else:
                    totalLossPct += y1_change
                    totalLosses += 1
            else:
                #print("Bought", x2['ticker'], "for a change of", str(y2_change) + '%')
                modelChoice *= 1 + (y2_change/100)
                modelPass *= 1 + (y1_change/100)

                if(y2_change > 0):
                    totalGainPct += y2_change
                    totalGains += 1
                else:
                    totalLossPct += y2_change
                    totalLosses += 1

    print("Model's Average Earnings:", modelChoice)
    print("Unbought Stock's Average Earnings:", modelPass)
    print("Model chose stocks that outperformed by", str(100*(modelChoice/modelPass)) + '%')
    print("Over 60 trade opportunities, the model bought", totalBuys, "times")
    print("Trades rose", totalGains, "times for an average gain of", totalGainPct/totalGains)
    print("Trades dipped", totalLosses,"time for an average loss of", totalLossPct/totalLosses)



