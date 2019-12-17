import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

data = np.zeros((1343*72, 1901))
for i in range(len(universe)):
    print("Processing " + str(i+1) + "/" + str(len(universe)))
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
        data[1343*i + j] = f

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

print("Scaled Data")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 100, 19)
X_test = X_test.reshape(-1, 100, 19)

print("Reshaped Data")

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

y_train = relabel(y_train.values)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten

lstm = Sequential()

lstm.add(LSTM(units = 100, return_sequences = True, input_shape = (100, 19), batch_size=25))
lstm.add(Dropout(0.2))

lstm.add(LSTM(units = 50, return_sequences = True))
lstm.add(Dropout(0.2))

lstm.add(LSTM(units = 50, return_sequences = True))
lstm.add(Dropout(0.2))
lstm.add(Flatten())
lstm.add(Dense(units = 3)) #sell, hold, buy

lstm.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['acc'])
lstm.load_weights('lstm_weights2.h5')
lstm.fit(X_train[0:73300], y_train[0:73300], epochs = 20, batch_size = 25)
print("Model Trained")
lstm.save_weights('lstm_weights3.h5')
lstm.save('lstm_model.h5')

lstm.evaluate(X_test[0:18300], relabel(y_test[0:18300].values), batch_size=25)