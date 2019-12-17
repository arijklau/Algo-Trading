import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from sklearn.preprocessing import StandardScaler
import alpaca
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
universe = ['MMM', 'AXP','AAPL','BA','CAT','CVX','CSCO','KO','DIS','XOM',
            'GS','HD','IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE',
            'PG','TRV','UTX','UNH','VZ','V','WMT','WBA', 'AAN', 'ACM', 'ATR',
            'BKH', 'BRO', 'BYD', 'CAKE', 'CBSH', 'CHE', 'CIEN', 'DDS', 'DNKN',
            'ENR', 'ERI', 'EVR', 'FFIN', 'FLR', 'GATX', 'HWC', 'HOMB', 'IDA',
            'INGR', 'JBL', 'LECO', 'LM', 'MAN', 'MDP', 'NNN', 'NYT', 'OII',
            'PACW', 'PB', 'R', 'RYN', 'SBNY', 'SFM', 'STLD', 'TOL', 'URBN',
            'WERN', 'Y', 'ZBRA']

data = np.zeros((len(universe), 1900))
streamer = alpaca.AlpacaStreamer()
for i in range(len(universe)):
    data[i] = streamer.loadData(universe[i]).values.flatten()

scaler = StandardScaler()
X = scaler.fit_transform(data)

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

predMap = {0:'Sell', 1:'Hold', 2:'Buy'}

class Predictor:
    
    def getPredictions(self):
        buys = []
        for i in range(X.shape[0]):
            prediction = lstm.predict(X[i].reshape(-1, 100, 19))[0]
            if(np.argmax(prediction) == 2):
                buys.append(universe[i])
        return buys

