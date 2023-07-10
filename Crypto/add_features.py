import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tuneta.config import *

from tuneta.tune_ta import TuneTA
import talib
from finta import TA as fta
import pandas_ta as pta

seed = 114514

raw_X_train = pd.read_csv("Crypto/X_train.csv", index_col= "datetime")
raw_X_test = pd.read_csv("Crypto/X_test.csv", index_col= "datetime")

raw_X_train.index = pd.to_datetime(raw_X_train.index)
raw_X_test.index = pd.to_datetime(raw_X_test.index)


raw_y_train = pd.read_csv("Crypto/y_train.csv", index_col= "datetime")
raw_y_test = pd.read_csv("Crypto/y_test.csv", index_col= "datetime")

raw_y_train.index = pd.to_datetime(raw_y_train.index)
raw_y_test.index = pd.to_datetime(raw_y_test.index)



# Split raw dataset into train and test with seed 114514

sma_short = talib.SMA(raw_X_train['Close'], timeperiod=8)
sma_long = talib.SMA(raw_X_train['Close'], timeperiod=63)

rsi_short = talib.RSI(raw_X_train['Close'], timeperiod=26)
rsi_long = talib.RSI(raw_X_train['Close'], timeperiod=97)

natr = talib.NATR(raw_X_train['High'],raw_X_train['Low'],raw_X_train['Close'],timeperiod=26)

Minus_DI = talib.MINUS_DI(raw_X_train['High'],raw_X_train['Low'],raw_X_train['Close'],timeperiod=26)

Plus_DI = talib.PLUS_DI(raw_X_train['High'],raw_X_train['Low'],raw_X_train['Close'],timeperiod=59)

bbands = talib.BBANDS(raw_X_train['Close'], timeperiod=15)

print(bbands)

# Merge X and y, clean NA values

# , natr, Minus_DI, Plus_DI, bbands, y_train
train = pd.concat([sma_short, 
                   sma_long,
                   rsi_short,
                   rsi_long, 
                   natr, 
                   Minus_DI, 
                   Plus_DI, 
                   bbands[1],
                   raw_y_train['Target']
                   ], ignore_index=False, join='inner',axis=1)
print(train)
train = train.rename(columns={train.columns[0]: 'SMA_8',
                              train.columns[1]: 'SMA_63',
                              train.columns[2]: 'RSI_26',
                              train.columns[3]: 'RSI_97',
                              train.columns[4]: 'NATR_26',
                              train.columns[5]: 'Minus_DI_26',
                              train.columns[6]: 'Plus_DI_59',
                              train.columns[7]: 'BBANDS_15',
                              })
train = train.dropna()
print(train)

train.to_csv('Crypto/train_with_F.csv')


# Apply the same to test sub dataset

sma_short = talib.SMA(raw_X_test['Close'], timeperiod=8)
sma_long = talib.SMA(raw_X_test['Close'], timeperiod=63)

rsi_short = talib.RSI(raw_X_test['Close'], timeperiod=26)
rsi_long = talib.RSI(raw_X_test['Close'], timeperiod=97)

natr = talib.NATR(raw_X_test['High'],raw_X_test['Low'],raw_X_test['Close'],timeperiod=26)

Minus_DI = talib.MINUS_DI(raw_X_test['High'],raw_X_test['Low'],raw_X_test['Close'],timeperiod=26)

Plus_DI = talib.PLUS_DI(raw_X_test['High'],raw_X_test['Low'],raw_X_test['Close'],timeperiod=59)

bbands = talib.BBANDS(raw_X_test['Close'], timeperiod=15)
test = pd.concat([sma_short, 
                sma_long,
                rsi_short,
                rsi_long, 
                natr, 
                Minus_DI, 
                Plus_DI, 
                bbands[1],
                raw_y_test['Target']
                ], ignore_index=False, join='inner',axis=1)

test = test.rename(columns={test.columns[0]: 'SMA_8',
                              test.columns[1]: 'SMA_63',
                              test.columns[2]: 'RSI_26',
                              test.columns[3]: 'RSI_97',
                              test.columns[4]: 'NATR_26',
                              test.columns[5]: 'Minus_DI_26',
                              test.columns[6]: 'Plus_DI_59',
                              test.columns[7]: 'BBANDS_15',
                              })

test = test.dropna()
print(test)
test.to_csv('Crypto/test_with_F.csv')
