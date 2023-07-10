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


raw_data = pd.read_csv("123181.csv", index_col= "Timestamp")
# raw_data.index = pd.to_datetime(raw_data.index)


y = raw_data['Predict_3min_mean']

X = raw_data[['Open',
            'High',
            'Low',
            'Volume',
            'Turnover',
            'Close']]

# Split raw dataset into train and test with seed 114514
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=seed)

sma = talib.SMA(X_train_raw['Close'], timeperiod=20)
rsi = talib.RSI(X_train_raw['Close'], timeperiod=27)
natr = talib.NATR(X_train_raw['High'],X_train_raw['Low'],X_train_raw['Close'],timeperiod=23)
Minus_DI = talib.MINUS_DI(X_train_raw['High'],X_train_raw['Low'],X_train_raw['Close'],timeperiod=25)
Plus_DI = talib.PLUS_DI(X_train_raw['High'],X_train_raw['Low'],X_train_raw['Close'],timeperiod=20)
bbands = talib.BBANDS(X_train_raw['Close'], timeperiod=20)

print(bbands)

# Merge X and y, clean NA values

# , natr, Minus_DI, Plus_DI, bbands, y_train
train = pd.concat([sma, 
                   rsi, 
                   natr, 
                   Minus_DI, 
                   Plus_DI, 
                   bbands[1],
                   y_train
                   ], ignore_index=False, join='inner',axis=1)
print(train)
train = train.rename(columns={train.columns[0]: 'SMA_20',
                              train.columns[1]: 'RSI_27',
                              train.columns[2]: 'NATR_23',
                              train.columns[3]: 'Minus_DI_25',
                              train.columns[4]: 'Plus_DI_25',
                              train.columns[5]: 'BBANDS_20',
                              })
train = train.dropna()
print(train)


# Apply the same to test sub dataset

sma = talib.SMA(X_test_raw['Close'], timeperiod=20)
rsi = talib.RSI(X_test_raw['Close'], timeperiod=27)
natr = talib.NATR(X_test_raw['High'],X_test_raw['Low'],X_test_raw['Close'],timeperiod=23)
Minus_DI = talib.MINUS_DI(X_test_raw['High'],X_test_raw['Low'],X_test_raw['Close'],timeperiod=25)
Plus_DI = talib.PLUS_DI(X_test_raw['High'],X_test_raw['Low'],X_test_raw['Close'],timeperiod=20)
bbands = talib.BBANDS(X_test_raw['Close'], timeperiod=20)

test = pd.concat([sma, 
                   rsi, 
                   natr, 
                   Minus_DI, 
                   Plus_DI, 
                   bbands[1],
                   y_test
                   ], ignore_index=False, join='inner',axis=1)

test = test.rename(columns={test.columns[0]: 'SMA_20',
                              test.columns[1]: 'RSI_27',
                              test.columns[2]: 'NATR_23',
                              test.columns[3]: 'Minus_DI_25',
                              test.columns[4]: 'Plus_DI_25',
                              test.columns[5]: 'BBANDS_20',
                              })

test = test.dropna()
print(test)

# Apply LR model
train.to_csv("train_20.csv")
test.to_csv('test_20.csv')


model = LinearRegression()
model.fit(train[['SMA_20', 'RSI_27', 'NATR_23','Minus_DI_25', 'Plus_DI_25','BBANDS_20']], train['Predict_3min_mean'])

# Calculate prediction from model
predictions = model.predict(test[['SMA_20', 'RSI_27', 'NATR_23','Minus_DI_25', 'Plus_DI_25','BBANDS_20']])

print(predictions)

# Results
print(model.coef_)
print("MSE: %5f"%mean_squared_error(test['Predict_3min_mean'],predictions))
print("R^2: %5f"%r2_score(test['Predict_3min_mean'],predictions))

# plt.scatter(test['AROONU_18'], test['Predict_1min'], color='black')

# plt.plot(test['AROONU_18'], predictions, color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())
# plt.show()