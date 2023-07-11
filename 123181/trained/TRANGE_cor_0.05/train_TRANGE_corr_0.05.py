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

trained_X = pd.read_csv("Indicators_v2_Full_X.csv", index_col= "Timestamp")

y = raw_data['Predict_1min']

X = raw_data[['Open',
            'High',
            'Low',
            'Volume',
            'Turnover',
            'Close']]

# Split raw dataset into train and test with seed 114514
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=seed)

# Apply testing param "AROON"
X_train = talib.TRANGE(X_train_raw['High'],X_train_raw['Low'],X_train_raw['Close'])
# Merge X and y, clean NA values
train = pd.concat([X_train, y_train], ignore_index=False, join='inner',axis=1)
train = train.rename(columns={train.columns[0]: 'TRANGE'})
print(train)
train = train.dropna(subset = ['TRANGE'])
print(train)

train.to_csv("TRANGE.csv")


# Apply the same to test sub dataset
X_test = talib.TRANGE(X_test_raw['High'],X_test_raw['Low'],X_test_raw['Close'])
test = pd.concat([X_test, y_test], ignore_index=False, join='inner',axis=1)
test = test.rename(columns={test.columns[0]: 'TRANGE'})

print(test)
test = test.dropna(subset = ['TRANGE'])
print(test)

# Apply LR model
model = LinearRegression()
model.fit(train[['TRANGE']], train['Predict_1min'])

# Calculate prediction from model
predictions = model.predict(test[['TRANGE']])

print(predictions)

# Results
print(model.coef_)
print("MSE: %5f"%mean_squared_error(test['Predict_1min'],predictions))
print("R^2: %5f"%r2_score(test['Predict_1min'],predictions))

plt.scatter(test['TRANGE'], test['Predict_1min'], color='black')

plt.plot(test['TRANGE'], predictions, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()