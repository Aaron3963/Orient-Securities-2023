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

y = raw_data['Predict_3min_mean']

X = raw_data[['Open',
            'High',
            'Low',
            'Volume',
            'Turnover',
            'Close']]

# Split raw dataset into train and test with seed 114514
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=seed)

# Apply testing param "AROON"
MFI = pta.mfi(X_train_raw['High'],X_train_raw['Low'],X_train_raw['Close'],X_train_raw['Volume'], length=18)
ADOSC = talib.ADOSC(X_train_raw['High'],X_train_raw['Low'],X_train_raw['Close'],X_train_raw['Volume'], fastperiod=21, slowperiod=7)
NATR = pta.natr(X_train_raw['High'],X_train_raw['Low'],X_train_raw['Close'],length=25)
MOM = pta.mom(X_train_raw['Close'], length=18)
ROC = pta.roc(X_train_raw['Close'], length=12)

# Merge X and y, clean NA values
train = pd.concat([ADOSC, MFI, NATR, MOM, ROC, y_train], ignore_index=False, join='inner',axis=1)
train = train.rename(columns={train.columns[0]: 'ADOSC'})
train = train.dropna()
print(train)

# Apply the same to test sub dataset
MFI = pta.mfi(X_test_raw['High'],X_test_raw['Low'],X_test_raw['Close'],X_test_raw['Volume'], length=18)
ADOSC = talib.ADOSC(X_test_raw['High'],X_test_raw['Low'],X_test_raw['Close'],X_test_raw['Volume'], fastperiod=21, slowperiod=7)
NATR = pta.natr(X_test_raw['High'],X_test_raw['Low'],X_test_raw['Close'],length=25)
MOM = pta.mom(X_test_raw['Close'], length=18)
ROC = pta.roc(X_test_raw['Close'], length=12)
test = pd.concat([ADOSC, MFI, NATR, MOM, ROC, y_test], ignore_index=False, join='inner',axis=1)
test = test.rename(columns={test.columns[0]: 'ADOSC'})
test = test.dropna()
print(test)

# Apply LR model
model = LinearRegression()
model.fit(train[['ADOSC', 'MFI_18', 'NATR_25', 'MOM_18', 'ROC_12']], train['Predict_3min_mean'])

# Calculate prediction from model
predictions = model.predict(test[['ADOSC', 'MFI_18', 'NATR_25', 'MOM_18', 'ROC_12']])

print(predictions)

# Results
print(model.coef_)
print("MSE: %5f"%mean_squared_error(test['Predict_3min_mean'],predictions))
print("R^2: %5f"%r2_score(test['Predict_3min_mean'],predictions))

# plt.scatter(test['AROONU_18'], test['Predict_3min_mean'], color='black')

# plt.plot(test['AROONU_18'], predictions, color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())
# plt.show()