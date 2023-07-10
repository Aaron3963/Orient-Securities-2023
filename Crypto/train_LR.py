import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tuneta.config import *

# seaborn #

# scatter_plot()

##
    ## target close
    

    ## x: close, avg, xx




## 1min  5min 10min  15min 

#####  return related ######
## hist_ret(param = 10):

# df['Close'] / df['Close'].shift(param)
# df['Avg'] = df['turnover'] / df['volume']
# df['High'] + df['Low'] + df['Close'] + df['Open']

# hist_ret(5) - hist_ret(10): 

# log_ret_1m => volitity


## vol related 
# -> rolling(N).std()
# -> rolling(N).max() - rolling(N).min()

# hist_ret(5) * volitity


## macd related 

## kdj 

## bolling band 


for pred_len in [1,5,10,15]:
    rolling_win = pred_len * 5

seed = 114514

train = pd.read_csv('Crypto/train_with_F.csv',index_col='datetime')
test = pd.read_csv('Crypto/test_with_F.csv', index_col='datetime')

print(train['Target'].describe() )

X_COLS = ['SMA_8','SMA_63', 
                 'RSI_26', 
                 'RSI_97', 
                 'NATR_26', 
                 'Minus_DI_26',
                 'Plus_DI_59', 
                 'BBANDS_15']


for i in X_COLS:
    print(i, " corr: ", train[i].corr(train['Target']) )


# train['Target'].hist()


model = Ridge(alpha=0.1) ##LinearRegression()
model.fit(train[['SMA_8', 
                 'SMA_63', 
                 'RSI_26', 
                 'RSI_97', 
                 'NATR_26', 
                 'Minus_DI_26',
                 'Plus_DI_59', 
                 'BBANDS_15']], train['Target'])

# Calculate prediction from model
predictions = model.predict(test[['SMA_8', 
                                'SMA_63', 
                                'RSI_26', 
                                'RSI_97', 
                                'NATR_26',
                                'Minus_DI_26',
                                'Plus_DI_59', 
                                'BBANDS_15']])

print(predictions)

# Results
print(model.coef_)
print("MSE: %5f"%mean_squared_error(test['Target'],predictions))
print("R^2: %5f"%r2_score(test['Target'],predictions))

# plt.scatter(test['AROONU_18'], test['Predict_1min'], color='black')

# plt.plot(test['AROONU_18'], predictions, color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())
# plt.show()
