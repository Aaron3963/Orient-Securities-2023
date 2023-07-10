import os
import pandas as pd
import random
import numpy as np
import pandas_ta as pta

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

SEED = 114514


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
fix_all_seeds(SEED)



raw_data = pd.read_csv("C:/Users/Aaron/OneDrive/Internships and Jobs/2023 暑假/东方证券/123181/123181.csv", index_col= "Timestamp")



def calc_sma_diff_test(close, timeperiod_short, timeperiod_long):
    res_short = close[-timeperiod_short:].mean()
    res_long = close[-timeperiod_long:].mean()
    res = (res_long - res_short) / res_long
    return res

def calc_bbands_test(close, timeperiod, std=2):
    close_std = close[-timeperiod:].std(ddof=0)
    rol = close[-timeperiod:].mean()
    upper = rol + close_std * std
    lower = rol - close_std * std
    res = (upper - close[-1]) / (upper - lower)
    return res

def calc_atr_test(high, low, close, timeperiod):
    A = high[-timeperiod:] - close[-(timeperiod+1):-1]
    B = close[-(timeperiod+1):-1] - low[-timeperiod:]
    C = high[-timeperiod:] - low[-timeperiod:]
    res = np.vstack((A, B, C)).max(axis=0).mean()
    return res

def calc_natr_test(high, low, close, timeperiod):
    res = calc_atr_test(high, low, close, timeperiod) / close[-1]
    return res

def calc_minus_di_test(high, low, close, timeperiod):
    high_diff = np.diff(high[-(timeperiod+1):])
    low_diff = np.diff(low[-(timeperiod+1):])
    high_diff[(high_diff<0)] = 0
    low_diff[(low_diff<0)] = 0
#     high_diff[(high_diff<low_diff)] = 0
    low_diff[(high_diff>low_diff)] = 0
    tr = calc_atr_test(high, low, close, timeperiod)*timeperiod
    res = 100 * low_diff.sum() / tr
    return res

def calc_plus_di_test(high, low, close, timeperiod):
    high_diff = np.diff(high[-(timeperiod+1):])
    low_diff = np.diff(low[-(timeperiod+1):])
    high_diff[(high_diff<0)] = 0
    low_diff[(low_diff<0)] = 0
    high_diff[(high_diff<low_diff)] = 0
#     low_diff[(high_diff>low_diff)] = 0
    tr = calc_atr_test(high, low, close, timeperiod)*timeperiod
    res = 100 * high_diff.sum() / tr
    return res

def calc_log_ret_test(close, lag=0):
    return np.log(close[-(1+15*lag)] / close[-(1+15*(lag+1))])

def upper_shadow_15_perc_test(high, close, open_, lag=0):
    if lag:
        return high[-15*(lag+1):-15*lag].max() * 100 / np.maximum(close[-15*lag-1], open_[-15*(lag+1)-1])
    else:
        return high[-15:].max() * 100 / np.maximum(close[-1], open_[-16])

def lower_shadow_15_perc_test(low, close, open_, lag=0):
    if lag:
        return np.minimum(close[-15*lag-1], open_[-15*(lag+1)-1]) * 100 / low[-15*(lag+1):-15*lag].min()
    else:
        return np.minimum(close[-1], open_[-16]) * 100 / low[-15:].min()





def feature_generation(open_, high, low, close, volume):
    ### original def features
    pta.sm
    res.append(calc_sma_diff_test(close, 12, 26))
#     res.append(calc_sma_diff_test(close, 12*4*4, 24*4*4))
#     res.append(calc_sma_diff_test(close, 12*4*4*4, 24*4*4*4))
#     res.append(calc_sma_diff_test(close, 12*4*4*4*4, 24*4*4*4*4))
#     res.append(calc_sma_diff_test(volume, 12*4*4, 24*4*4))
#     res.append(calc_rsi_test(close, 14*4))
#     res.append(calc_natr_test(high, low, close, 14*4*4*4))
#     res.append(calc_minus_di_test(high, low, close, 14))
#     res.append(calc_minus_di_test(high, low, close, 14*4*4*4))
#     res.append(calc_plus_di_test(high, low, close, 14*4))
#     res.append(calc_plus_di_test(high, low, close, 14*4*4*4))
#     res.append(calc_log_ret_test(close))
#     res.append(calc_log_ret_test(close, lag=1))
#     res.append(calc_log_ret_test(close, lag=2))
#     res.append(upper_shadow_15_perc_test(high, close, open_))
#     res.append(lower_shadow_15_perc_test(low, close, open_))
#     res.append(calc_bbands_test(close, 5*4*2))
#     res.append(calc_bbands_test(close, 5*4*4))
#     res.append(calc_bbands_test(close, 5*4*4*2))
#     res.append(calc_bbands_test(close, 5*4*4*4))
    
    res = np.array(res).reshape([1, -1])
    return res


y = raw_data['Predict_1min']

X = raw_data[['Open',
            'High',
            'Low',
            'Volume',
            'Turnover',
            'Close']]

# Split raw dataset into train and test with seed 114514
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=SEED)

print(X_train_raw)



X_train = pd.DataFrame(index = X_train_raw.index)
for index,row in X_train_raw.iterrows():
    
    X_train[index] = get_features_test(row['Open'],row['High'],row['Low'],row['Close'],row['Volume'])
        
#     # Skipping row if errors like index-OOR
#     except:
#         print("Skip a line")
        
print(X_train)

