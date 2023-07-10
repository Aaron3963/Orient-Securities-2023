import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tuneta.config import *

from tuneta.tune_ta import TuneTA


if __name__ == "__main__":
    X_data = pd.read_csv("Crypto/X_train.csv", index_col= "datetime")
    X_data.index = pd.to_datetime(X_data.index)

    y_data = pd.read_csv('Crypto/y_train.csv', index_col='datetime')
    y_data.index = pd.to_datetime(y_data.index)
    
    seed = 114514

    # print(data.index)

    x = X_data[['Count',
              'Open',
              'High',
              'Low',
              'Close',
              'Volume',
              'VWAP']]

    y = y_data['Target']


    tt = TuneTA(n_jobs=8, verbose=True)

    try:
        tt.fit(x, y,
        # indicators=['tta.SMA','tta.RSI','tta.NATR','tta.MINUS_DI','tta.PLUS_DI','tta.BBANDS'],
        
        indicators=['tta.NATR'],

        ranges=[(4,40),(40,150)],
        trials=200,
        early_stop=20)
    except ValueError as e:
            print("An error occurred:", str(e))
            print("!!!!!!!!!!!!!!!!!!!! Skipping to next param !!!!!!!!!!!!!!!!!!")
            

    tt.report(target_corr=True, features_corr=True) 

    # tt.prune(max_inter_correlation=.7)

    # features = tt.transform(X_train)