import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tuneta.config import *

from tuneta.tune_ta import TuneTA


if __name__ == "__main__":
    df = pd.read_csv("正股+大盘+123181_1min/123181_train_with_Stock_F.csv", index_col= "Timestamp")
    df.index = pd.to_datetime(df.index)

    
    seed = 114514

    # print(data.index)

    x = df[['Turnover',
              'Open',
              'High',
              'Low',
              'Close',
              'Volume',
              ]]

    y = df['Target_Avg_1min']


    tt = TuneTA(n_jobs=8, verbose=True)

    try:
        tt.fit(x, y,
        # indicators=['tta.SMA','tta.RSI','tta.NATR','tta.MINUS_DI','tta.PLUS_DI','tta.BBANDS'],
        
        indicators=["tta"],

        ranges=[(4,40),(40,150)],
        trials=200,
        early_stop=20)
    except ValueError as e:
            print("An error occurred:", str(e))
            print("!!!!!!!!!!!!!!!!!!!! Skipping to next param !!!!!!!!!!!!!!!!!!")
            

    tt.report(target_corr=True, features_corr=True) 

    # tt.prune(max_inter_correlation=.7)

    # features = tt.transform(X_train)