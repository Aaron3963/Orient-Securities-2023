import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tuneta.config import *

from tuneta.tune_ta import TuneTA


if __name__ == "__main__":
    data = pd.read_csv("123181.csv", index_col= "Timestamp")
    data.index = pd.to_datetime(data.index)
    
    seed = 114514

    # print(data.index)

    x = data[['Open',
            'High',
            'Low',
            'Volume',
            'Turnover',
            'Close']]

    y = data['Predict_3min_mean']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=seed)
   

    
    indicator_df = pd.DataFrame()

    tt = TuneTA(n_jobs=8, verbose=True)

    try:
        tt.fit(X_train, y_train,
        indicators=['tta.ADOSC','pta.mfi','tta.NATR','tta.ROC','tta.MOM','tta.'],
        ranges=[(4, 30)],
        trials=500,
        early_stop=100,
        min_target_correlation=.05)
    except ValueError as e:
            print("An error occurred:", str(e))
            print("!!!!!!!!!!!!!!!!!!!! Skipping to next param !!!!!!!!!!!!!!!!!!")
            

    tt.report(target_corr=True, features_corr=True) 

    tt.prune(max_inter_correlation=.7)

    features = tt.transform(X_train)
    
    report = tt.report(target_corr=True, features_corr=False)        