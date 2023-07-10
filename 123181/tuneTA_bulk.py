import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tuneta.config import *

from tuneta.tune_ta import TuneTA


if __name__ == "__main__":
    data = pd.read_csv("123181.csv", index_col= "Timestamp")
    data.index = pd.to_datetime(data.index)
    
    path = "Indicators_v2.csv"
    result_path = "results.csv"

    result = pd.read_csv(result_path)

    seed = 114514

    # print(data.index)

    x = data[['Open',
            'High',
            'Low',
            'Volume',
            'Turnover',
            'Close']]

    y = data['Predict_1min']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=seed)
   
    indicators = talib_indicators + pandas_ta_indicators + finta_indicatrs

    # indicators = ['fta.RSI', 'fta.TR']
    
    indicator_df = pd.DataFrame()

    for indicator in indicators:

        tt = TuneTA(n_jobs=8, verbose=True)
    
        indicator_df = pd.read_csv("Indicators_v2.csv")

        try:
            tt.fit(X_train, y_train,
            # 优化指标
            indicators=[indicator],
            # 待优化参数的两个参数范围（时间的短期和长期）：4-30和31-180
            ranges=[(4, 30), (31, 180)],
            # 每个时间段最多100次试验，以搜索最佳指标参数
            trials=100,
            # 在每个时间段持续20次试验没有改善后停止搜索参数
            early_stop=50,
            )
        except KeyError as e:
            print("multiProcess.pool Error")
            continue

        # tt.report(target_corr=True, features_corr=True)        

        try:
            features = tt.transform(X_train)
           # Additional code...
        except ValueError as e:
            print("An error occurred:", str(e))
            print("!!!!!!!!!!!!!!!!!!!! Skipping to next param !!!!!!!!!!!!!!!!!!")
            continue


        # features['Timestamp'] = pd.to_datetime(features['Timestamp'], format="%Y/%m/%d %H:%M")

        features_sorted = features.sort_values(by='Timestamp', ascending=True)
        print(features_sorted.columns)

        # features_sorted = features_sorted.drop(columns='Timestamp')
        features_sorted = features_sorted.reset_index(drop=True)

        merged = pd.concat([indicator_df, features_sorted], axis = 1)
        print(merged.shape)

        merged.to_csv("Indicators_v2.csv", index=False)

        print("----------- %s column written to %s -----------"%(indicator,path))


        report = tt.report(target_corr=True, features_corr=False)        

        for index, row in report[0].iterrows():
            # Access the index and individual cells
            param_Name = index
            corr = row['Correlation']

            result.loc[len(result)]= [param_Name, corr,'','','']
            result.to_csv("results.csv", index=False)


        print("----------- Logged into results.csv -----------")
