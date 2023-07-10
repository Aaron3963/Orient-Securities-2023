import pandas as pd


title_df = pd.read_csv("Indicators.csv")
params_df = pd.read_csv("new.csv")

merged = pd.concat([title_df,params_df], axis=1)

merged.to_csv("Indicators.csv",index=False)