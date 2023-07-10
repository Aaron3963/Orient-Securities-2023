import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tuneta.config import *

from tuneta.tune_ta import TuneTA

path = "results_Full_X.csv"

df = pd.read_csv(path)

# Create a DataFrame from the sample data
# df = pd.DataFrame(data)

# Convert the 'timestamp' column to datetime format
#df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%Y/%m/%d %H:%M")

# Sort the DataFrame by the 'timestamp' column
df_sorted = df.sort_values(by='Correlation_trained', ascending=False) 

# Display the sorted DataFrame
print(df_sorted)

df_sorted.to_csv("results_Full_X_sorted.csv", index=False)
