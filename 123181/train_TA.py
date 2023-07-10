import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tuneta.config import *


from finta import TA as fta

from tuneta.tune_ta import TuneTA

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

print(y)

NO_USE_TRAIN, x_test_raw, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=seed)

# X_train = trained_X['fta_BBANDS_period_48_0','fta_BBANDS_period_48_1','fta_BBANDS_period_48_2']

# for x in trained_X:
#     model = LinearRegression()

#     print(x)
#     print(y_train)

#     model.fit(x, y_train)

#     predictions = model.predict(  )
    
#     print(model.coef_)
#     print("MSE: %5f"%mean_squared_error(y_test,predictions))
#     print("R^2: %5f"%r2_score(y_test,predictions))

#     results.loc[len(results)] = {'param' : x,
#                                  'coef' : model.coef_,
#                                  'MSE' : mean_squared_error(y_test,predictions),
#                                  'R^2' : r2_score(y_test,predictions)
#                                  }
    
#     print("---------- Logged ----------")

x_test = fta.BBANDS(x_test_raw,48)

merged_test = pd.DataFrame({'BB_UPPER':x_test['BB_UPPER'],
                       'BB_MIDDLE':x_test['BB_MIDDLE'],
                       'BB_LOWER':x_test['BB_LOWER'],
                       'y_test': y_test
                       })


merged = pd.DataFrame({'BB_UPPER':trained_X['fta_BBANDS_period_48_0'],
                       'BB_MIDDLE':trained_X['fta_BBANDS_period_48_1'],
                       'BB_LOWER':trained_X['fta_BBANDS_period_48_2'],
                       'y_train':y_train})

cleaned = merged.dropna(subset=['BB_UPPER','BB_MIDDLE', 'BB_LOWER', 'y_train'])
cleaned_test = merged_test.dropna(subset=['BB_UPPER','BB_MIDDLE', 'BB_LOWER', 'y_test'])
print(cleaned)





model = LinearRegression()

model.fit(cleaned[['BB_UPPER','BB_MIDDLE', 'BB_LOWER']], cleaned['y_train'])


x_test_selected = pd.DataFrame({'BB_UPPER':cleaned_test['BB_UPPER'],
                       'BB_MIDDLE':cleaned_test['BB_MIDDLE'],
                       'BB_LOWER':cleaned_test['BB_LOWER'],})

x_test_selected.to_csv("cleaned.csv")
predictions = model.predict(x_test_selected)


print(model.coef_)
print("MSE: %5f"%mean_squared_error(cleaned_test['y_test'],predictions))
print("R^2: %5f"%r2_score(cleaned_test['y_test'],predictions))


plt.scatter(cleaned_test['y_test'],predictions)
plt.show()


# results.to_excel("Results.xlsx", index=False)