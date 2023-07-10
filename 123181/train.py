import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tuneta.config import *



if __name__ == "__main__":
    path = "C:/Users/Aaron/OneDrive/Internships and Jobs/2023 暑假/东方证券/123181/123181.csv"
    data = pd.read_csv(path, index_col= "Timestamp")
    data.index = pd.to_datetime(data.index)

    print(data.index)

    # x = data[['Open',
    #         'High',
    #         'Low',
    #         'Volume',
    #         'Turnover',
    #         'Close',
    #         'HighHalf']]

    x = data[['HighHalf']]

    y = data['Predict_1min']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


    model = LinearRegression()
    model.fit(X_train, y_train)


    predictions = model.predict(X_test)


    print(model.coef_)
    print("MSE: %5f"%mean_squared_error(y_test,predictions))
    print("R^2: %5f"%r2_score(y_test,predictions))


    plt.scatter(y_test,predictions)
    plt.show()