import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 114514

raw_ETH = pd.read_csv('Crypto/ETH.csv')
raw_ETH = raw_ETH.drop(['timestamp'], axis= 1)

raw_ETH = raw_ETH.set_index("datetime")

raw_ETH.index = pd.to_datetime(raw_ETH.index)


print(raw_ETH)

y = raw_ETH[['Target']]
X = raw_ETH.drop(['Target'], axis= 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=SEED)

# X_train = X_train.set_index("datetime")
# X_test = X_test.set_index("datetime")
# y_train = y_train.set_index("datetime")
# y_test = y_test.set_index("datetime")

print(X_train)
X_train.to_csv('Crypto/X_train.csv',)

print(X_test)
X_test.to_csv('Crypto/X_test.csv')

print(y_train)
y_train.to_csv('Crypto/y_train.csv')

print(y_test)
y_test.to_csv('Crypto/y_test.csv')
