import sklearn
from sklearn import datasets
import sklearn.datasets as sk_datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearnex import patch_sklearn
patch_sklearn()


iris = sk_datasets.load_iris()
iris_X = iris.data #导入数据
iris_y = iris.target #导入标签

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2)

#数据处理/cleaning
StandardScaler().fit_transform(iris.data)

knn = KNeighborsClassifier()
#进行训练
knn.fit(X_train,y_train)
# 使用训练好的knn进行数据预测
predict = knn.predict(X_test)
ground = y_test

# 准确率
numSuccess = 0

for index in range(len(predict)):
    if predict[index] == ground[index]:
        numSuccess +=1

print("Score: %.5f" %(numSuccess/len(predict)))

