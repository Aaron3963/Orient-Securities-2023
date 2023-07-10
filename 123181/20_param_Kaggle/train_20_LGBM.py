import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score# mas 绝对值误差均值
# from sklearn.preprocessing import Imputer # 数据的预处理，一般是特征缩放和特征编码


raw_train = pd.read_csv('train_20.csv')
raw_test = pd.read_csv('test_20.csv')

# 2.切分数据输入：特征 输出：预测目标变量


# 3.切分训练集、测试集,切分比例7.5 : 2.5
train_X = raw_train.drop(['Predict_3min_mean'], axis=1)
test_X = raw_test.drop(['Predict_3min_mean'], axis=1)

train_y = raw_train['Predict_3min_mean']
test_y = raw_test['Predict_3min_mean']

# 4.空值处理，默认方法：使用特征列的平均值进行填充
# my_imputer = Imputer()
# train_X = my_imputer.fit_transform(train_X)
# test_X = my_imputer.transform(test_X)

# 5.转换为Dataset数据格式
lgb_train = lgb.Dataset(train_X, train_y)
lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train)

# 6.参数
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

# 7.调用LightGBM模型，使用训练集数据进行训练（拟合）
# Add verbosity=2 to print messages while running boosting
my_model = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)

# 8.使用模型对测试集数据进行预测
predictions = my_model.predict(test_X, num_iteration=my_model.best_iteration)

# 9.对模型的预测结果进行评判（平均绝对误差）
print("MAE: " + str(mean_absolute_error(test_y, predictions)))
print("MSE: %5f"%mean_squared_error(test_y, predictions))
print("R^2: %5f"%r2_score(test_y,predictions))