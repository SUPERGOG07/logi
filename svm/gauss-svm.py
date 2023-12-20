import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv('./data/svmdata3.csv')

# 获取特征和目标变量
X = data[['X1', 'X2']]
y = data['y']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义参数搜索范围
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [0.01, 0.1, 1, 10, 100]
}

# 创建SVM模型和交叉验证对象
svm = SVC(kernel='rbf')
grid_search = GridSearchCV(svm, param_grid, cv=5)

# 在训练集上进行参数搜索
grid_search.fit(X_train, y_train)

# 获取最优参数和最优模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# 在测试集上评估最优模型
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 输出最优参数和测试集准确率
print("最优参数：", best_params)
print("测试集准确率：", accuracy)