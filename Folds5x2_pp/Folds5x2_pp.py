# author: Liu Junhui

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 设置中文字体为黑体或其他中文字体
plt.rcParams['font.sans-serif'] = 'SimHei'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 读取数据集
data = pd.read_csv("Folds5x2_pp.csv")

# 划分特征和目标变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立线性回归模型并训练数据
model = LinearRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 打印评估结果
print("均方误差 (MSE):", mse)
print("决定系数 (R^2):", r2)

# 可视化模型评估结果
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-')
plt.xlabel('实际值', fontproperties='SimHei')
plt.ylabel('预测值', fontproperties='SimHei')
plt.title('线性回归模型评估', fontproperties='SimHei')
plt.show()
