# author: Liu Junhui

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 设置中文字体为黑体或其他中文字体
plt.rcParams['font.sans-serif'] = 'SimHei'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 读取 MNIST 数据集
data = pd.read_csv("mnist-train.csv")

# 提取特征和目标变量
X = data.iloc[:, 1:-1]  # 特征
y = data.iloc[:, 0]  # 目标变量

print(np.shape(X.head()))

# 选择数字为 8 和 9 的样本
X = X[(y == 8) | (y == 9)]
y = y[(y == 8) | (y == 9)]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立逻辑回归模型并训练数据
model = LogisticRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)

# 可视化预测结果
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"预测值: {y_pred[i]}", fontproperties='SimHei')
    plt.axis('off')
plt.tight_layout()
plt.show()
