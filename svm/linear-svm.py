import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm

data1 = pd.read_csv('./data/svmdata1.csv')
# print(data1['X1'])
groups = data1.groupby('y')

svc = svm.LinearSVC(C=200, loss='hinge', max_iter=1000)
x = data1[['X1', 'X2']]
y = data1['y']
svc.fit(x, y)
data1['SVM 1 Confidence'] = svc.decision_function(x)

w = svc.coef_[0]
b = svc.intercept_[0]

plt.figure()
plt.axis('equal')

x_plot = np.linspace(0, 4.5, 100)
y_plot = (-w[0] * x_plot - b) / w[1]
plt.plot(x_plot, y_plot, color='grey')

plt.scatter(data1['X1'], data1['X2'], c=data1['SVM 1 Confidence'], cmap='coolwarm')
plt.colorbar()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Decision Function Confidence')
plt.show()
