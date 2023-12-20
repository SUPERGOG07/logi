import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np

df = pd.read_csv('./data/svmdata2.csv')
# print(df['X1'])

svc = svm.SVC(C=100, gamma=10, probability=True)
x = df[['X1', 'X2']]
y = df['y']
svc.fit(x, y)
df['SVM Confidence'] = svc.decision_function(x)

fig, ax = plt.subplots(figsize=(16, 9))
plt.axis('equal')

plt.scatter(df['X1'], df['X2'], c=-df['SVM Confidence'], cmap='Reds')
plt.colorbar()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Decision Function Confidence')
plt.show()