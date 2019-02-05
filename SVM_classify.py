#-*- coding:utf-8-*-

from __future__ import division
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import  matplotlib.pyplot as plt
import matplotlib as mlt

train_data = pd.read_csv(r'C:\Users\AING\Desktop\Logistics_Regression\Dataset\breast-cancer-train.csv')
test_data = pd.read_csv(r'C:\Users\AING\Desktop\Logistics_Regression\Dataset\breast-cancer-test.csv')

# 数据
X_train = train_data[['Clump Thickness','Cell Size']]
y_train = train_data['Type']
X_test = test_data[['Clump Thickness','Cell Size']]
y_test = test_data['Type']
# 散点图数据
test_positive = test_data.loc[test_data['Type'] == 1][['Clump Thickness','Cell Size']] #正样本
test_negative = test_data.loc[test_data['Type'] == 0][['Clump Thickness','Cell Size']] #负样本
# SVM拟合，使用线性核函数
clf = SVC(kernel='rbf')
clf.fit(X_train,y_train)
# 分类准确率
score = clf.score(X_test,y_test)
print clf.decision_function(X_train)
print clf.decision_function_shape
# 参数
# intercept = clf.intercept_
# coef = clf.coef_
# print '参数为',coef
# print '截距为',intercept
print '准确率',score
print '不同类别支持向量',clf.n_support_

M,N = 500,500
x1_min,x1_max = X_test['Clump Thickness'].min(),X_test['Clump Thickness'].max()
x2_min,x2_max = X_test['Cell Size'].min(),X_test['Cell Size'].max()

t1 = np.linspace(x1_min,x1_max,M)
t2 = np.linspace(x2_min,x2_max,N)


x1,x2 = np.meshgrid(t1,t2)

x = np.stack((x1.flatten(),x2.flatten()),axis=1)
y = clf.predict(x).reshape(x1.shape)

cm_light = mlt.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mlt.colors.ListedColormap(['g', 'r', 'b'])

plt.figure(facecolor='w')
plt.pcolormesh(x1,x2,y,cmap=cm_light)
plt.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(test_positive['Clump Thickness'],test_positive['Cell Size'],marker='x',s=150,c='black')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Gaussian kernel function')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.grid(True)
#plt.savefig(r"D:\gaussian_kernel.png")
plt.show()
