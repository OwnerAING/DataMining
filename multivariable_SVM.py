#-*- coding:utf-8-*-
from __future__ import division
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn

# 数据
column_names = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses','Type']
data = pd.read_csv(r'C:\Users\AING\Desktop\Logistics_Regression\Dataset\breast_cancer.csv')

f1_list = []
for i in range(1,10):
    x_train, x_test, y_train, y_test = train_test_split(data[column_names[0:8]], data[column_names[9]], train_size=0.75)
    # SVM拟合，使用高斯核函数
    clf = SVC(kernel='rbf')
    clf.fit(x_train, y_train)
    # 分类准确率
    score = clf.score(x_test, y_test)

    # 每个样本距离决策超平面的距离
    #print clf.decision_function(x_train)
    # 获取类的超参数等（并非训练后的参数）
    coef = clf.get_params()
    #print coef

    # 拉格朗日乘子
    dual_coef = clf.dual_coef_
    #print dual_coef

    # 正确率
    print '准确率', score
    # 支持向量索引
    # print clf.support_

    # 预测数据
    y_test_hat = clf.predict(x_test)
    y_train_hat = clf.predict(x_train)
    # F1分数
    f1 = sklearn.metrics.f1_score(y_test, y_test_hat)
    print 'F1分数', f1
    f1_list.append(f1)

    print '训练集混淆矩阵', sklearn.metrics.confusion_matrix(y_train, y_train_hat)
    print '测试集混淆矩阵',sklearn.metrics.confusion_matrix(y_test,y_test_hat)
    print '报告',sklearn.metrics.classification_report(y_test,y_test_hat)

print '平均F1分数',np.array(f1_list).mean()



