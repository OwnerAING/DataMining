#-*-coding:utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\AING\Desktop\v.csv')
v1 = data['V1']
v2 = data['V2']
# 初始化综合指标
I1_index = [100]
I2_index = [100]
for i in range(1,len(v1)):
    num1 = (I1_index[i - 1] * (200 + v1[i])) / (200 - v1[i])
    num2 = (I2_index[i - 1] * (200 + v2[i])) / (200 - v2[i])
    I1_index.append(num1)
    I2_index.append(num2)
print I1_index
print I2_index
I1 = pd.DataFrame(I1_index)
I2 = pd.DataFrame(I2_index)
I1.to_csv(r'C:\Users\AING\Desktop\i1.csv')
I2.to_csv(r'C:\Users\AING\Desktop\i2.csv')

# 合成指数
init_CI1 = (100*np.array(I1_index))/np.array(I1_index).mean()
init_CI2 = (100*np.array(I2_index))/np.array(I2_index).mean()

CI1 = pd.DataFrame(init_CI1)
CI2 = pd.DataFrame(init_CI2)

CI1.to_csv(r'C:\Users\AING\Desktop\CI1.csv')
CI2.to_csv(r'C:\Users\AING\Desktop\CI2.csv')