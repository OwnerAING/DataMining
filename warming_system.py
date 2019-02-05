# -*- coding:utf-8 -*-

from __future__ import division
import numpy as np
from scipy.integrate import quad
import pandas as pd
data = pd.read_csv(r'C:\Users\AING\Desktop\data.csv')

CI1 = data['CI1'].values.tolist()
CI2 = data['CI2'].values.tolist()

CI1_list = []
CI2_list = []

mean = [100,100]
var = [2.926772936,1.726904514]
fun1 = lambda x:(1/((2*np.pi*2.926772936)**(1/2)))*np.exp(-(x-100)**2/(2*2.926772936))

for i in CI1:
    x,err = quad(fun1,-np.inf,i)
    CI1_list.append(x*100)
    pd.DataFrame(CI1_list).to_csv(r'C:\Users\AING\Desktop\CI1_list.csv')

fun2 = lambda x:(1/((2*np.pi*1.726904514)**(1/2)))*np.exp(-(x-100)**2/(2*1.726904514))

for j in CI2:
    y, err = quad(fun2, -np.inf, j)
    CI2_list.append(y*100)
    pd.DataFrame(CI2_list).to_csv(r'C:\Users\AING\Desktop\CI2_list.csv')


