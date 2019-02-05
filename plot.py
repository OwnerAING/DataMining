#_*_ coding:utf-8_*_
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\AING\Desktop\index.csv')
date = pd.to_datetime(data.date,format='%Y')

plt.plot(date,data['index'])
# 画直线
# plt.axvline(x = 40)
# plt.axvline(x = 50)
# plt.axvline(x = 70)
# plt.axvline(x = 80)

# 横行填充颜色
plt.axhspan(0, 40, facecolor='#0033CC', alpha=0.5)
plt.axhspan(40, 50, facecolor='#99CCFF', alpha=0.5)
plt.axhspan(50, 70, facecolor='#00CC00', alpha=0.5)
plt.axhspan(70, 80, facecolor='#FFCC00', alpha=0.5)
plt.axhspan(80, 100, facecolor='#CC0000', alpha=0.5)

# 纵向垂直填充
# plt.axvspan(1.25, 1.55, facecolor='#2ca02c', alpha=0.5)

plt.ylim(0,100)
plt.title("Comprehensive index early warning system")
plt.xlabel("Year")
plt.ylabel("Warning index")
plt.show()