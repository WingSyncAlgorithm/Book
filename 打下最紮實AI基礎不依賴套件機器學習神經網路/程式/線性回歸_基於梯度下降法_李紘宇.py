import os
import numpy as np
import random
import matplotlib.pyplot as plt

# 分別對y和x建立一個長度100的向量
x=np.zeros(100)
y=np.zeros(100)
# 這裡我們設立我們的斜率為m截距為b
# d為資料點的離散程度
m=25
b=20
d=1
# 這裡我們設x=1~100
# y=mx+b加上一個-1~1的隨機值
for i in range(0,100):
    x[i]=i/100
    y[i]=m*x[i]+b+d*random.uniform(-1,1)
# 設定我們輸出圖片的大小
plt.rcParams["figure.figsize"] = (18,18)

# 初始畫我們的權重
w=np.zeros(2)
# 分別算出x、y、x^2、x*y的平均值
x_mean=np.mean(x)
y_mean=np.mean(y)
x2_mean=np.mean(x**2)
xy_mean=np.mean(x*y)
# 設我們的學習參數為1
a=1
# 我們讓他跑100次迴圈
T=100
for t in range(0,T):
    #更新我們權重w
    w[0]=w[0]-a*(w[0]+w[1]*x_mean-y_mean)
    w[1]=w[1]-a*(w[0]*x_mean+w[1]*x2_mean-xy_mean)
print(w[1])
print(w[0])
# 設定我們輸出圖片的大小
plt.rcParams["figure.figsize"] = (18,18)
#畫出圈圈的點，bo：b代表藍色、o代表圈圈、marksize為大小
plt.plot([0,1],[w[0],w[1]+w[0]],"r-",linewidth=6)
plt.plot(x,y,"bo",markersize=10)
# 設定x和y軸座標的字體大小
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
#畫出來
plt.show()