import numpy as np
import matplotlib.pyplot as plt

#以Momentum法尋找f的局部極小值，並輸出每步更新的點
def gradient_descent_momentum(f,x,alpha,gamma,iterations,epsilon):
    history = [x]
    v = np.zeros_like(x)
    for i in range(iterations): #迭代
        df = (f(x + epsilon) - f(x)) / epsilon #計算df/dx
        if abs(df)<epsilon:
            print("梯度夠小！")
            break
        v = gamma*v+alpha*df
        x = x-v
        history.append(x) #保存更新點
    return history

f = lambda x: x**3-3*x**2-9*x+2

path = gradient_descent_momentum(f,1.0,0.01,0.8,100,0.000001) #尋找局部極小值
print('(',path[-1],',',f(path[-1]),')') #顯示局部極小值點的座標

#繪製尋找局部極小值的過程
x = np.arange(-3, 4, 0.01)
y = f(x)
plt.plot(x,y)
path_x = np.asarray(path)
path_y=f(path_x)
plt.quiver(path_x[:-1],path_y[:-1],path_x[1:]-path_x[:-1],path_y[1:]-path_y[:-1],scale_units='xy',angles='xy',scale=1,color='k') #繪製箭頭
plt.scatter(path[-1],f(path[-1]))
plt.show()


