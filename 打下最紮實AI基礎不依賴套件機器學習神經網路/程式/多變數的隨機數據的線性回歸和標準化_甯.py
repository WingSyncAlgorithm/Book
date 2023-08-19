import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 引入 3D 繪圖工具
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 生成更多的隨機數據
np.random.seed(0)
num_samples = 100  # 增加樣本數
population = np.random.randint(1000, 10000, num_samples)  # 調整隨機數據範圍
expenses = np.random.randint(200, 2000, num_samples)  # 新增特徵：開銷
advertising = np.random.randint(10, 100, num_samples)  # 新增特徵：廣告投入

profit = 200 + 0.02 * population + 0.5 * expenses + 5 * advertising + np.random.normal(0, 300, num_samples)

data = {
    'Population': population,
    'Expenses': expenses,
    'Advertising': advertising,
    'Profit': profit
}

df = pd.DataFrame(data)

# 提取特徵和標籤
X = df[['Population', 'Expenses', 'Advertising']]
y = df['Profit']

# 線性回歸模型
model = LinearRegression()
model.fit(X, y)

# 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 創建兩個子圖
fig = plt.figure(figsize=(12, 6))

# 子圖1：顯示未標準化數據
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X['Population'], X['Expenses'], y, c='b', marker='o')
ax1.set_xlabel('Population')
ax1.set_ylabel('Expenses')
ax1.set_zlabel('Profit')
ax1.set_title('Original Data')
ax1.view_init(elev=20, azim=30)  # 設定視角

# 子圖2：顯示標準化數據
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], y, c='r', marker='o')
ax2.set_xlabel('Standardized Population')
ax2.set_ylabel('Standardized Expenses')
ax2.set_zlabel('Profit')
ax2.set_title('Standardized Data')
ax2.view_init(elev=20, azim=30)  # 設定視角

plt.tight_layout()
plt.show()

# 三維線性回歸模型
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)

# 繪製三維線性回歸平面
x_surf, y_surf = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100),
                             np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100))
z_surf = model_scaled.predict(np.array([x_surf.ravel(), y_surf.ravel(), np.zeros_like(x_surf.ravel())]).T)
z_surf = z_surf.reshape(x_surf.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], y, c='b', marker='o')
ax.plot_surface(x_surf, y_surf, z_surf, color='r', alpha=0.5)
ax.set_xlabel('Standardized Population')
ax.set_ylabel('Standardized Expenses')
ax.set_zlabel('Profit')
plt.title('3D Linear Regression')
plt.show()

# 計算標準化數據的MSE
predictions_scaled = model_scaled.predict(X_scaled)
mse_scaled = mean_squared_error(y, predictions_scaled)
print(f'Standardized Data MSE: {mse_scaled:.2f}')
