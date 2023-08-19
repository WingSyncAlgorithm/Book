import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 生成更多的隨機數據
np.random.seed(0)
num_samples = 100  # 增加樣本數
population = np.random.randint(1000, 10000, num_samples)  # 調整隨機數據範圍
profit = 200 + 0.02 * population + np.random.normal(0, 300, num_samples)

data = {
    'Population': population,
    'Profit': profit
}

df = pd.DataFrame(data)

# 提取特徵和標籤
X = df[['Population']]
y = df['Profit']

# 線性回歸模型
model = LinearRegression()
model.fit(X, y)

# 預測
predictions = model.predict(X)

# 原始數據散點圖和線性回歸線
plt.scatter(X, y, label='Original Data')
plt.plot(X, predictions, color='red', label='Linear Regression')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.legend()
plt.title('Linear Regression: Population vs Profit')
plt.show()

# 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 標準化數據散點圖
plt.scatter(X_scaled, y, label='Standardized Data')
plt.xlabel('Standardized Population')
plt.ylabel('Profit')
plt.legend()
plt.title('Standardized Data: Population vs Profit')
plt.show()

# 再次進行線性回歸
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)

# 預測
predictions_scaled = model_scaled.predict(X_scaled)

# 標準化數據線性回歸線
plt.scatter(X_scaled, y, label='Standardized Data')
plt.plot(X_scaled, predictions_scaled, color='red', label='Linear Regression')
plt.xlabel('Standardized Population')
plt.ylabel('Profit')
plt.legend()
plt.title('Linear Regression with Standardized Data: Population vs Profit')
plt.show()

# 計算標準化數據的MSE
mse_scaled = mean_squared_error(y, predictions_scaled)
print(f'Standardized Data MSE: {mse_scaled:.2f}')
