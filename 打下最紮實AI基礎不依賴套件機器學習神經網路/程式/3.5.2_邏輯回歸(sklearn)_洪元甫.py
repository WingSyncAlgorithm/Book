# -*- coding: utf-8 -*-
"""邏輯回歸(sklearn)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sFimBeVwWXRzPCFPcqwApmw6dpLrtTPR
"""

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 使用 scikit-learn 庫的 LogisticRegression 類創建模型
scikit_log_reg = LogisticRegression()

# 將數據分成訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 使用訓練數據擬合模型
scikit_log_reg.fit(X_train, y_train)

# 計算 scikit-learn 模型的準確性
scikit_score = scikit_log_reg.score(X_test, y_test)
print('Scikit score:', scikit_score)

# 使用模型預測
Y_predictions = scikit_log_reg.predict(X)

# 計算並輸出準確性
accuracy = float((np.dot(Y, Y_predictions) + np.dot(1 - Y, 1 - Y_predictions)) / float(Y.size) * 100)
print("預測的準確性是：", accuracy, "%")

# 計算決策曲線的兩個點以繪製決策曲線
x1 = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])
x2 = -scikit_log_reg.intercept_ / scikit_log_reg.coef_[0][1] - (scikit_log_reg.coef_[0][0] / scikit_log_reg.coef_[0][1]) * x1

# 創建一個圖表
fig, ax = plt.subplots(figsize=(4, 4))

# 繪製散點圖和決策曲線
ax.scatter(X[:n_pts, 0], X[:n_pts, 1], color='lightcoral', label='$Y = 0$')
ax.scatter(X[n_pts:, 0], X[n_pts:, 1], color='blue', label='$Y = 1$')
ax.plot(x1, x2, color='k', linestyle='--', linewidth=2)
ax.set_title('Sample Dataset')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend(loc=4)

# 顯示圖表
plt.show()

