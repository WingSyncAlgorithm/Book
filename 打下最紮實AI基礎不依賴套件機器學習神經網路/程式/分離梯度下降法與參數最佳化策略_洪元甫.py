import numpy as np

# 定義基礎優化器類別
class Optimizer:
    def __init__(self, params):
        # 初始化優化器類別。
        # Args:
        #     params (list): 要優化的參數列表。
        self.params = params

    def step(self, grads):
        # 優化步驟。根據梯度更新參數。
        # Args:
        #     grads (list): 梯度列表，與參數列表對應。
        pass  # 子類別需要實現具體的優化步驟

    def parameters(self):
        # 返回優化器中優化的參數列表。
        # Returns:
        #     list: 優化的參數列表。
        return self.params

def gradient_descent(df, optimizer, iterations, epsilon=1e-8):
    # Gradient Descent 優化過程的封裝函數。
    # Args:
    #     df: 要優化的目標函數的梯度。
    #     optimizer: 使用的優化器實例。
    #     iterations (int): 迭代次數。
    #     epsilon (float): 梯度大小閾值，用於判斷是否停止迭代。
    # Returns:
    #     list: 優化過程中參數值的歷史記錄。

    x, = optimizer.parameters()  # 取得優化器中的初始參數值
    x = x.copy()  # 複製一份，避免修改原始參數
    #一個x是外面的參數(lambda)，一個是裡面需要用history存起來的
    history = [x]  # 初始化參數歷史記錄

    for i in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("梯度夠小！")
            break

        grad = df(x)  # 計算當前位置的梯度
        x, = optimizer.step([grad])  # 使用優化器更新參數
        x = x.copy()  # 複製一份更新後的參數

        history.append(x)  # 將更新後的參數值加入歷史記錄

    return history

# 定義隨機梯度下降（SGD）優化方法
class SGD(Optimizer):
    def __init__(self, params, learning_rate):
        # 初始化 SGD 優化方法。
        # Args:
        #     params (list): 要優化的參數列表。
        #     learning_rate (float): 學習率。
        super().__init__(params)
        self.lr = learning_rate

    def step(self, grads):
        # 優化步驟。根據梯度和學習率更新參數。
        # Args:
        #     grads (list): 梯度列表，與參數列表對應。
        for i in range(len(self.params)):
            self.params[i] -= self.lr * grads[i]
        return self.params


# 定義目標函數的梯度 df
df = lambda x: np.array(((1/8) * x[0], 18 * x[1]))

# 初始參數
x0 = np.array((-2.4, 0.2))

# 創建 SGD 優化器實例
optimizer = SGD([x0], learning_rate=0.1)

# 使用 gradient_descent_ 函數進行優化，迭代次數為 100
# 自己實測需要大概迭代1500次左右才能到達預設的1e-8
path = gradient_descent(df, optimizer, iterations=100)

# 輸出最終的梯度
print("最終梯度:", path[-1])

# 轉numpy在轉置
path = np.asarray(path)
path = path.transpose()