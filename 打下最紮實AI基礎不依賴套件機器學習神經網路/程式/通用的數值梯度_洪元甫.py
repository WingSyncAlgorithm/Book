import numpy as np

# f()自己用lamda定義
# param裡面丟numpy
def numerical_gradient(f, params, eps=1e-6):
    numerical_grads = []  # 儲存數值梯度的列表

    for x in params:
        grad = np.zeros(x.shape)  # 初始化梯度為零陣列

        # 創建用於遍歷 x 的迭代器，設定返回多維索引和可讀寫操作的標誌
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

        # 簡單來說就是對每個x去算他的梯度

        # 遍歷每個元素
        while not it.finished:
            idx = it.multi_index  # 當前元素的多維索引
            old_value = x[idx]    # 儲存原始值

            # 對該元素進行微小變化，計算函數在新值下的變化
            x[idx] = old_value + eps
            fx_plus = f()  # 計算 f(x + eps)

            x[idx] = old_value - eps
            fx_minus = f()  # 計算 f(x - eps)

            # 根據數值變化計算該元素的數值偏導數
            grad[idx] = (fx_plus - fx_minus) / (2 * eps)

            x[idx] = old_value  # 恢復原始值
            it.iternext()  # 移動到下一個元素

        numerical_grads.append(grad)  # 將該參數的數值梯度加入列表

    return numerical_grads  # 返回數值梯度列表