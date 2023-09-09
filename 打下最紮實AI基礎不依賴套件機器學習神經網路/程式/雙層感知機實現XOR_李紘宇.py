import numpy as np

# 定义感知机
def perceptron(x, w, b):
    weighted_sum = np.dot(x, w) + b
    return 1 if weighted_sum >= 0 else 0

# 定义 XOR 函数
def xor(x1, x2):
    # 第1層第1個神經元(NAND)
    w1 = np.array([-0.5, -0.5])  # 權重
    b1 = 0.6 # 偏差
    s1 = perceptron(np.array([x1, x2]), w1, b1)

    # 第1層第2個神經元(OR)
    w2 = np.array([1, 1])  # 權重
    b2 = -0.5  # 偏差
    s2 = perceptron(np.array([x1, x2]), w2, b2)
    
    # 第2層第1個神經元(AND)
    w3 = np.array([0.5, 0.5])  # 權重
    b3 = -0.6  # 偏差
    s3 = perceptron(np.array([s1, s2]), w3, b3)

    return s3

# 測試 XOR
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

for x1, x2 in inputs:
    output = xor(x1, x2)
    print(f'Input: ({x1}, {x2}), Output: {output}')
