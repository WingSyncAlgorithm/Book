class Layer:
    def __init__(self):
        pass
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_dim, out_dim, activation=None):
        super().__init__()
        self.W = np.random.randn(input_dim, out_dim) * 0.01
        self.b = np.zeros((1,out_dim))
        self.activation = activation
        self.A = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        z = np.matmul(x, self.W) + self.b
        self.A = self.g(z)
        return self.A

    def backward(self, dA_out):
        A_in = self.x
        dZ = self.dZ_(dA_out)

        self.dW = np.dot(A_in.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dA_in = np.dot(dZ, np.transpose(self.W))
        return dA_in

    def g(self, z):
        if self.activation == 'relu':
            return np.maximum(0,z)
        elif self.activation == 'sigmoid':
            return 1/(1+np.exp(-z))
        else:
            return z

    def dZ_(self, dA_out):
        if self.activation == 'relu':
            grad_g_z = 1. * (self.A>0)
            return np.multiply(dA_out, grad_g_z)
        elif self.activation == 'sigmoid':
            grad_g_z = self.A(1-self.A)
            return np.multiply(dA_out, grad_g_z)
        else:
            return dA_out

class NeuralNetwork:
    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def forward(self, X):
        self.X = X
        for layer in self._layers:
            X = layer.forward(X)
        return X

    def predict(self, X):
        p = self.forward(X)

        if p.ndim == 1:
            return np.argmax(p)
        else:
            return np.argmax(p, axis=1)
    def backward(self, loss_grad, reg=0.):
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            loss_grad = layer.backward(loss_grad)
        for i in range(len(self._layers)):
            self._layers[i].dW += 2*reg*self._layers[i].W

    def reg_loss(self, reg):
        loss = 0
        for i in range(len(self._layers)):
            loss += reg*np.sum(self._layers[i].W*self._layers[i].W)
        return loss

    def update_parameters(self, learning_rate):
        for i in range(len(self._layers)):
            self._layers[i].W += -learning_rate * self._layers[i].dW
            #print(self._layers[i].W)
            self._layers[i].b += -learning_rate * self._layers[i].db
    def parameters(self):
        params = []
        for i in range(len(self._layers)):
            params.append(self._layers[i].W)
            params.append(self._layers[i].b)
        return params

    def grads(self):
        grads = []
        for i in range(len(self._layers)):
            grads.append(self._layers[i].dW)
            grads.append(self._layers[i].db)
        return grads

def cross_entropy_loss(F, Y, onehot = False):
    m = len(F)
    if onehot:
        return -(1./m) *np.sum(np.multiply(Y, np.log(F)))
    else:
        return -(1./m) * np.sum(np.log(F[range(m),Y]))

def softmax(Z):
    exp_Z = np.exp(Z-np.max(Z, axis=1, keepdims=True))
    return exp_Z/np.sum(exp_Z, axis=1, keepdims=True)

def softmax_cross_entropy(Z, y, onehot=False):
    m = len(Z)
    F = softmax(Z)
    if onehot:
        loss = -np.sum(y*np.log(F))/m
    else:
        y.flatten()

        log_Fy = -np.log(F[range(m),y])
        loss = np.sum(log_Fy) / m
    return loss

def cross_entropy_grad(Z, Y, onehot=False, softmax_out=False):
    if softmax_out:
        F = Z
    else:
        F = softmax(Z)
    if onehot:
        dZ = (F-Y) / len(Z)
    else:
        m = len(Y)
        dZ = F.copy()
        dZ[np.arange(m),Y] -= 1
        dZ /= m
    return dZ

def cross_entropy_grad_loss(F, y, softmax_out=False, onehot=False):
    if softmax_out:
        loss = cross_entropy_loss(F, y, onehot)
    else:
        loss = softmax_cross_entropy(F, y, onehot)
    loss_grad = cross_entropy_grad(F, y, onehot, softmax_out)
    return loss, loss_grad
def data_iter(X, y, batch_size, shuffle=False):
    m = len(X)
    indices = list(range(m))
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, m - batch_size + 1, batch_size):
        batch_indices = np.array(indices[i:min(i+batch_size, m)])
        yield X.take(batch_indices, axis=0), y.take(batch_indices, axis=0)

def train(nn, X, y, loss_function, epochs=10000, learning_rate=1e-0, reg=1e-3, print_n=10):
    for epoch in range(epochs):
        f = nn.forward(X)
        loss, loss_grad = loss_function(f, y)
        loss += nn.reg_loss(reg)

        nn.backward(loss_grad, reg)

        nn.update_parameters(learning_rate)

        if epoch % print_n == 0:
            print("iteration %d: loss %f" % (epoch, loss))

def train_batch(nn, XX, YY, loss_function, epochs=10000, batch_size=50, learning_rate=1e-0, reg=1e-3, print_n=10):
    iter =0
    for epoch in range(epochs):
        for X,y in data_iter(XX, YY, batch_size, True):
            f = nn.forward(X)
            loss, loss_grad = loss_function(f, y)
            loss += nn.reg_loss(reg)

            nn.backward(loss_grad, reg)

            nn.update_parameters(learning_rate)

            if iter % print_n == 0:
                print("iteration %d: loss %f" % (iter, loss))
            iter += 1


def gen_spiral_dataset(N=100, D=2, K=3):
    N = 100
    D = 2
    K = 3
    X = np.zeros((N*K, D))
    y = np.zeros(N*K, dtype='uint8')
    for j in range(K):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y

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

import numpy as np
np.random.seed(89)
X,y = gen_spiral_dataset()

nn = NeuralNetwork()
nn.add_layer(Dense(2, 100, 'relu'))
nn.add_layer(Dense(100, 3))

X_temp = np.random.randn(2,2)
y_temp = np.random.randint(3, size=2)
F = nn.forward(X_temp)
loss = softmax_cross_entropy(F, y_temp)
loss_grad = cross_entropy_grad(F,y_temp)
print(loss,np.mean(loss_grad))



nn.backward(loss_grad)
grads = nn.grads()

def loss_fun():
    F = nn.forward(X_temp)
    return softmax_cross_entropy(F,y_temp)

params = nn.parameters()
numerical_grads = numerical_gradient(loss_fun, params, 1e-6)
for i in range(len(params)):
    print(numerical_grads[i].shape, grads[i].shape)

def diff_error(x, y):
    return np.max(np.abs(x-y))

def diff_errors(xs, ys):
    errors = []
    for i in range(len(xs)):
        errors.append(diff_error(xs[i],ys[i]))
    return np.max(errors)

print(diff_errors(numerical_grads, grads))



