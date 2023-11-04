import numpy as np
import math

# Set the random seed for reproducibility
np.random.seed(1)

# RNN parameters initialization function
def rnn_params_init(input_dim, hidden_dim, output_dim, scale=0.01):
    Wx = np.random.randn(input_dim, hidden_dim) * scale
    Wh = np.random.randn(hidden_dim, hidden_dim) * scale
    bh = np.zeros((1, hidden_dim))
    Wf = np.random.randn(hidden_dim, output_dim) * scale
    bf = np.zeros((1, output_dim))
    return [Wx, Wh, bh, Wf, bf]

# Function to initialize the hidden state
def rnn_hidden_state_init(batch_dim, hidden_dim):
    return np.zeros((batch_dim, hidden_dim))

# RNN forward step function
def rnn_forward_step(params, X, prevH):
    Wx, Wh, bh, Wf, bf = params
    H = np.tanh(np.dot(X, Wx) + np.dot(prevH, Wh) + bh)
    F = np.dot(H, Wf) + bf
    return F, H

# RNN full forward function
def rnn_forward(params, Xs, H_):
    Wx, Wh, bh, Wf, bf = params
    H = H_
    Fs = []
    Hs = {}
    Hs[-1] = np.copy(H)

    for t in range(len(Xs)):
        X = Xs[t]
        H = Hs[t-1]
        F, H = rnn_forward_step(params, X, H)
        Fs.append(F)
        Hs[t] = H
    return Fs, Hs

# Gradient clipping function
def grad_clipping(grads, alpha):
    norm = math.sqrt(sum((grad ** 2).sum() for grad in grads))
    if norm > alpha:
        ratio = alpha / norm
        for i in range(len(grads)):
            grads[i] *= ratio

# RNN backward step function
def rnn_backward_step(params, dZ, X, H, H_prev, dH_next):
    Wx, Wh, bh, Wf, bf = params
    dWf = np.dot(H.T, dZ)
    dbf = np.sum(dZ, axis=0, keepdims=True)
    dH = np.dot(dZ, Wf.T) + dH_next
    dZh = (1 - H * H) * dH
    dbh = np.sum(dZh, axis=0, keepdims=True)
    dWx = np.dot(X.T, dZh)
    dWh = np.dot(H_prev.T, dZh)
    dH_next = np.dot(dZh, Wh.T)
    return dWx, dWh, dbh, dWf, dbf, dH_next

# RNN full backward function
def rnn_backward(params, Xs, Hs, dZs, clip_value=5.):
    Wx, Wh, bh, Wf, bf = params
    dWx, dWh, dbh, dWf, dbf = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(bh), np.zeros_like(Wf), np.zeros_like(bf)
    dH_next = np.zeros_like(Hs[0])
    T = len(Xs)
    for t in reversed(range(T)):
        dZ = dZs[t]
        H = Hs[t]
        H_prev = Hs[t-1] if t > 0 else np.zeros_like(Hs[0])
        X = Xs[t]
        grads_step = rnn_backward_step(params, dZ, X, H, H_prev, dH_next)
        
        for grad, grad_step in zip([dWx, dWh, dbh, dWf, dbf], grads_step):
            grad += grad_step
        
        dH_next = grads_step[-1]
        
    grads = [dWx, dWh, dbh, dWf, dbf]
    if clip_value is not None:
        grad_clipping(grads, clip_value)
    return grads

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Prevent overflow
    return e_x / e_x.sum(axis=1, keepdims=True)

# Cross entropy loss function
def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss

# Loss and gradient function for softmax cross-entropy
def loss_gradient_softmax_crossentropy(F, Y):
    probs = softmax(F)
    loss = cross_entropy_loss(probs, Y)
    dF = probs
    dF[range(len(Y)), Y] -= 1
    dF = dF / len(Y)
    return loss, dF

# Loss and gradient computation function for the RNN
def rnn_loss_grad(Fs, Ys, loss_fn=loss_gradient_softmax_crossentropy, flatten=True):
    loss = 0
    dFs = {}

    for t in range(len(Fs)):
        F = Fs[t]
        Y = Ys[t]
        if flatten and Y.ndim == 2:
            Y = Y.flatten()
        loss_t, dF_t = loss_fn(F, Y)
        loss += loss_t
        dFs[t] = dF_t

    return loss, dFs

# Test the RNN functions
input_dim = 3  # Example input dimension
hidden_dim = 5  # Example hidden layer dimension
output_dim = 2  # Example output dimension
batch_dim = 4  # Example batch size

# Initialize parameters
params = rnn_params_init(input_dim, hidden_dim, output_dim)

# Example input data (batch_dim x input_dim)
Xs = np.random.randn(batch_dim, input_dim)

# Example labels
Ys = np.array([0, 1, 0, 1])

# Initialize hidden state
H_ = rnn_hidden_state_init(batch_dim, hidden_dim)

# Forward pass
Fs, Hs = rnn_forward(params, [Xs], H_)

# Compute the loss and the gradient of the loss with respect to the output of the network
loss, dFs = rnn_loss_grad(Fs, [Ys], loss_fn=loss_gradient_softmax_crossentropy)

print("Loss:", loss)
