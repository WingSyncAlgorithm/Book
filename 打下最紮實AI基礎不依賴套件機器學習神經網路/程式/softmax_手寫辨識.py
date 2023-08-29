import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 載入 MNIST 資料集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 將圖像數據攤平為一維向量並歸一化
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

# 對標籤進行 one-hot 編碼
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# 定義模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='softmax', input_shape=(784,))
])

# 編譯模型
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(x_train, y_train, batch_size=100, epochs=1000, validation_data=(x_test, y_test), verbose=1)

# 繪製學習曲線
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs. Epoch')
plt.show()
