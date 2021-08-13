import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 准备数据
x_t = np.arange(-23/18, (2*np.pi-23)/18, 2*np.pi/18/2000, dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis=1)
y_train = np.cos(18*x_t+23)
# print(x_train.shape)  (2000,3)

# 模型定义
# Dense实现以下操作：output = activation（dot（input，kernel）+ bias）
inputs = tf.keras.Input(shape=(3,), name='data')  #输入层
outputs = tf.keras.layers.Dense(units=1, input_dim=3)(inputs)  #输出层
Lm1 = tf.keras.Model(inputs=inputs, outputs=outputs, name='Lm1') #创建模型包装输入层和输出层
Lm1.summary()
# 模型训练
Lm1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse', metrics=['mse'])
Lm1.fit(x=x_train, y=y_train, epochs=1000)


# 测试
loss, acc = Lm1.evaluate(x_train, y_train)
print('loss=', loss)
print('acc=', acc)


forecast = Lm1(x_train)
plt.figure()
plot1 = plt.plot(x_t, y_train, 'b', label='original')
plpt2 = plt.plot(x_t, forecast, 'r', label='polyfit')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
