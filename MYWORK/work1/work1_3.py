import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D
import time
import os
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 数据准备，构建训练集
x_t = np.arange(-23/18, (2*np.pi-23)/18, 2*np.pi/18/2000, dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis=1)
y_train = np.cos(18*x_t+23)


inputs = tf.keras.Input(shape=(3,), name='data')
outputs = tf.keras.layers.Dense(units=1, input_dim=3)(inputs)
Lm1 = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear')


# 定义存储模型的回调函数，请补充完整
checkpoint_path = "./checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=100)


# 编译模型，定义相关属性
Lm1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse', metrics=['mse'])
# 在训练过程中使用回调函数，请补充
Lm1.fit(x_train, y_train, epochs=1000, callbacks=[cp_callback])


loss, acc = Lm1.evaluate(x_train, y_train)
print("saved model, loss:{:5.2f}".format(loss))


# 取出最后一次保存的断点
latest = tf.train.latest_checkpoint(checkpoint_dir)
# 构建模型加载参数，请补充完整
Lm2 = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear')
Lm2.compile(loss='mse')
Lm2.load_weights(latest)


loss = Lm2.evaluate(x_train, y_train)
print("saved model, loss:{:5.2f}".format(loss))
predictions = Lm2(x_train)
plt.figure()
plot1 = plt.plot(x_t, y_train, 'b', label='original values')
plpt2 = plt.plot(x_t, predictions, 'r', label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.show()