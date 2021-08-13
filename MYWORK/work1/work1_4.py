import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D
import time
import os
from pathlib import Path


# 数据准备，构建训练集
x_t = np.arange(-23/18, (2*np.pi-23)/18, 2*np.pi/18/2000, dtype=np.float32)
x_t = x_t[:, np.newaxis]
x_train = np.concatenate((x_t, np.power(x_t, 2), np.power(x_t, 3)), axis=1)
y_train = np.cos(18*x_t+23)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(64)  # batch(64)每个批次有64个数据


inputs = tf.keras.Input(shape=(3,), name='data')
outputs = tf.keras.layers.Dense(units=1, input_dim=3)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='linear')


loss_object = tf.keras.losses.MeanSquaredError()  # 创建损失函数
optimizer = tf.keras.optimizers.Adam(0.1)  # 创建优化器
train_loss = tf.keras.metrics.Mean(name='train_loss')  # 创建训练损失函数的计算方法
test_loss = tf.keras.metrics.Mean(name='test_loss')  # 创建测试损失函数的计算方法


# 定义python函数封装训练迭代过程，补充
@tf.function
def train_step(data, labels):
    with tf.GradientTape() as tape:
        y_pred = model(data)
        loss = loss_object(labels, y_pred)
    grads = tape.gradient(loss, model.variables)
    train_loss(loss)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    return


# 测试迭代
@tf.function
def test_step(data, labels):
    y_pred = model(data)
    loss = loss_object(labels, y_pred)
    test_loss.update_state(loss)
    return


EPOCHS = 1200
# 自定义循环对模型进行训练
for epoch in range(EPOCHS):
    start = time.time()
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    test_loss.reset_states()
    for data, labels in train_dataset:
        train_step(data, labels)
    for test_data, test_labels in train_dataset:
        test_step(test_data, test_labels)
    end = time.time()
    # 输出训练情况
    template = 'Epoch {}, loss:{:.3f}, test loss:{:.3f}, time used: {:.2f}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          test_loss.result(),
                          end-start))


# 将模型保存为model.h5文件
print(model.variables)
model.save("./save_model/model.h5")

# 从h5文件加载模型并赋值给model_load
model_load = tf.keras.models.load_model('./save_model/model.h5')
print(model_load.variables)
model_load.compile(loss='mse')


loss = model_load.evaluate(x_train, y_train)
predictions = model_load(x_train)
plt.figure()
plot1 = plt.plot(x_t, y_train, 'b', label='original values')
plpt2 = plt.plot(x_t, predictions, 'r', label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.show()
