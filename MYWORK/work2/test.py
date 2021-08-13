import tensorflow as tf
import numpy as np


# 本地读取MNIST流程
path = './mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()

h = x_train.shape[1] // 2
w = x_train.shape[2] // 2

# 为便于评测，图像尺寸缩小为原来的一半
x_train = np.expand_dims(x_train, axis=-1)
x_train = tf.image.resize(x_train, [h, w]).numpy()  # if we want to resize
x_test = np.expand_dims(x_test, axis=-1)
x_test = tf.image.resize(x_test, [h, w]).numpy()  # if we want to resize

# 图像归一化,易于网络学习
x_train, x_test = x_train / 255.0, x_test / 255.0

# 注意，即使同一个数字也有很多不同图像，
# 需要产生的是尽可能多的数字图像样例对的组合，
# 下面会采用两个随机列输入配对的方式去产生
# 因此，为扩充更多的图像对加法实例，先扩充两个随机输入列的长度
len_train = len(x_train)
len_test = len(x_test)
# print(x_train.shape) (60000, 14, 14, 1)
len_ext_train = len_train * 3
len_ext_test = len_test * 3

# 由于本实训采用线性全连接网络，需要将图片拉伸为一维向量
x_train = x_train.reshape((len_train, -1))
x_test = x_test.reshape((len_test, -1))
# print(x_train.shape) (60000, 196)

# 由于MNIST是按数字顺序排列，故将其打乱，通过随机交叉样本产生更多随机的图片数字加法组合
left_train_choose = np.random.choice(len_train, len_ext_train, replace=True)
right_train_choose = np.random.choice(len_train, len_ext_train, replace=True)
left_test_choose = np.random.choice(len_test, len_ext_test, replace=True)
right_test_choose = np.random.choice(len_test, len_ext_test, replace=True)

x_train_l = x_train[left_train_choose]
x_train_r = x_train[right_train_choose]
x_test_l = x_test[left_test_choose]
x_test_r = x_test[right_test_choose]
# ！！！！！！注意，本题标签不采用one-hot编码
y_train = y_train[left_train_choose] + y_train[right_train_choose]
y_test = y_test[left_test_choose] + y_test[right_test_choose]

# WORK1: --------------BEGIN-------------------
# 请补充完整训练集和测试集的产生方法：

train_datasets = tf.data.Dataset.zip(
    (tf.data.Dataset.from_tensor_slices((x_train_l, x_train_r)), tf.data.Dataset.from_tensor_slices(y_train))).batch(64)
print(train_datasets)

test_datasets = tf.data.Dataset.zip(
    (tf.data.Dataset.from_tensor_slices((x_test_l, x_test_r)), tf.data.Dataset.from_tensor_slices(y_test))).batch(64)
print(test_datasets)
