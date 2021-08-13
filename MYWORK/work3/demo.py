import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf


path = './dataset/mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()

# 统计y_test中不同标签个数
label_mask = np.unique(y_test)
label_count = {}
for v in label_mask:
    label_count[v] = np.sum(y_test == v)
print("label_mask值为：")
print(label_mask)
print("统计结果：")
print(label_count)

# 对每个类进行排序
yy = y_test
y_ind = yy.argsort()  # argsort()函数返回的是数组值从小到大的索引值

x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

n_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


def plot_digits(instances, images_per_row=10, **options):
    plt.figure()
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")
    plt.savefig("./digits.png")


example_images = []
# WORK1: --------------BEGIN-------------------
# 每个类按照y_ind排序后的新顺序，依次各取类中的
# 前10个样本作为plot_digits的图片拼图
# example_images最终的形状为按顺序排列的
# (100,28,28,1)多维数组
count = 0
for v in label_mask:
    for i in range(10):
        image = x_test[y_ind[i+count]]
        example_images.append(image)
    count = count + label_count[v]
example_images = np.array(example_images)
shape = example_images.shape[-1]
print(shape)
# WORK1: --------------END-------------------
# plot_digits(example_images, images_per_row=10)
