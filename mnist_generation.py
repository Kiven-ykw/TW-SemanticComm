import os
import numpy as np
import matplotlib.pyplot as plt

root_path = 'data'

data = np.load(root_path + '/mnist_sequence1_sample_5distortions5x5.npz')

X_train = data['X_train']
y_train = data['y_train']

X_val = data['X_valid']
y_val = data['y_valid']

X_test = data['X_test']
y_test = data['y_test']

if not os.path.exists(root_path + 'train'):
    os.mkdir(root_path + '/train')

f = open(root_path + '/train/path.txt', 'w')

for i in range(len(X_train)):

    img_path = root_path + '/train/%05d.jpg'%i

    img = X_train[i].reshape(40,40)
    plt.imsave(img_path, img)
    label = y_train[i,0]

    f.write(img_path+ ' %d\n'%(label))

f.close()

if not os.path.exists(root_path + 'val'):
    os.mkdir(root_path + '/val')

f = open(root_path + '/val/path.txt', 'w')

for i in range(len(X_val)):

    img_path = root_path + '/val/%05d.jpg'%i

    img = X_val[i].reshape(40,40)
    plt.imsave(img_path, img)
    label = y_val[i,0]

    f.write(img_path+ ' %d\n'%(label))

f.close()


if not os.path.exists(root_path + 'test'):
    os.mkdir(root_path + '/test')

f = open(root_path + '/test/path.txt', 'w')

for i in range(len(X_test)):

    img_path = root_path + '/test/%05d.jpg'%i

    img = X_test[i].reshape(40,40)
    plt.imsave(img_path, img)
    label = y_test[i,0]

    f.write(img_path+ ' %d\n'%(label))

f.close()