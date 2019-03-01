import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision as tv

from img2obj import Img2Obj
from img2num import Img2Num

import cv2

i2n = Img2Num()
i2o = Img2Obj()

########################### PLOT OF LeNet ON MNIST ################################
# l5_train_cross_entropy, l5_train_acc, l5_validation_cross_entropy, l5_validation_acc = i2n.train()

# l5_y = np.array(l5_mnist_train_cross_entropy)[0::2000]
# l5_y.shape
# l5_y_test = np.array(l5_mnist_validation_cross_entropy)
# l5_y_test.shape
# x = np.array(list(range(0, 120000, 2000)))
# x = x / 60000
# x
# plt.plot(x, l5_y_test, label='validation cross entropy')
# plt.xlabel('epochs')
# plt.ylabel('cross entropy')
# plt.title('lenet5 mnist validation loss')
# plt.legend()
# plt.show()

# plt.plot(x, l5_y, label='train cross entropy')
# plt.xlabel('epochs')
# plt.ylabel('cross entropy')
# plt.title('lenet5 mnist train loss')
# plt.legend()
# plt.show()
###############################################################################

########################### PLOT OF LeNet ON CIFAR100 ################################
# l5_train_cross_entropy, l5_train_acc, l5_validation_cross_entropy, l5_validation_acc = i2o.train()

# l5_y = np.array(l5_mnist_train_cross_entropy)[0::2000]
# l5_y.shape
# l5_y_test = np.array(l5_mnist_validation_cross_entropy)
# l5_y_test.shape
# x = np.array(list(range(0, 120000, 2000)))
# x = x / 60000
# x
# plt.plot(x, l5_y_test, label='validation cross entropy')
# plt.xlabel('epochs')
# plt.ylabel('cross entropy')
# plt.title('lenet5 mnist validation loss')
# plt.legend()
# plt.show()

# plt.plot(x, l5_y, label='train cross entropy')
# plt.xlabel('epochs')
# plt.ylabel('cross entropy')
# plt.title('lenet5 mnist train loss')
# plt.legend()
# plt.show()
###############################################################################

########################### LOADING BEST MODELS ################################
i2o.net.load_state_dict(torch.load('./trained_models/lenet5_cifar100', map_location='cpu'))
i2n.net.load_state_dict(torch.load('./trained_models/lenet5_mnist', map_location='cpu'))
###############################################################################

########################### TESTING VIEW ################################
dataset = tv.datasets.CIFAR100
train_dataset = dataset('./data', train=True, download=True,
                        transform=tv.transforms.Compose([
                            tv.transforms.ToTensor()
                        ]))

rand = train_dataset[0]
i2o.view(rand[0].view(1,*rand[0].shape), denormalize=True)
##########################################################################

########################### TESTING CAM ################################
i2o.cam()
##########################################################################