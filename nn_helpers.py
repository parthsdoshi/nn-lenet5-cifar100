import torch
import torch.nn as nn

def y_mse(Y, Y_hat):
    # first subtracts element wise from labels
    # then squares element wise
    # then reduces over columnns so that the dims become N * 1
    se = torch.sum((Y - Y_hat) ** 2, dim=1, keepdim=True)

    # then we sum rows and divide by number of rows, N
    mse = (1. / Y_hat.shape[0]) * torch.sum(se)

    return mse

# t is of dims N * 1 where N is the batch size
# C should be the number of values for the column
def oneHotEncodeOneCol(t, C=2):
    N = t.shape[0]
    onehot = torch.Tensor([
        [0] * C
    ] * N)
    for i, v in enumerate(t):
        onehot[i, v] = 1
    
    return onehot

# t is of dims N * m where N is the batch size and m is the number of features
# C should be an array of how many different values there are for each of your m features
# if you do not want to one hot encode a specific feature, set that C value to 0
def oneHotEncode(t, C):
    # not implemented yet...
    pass

# flattens so we can go from conv layers to linear layers
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

def createLenet5(in_channels=3, init_padding=(0, 0), classes=10, activation=nn.ReLU):
    lenet5 = nn.Sequential(
        nn.Conv2d(in_channels, 6, kernel_size=(5, 5), padding=init_padding),
        activation(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(6, 16, kernel_size=(5, 5)),
        activation(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        Flatten(),
        nn.Linear(16*5*5, 120),
        activation(),
        nn.Linear(120, 84),
        activation(),
        nn.Linear(84, classes)
    )

    return lenet5