import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O

from nn_helpers import oneHotEncodeOneCol, y_mse, createLenet5


class Img2Num():
    def __init__(self, batch_size=1, test_batch_size=1000, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.device = device

        dataset = tv.datasets.MNIST
        train_dataset = dataset('./data', train=True, download=True,
                                transform=tv.transforms.Compose([
                                    tv.transforms.ToTensor()
                                ]))
        test_dataset = dataset('./data', train=False, download=True,
                               transform=tv.transforms.Compose([
                                   tv.transforms.ToTensor()
                               ]))

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_test_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.test_batch_size, shuffle=True)

        # split train data in validation set too

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.test_batch_size, shuffle=True)

        self.train_size = len(self.train_loader)
        self.test_size = len(self.test_loader)

        self.net = createLenet5(
            in_channels=1, init_padding=(2, 2), classes=10).to(device)

    def forward(self, x):
        return self.net(x)

    def train(self, epochs=2, validate_every=2000, save=False):
        lenet5_mnist_dev = self.net
        opt = O.SGD(lenet5_mnist_dev.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss(reduction="mean")
        train_cross_entropy = []
        train_accuracy = []
        validation_cross_entropy = []
        validation_accuracy = []

        best_model_accuracy = 0

        for epoch in range(epochs):
            n_correct = 0
            n_total = 0
            for i, batch in enumerate(self.train_loader):
                x, labels = batch
                x, labels = x.to(self.device), labels.to(self.device)
                N = x.shape[0]

                # training mode (for things like dropout)
                lenet5_mnist_dev.train()

                # clear previous gradients
                opt.zero_grad()

                y_hat = lenet5_mnist_dev(x)
                loss = criterion(y_hat, labels)
                loss.backward()
                opt.step()

                train_cross_entropy.append(loss)

                n_correct += (torch.argmax(y_hat, dim=1)
                              == labels).sum().item()
                n_total += N

                # evaluation mode (e.g. adds dropped neurons back in)
                lenet5_mnist_dev.eval()
                if i % validate_every == 0:
                    n_val_correct = 0
                    n_val_total = 0
                    v_cross_entropy_sum = 0

                    # don't calculate gradients here
                    with torch.no_grad():
                        for j, v_batch in enumerate(self.test_loader):
                            v_x, v_labels = v_batch
                            v_x, v_labels = v_x.to(
                                self.device), v_labels.to(self.device)
                            v_N = v_x.shape[0]

                            v_y_hat = lenet5_mnist_dev(v_x)
                            v_loss = criterion(v_y_hat, v_labels)
                            v_cross_entropy_sum += v_loss
                            n_val_correct += (torch.argmax(v_y_hat,
                                                           dim=1) == v_labels).sum().item()
                            n_val_total += v_N

                    print(
                        f"[epoch {epoch + 1}, iteration {i}] \t accuracy: {n_val_correct / n_val_total} \t cross entropy: {v_cross_entropy_sum / n_val_total}")
                    validation_accuracy.append(n_val_correct / n_val_total)
                    validation_cross_entropy.append(
                        v_cross_entropy_sum / n_val_total)
                    if n_val_correct / n_val_total >= best_model_accuracy:
                        best_model_accuracy = n_val_correct / n_val_total
                        if save:
                            print("saving")
                            torch.save(lenet5_mnist_dev.state_dict(),
                                    './trained_models/lenet5_mnist')

            print(
                f"epoch {epoch + 1} accumulated train accuracy: {n_correct / n_total}")
            train_accuracy.append(n_correct / n_total)

        return (train_cross_entropy, train_accuracy, validation_cross_entropy, validation_accuracy)
