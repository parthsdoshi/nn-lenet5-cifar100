import pickle

import numpy as np

import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O

import cv2

from nn_helpers import oneHotEncodeOneCol, y_mse, createLenet5


class Img2Obj():
    def __init__(self, batch_size=64, test_batch_size=1000, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.device = device

        dataset = tv.datasets.CIFAR100
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

        self.net = createLenet5(in_channels=3, classes=100).to(device)
        self.labels = pickle.load(open('./data/cifar-100-python/meta', 'rb'))['fine_label_names']

    def forward(self, x):
        y = None
        with torch.no_grad():
            y = self.net(x)
        
        return torch.argmax(self.net(x), dim=1)

    # img should be a tensor of shape (1, 3, 32, 32)
    def view(self, img, denormalize=False):
        y = self.forward(img)
        label = self.labels[y]
        img_n = img.numpy()[0].transpose(1, 2, 0)
        if denormalize:
            img_n = (img_n * 255)
        img_n = img_n.astype(np.uint8)
        img_n = cv2.cvtColor(img_n, cv2.COLOR_RGB2BGR)
        cv2.namedWindow(label, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(label, 600, 600)
        cv2.imshow(label, img_n)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def cam(self):
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            img = cv2.resize(frame, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)
            img = torch.tensor(img).float()
            img = img / 255
            label = self.labels[self.forward(img.view(1, *img.shape))]
            cv2.putText(frame, label, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
            cv2.imshow("webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def train(self, epochs=50, validate_every=2000, save=False):
        lenet5_cifar100_dev = self.net
        opt = O.Adam(lenet5_cifar100_dev.parameters(), lr=0.001)
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
                lenet5_cifar100_dev.train()

                # clear previous gradients
                opt.zero_grad()

                y_hat = lenet5_cifar100_dev(x)
                loss = criterion(y_hat, labels)
                loss.backward()
                opt.step()

                train_cross_entropy.append(loss)

                n_correct += (torch.argmax(y_hat, dim=1)
                              == labels).sum().item()
                n_total += N

                # evaluation mode (e.g. adds dropped neurons back in)
                lenet5_cifar100_dev.eval()
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

                            v_y_hat = lenet5_cifar100_dev(v_x)
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
                            torch.save(lenet5_cifar100_dev.state_dict(),
                                    './trained_models/lenet5_cifar100')

            print(
                f"epoch {epoch + 1} accumulated train accuracy: {n_correct / n_total}")
            train_accuracy.append(n_correct / n_total)

        return (train_cross_entropy, train_accuracy, validation_cross_entropy, validation_accuracy)
