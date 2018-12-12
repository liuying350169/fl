import torch
import numpy as np
from torchvision import datasets, transforms 

class Mnist():
    IID = False
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100





    def __init__(self):
        # use numpy
        path_np = './mnist.npz'
        f = np.load(path_np)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()



        self.train_data = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())


        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=Mnist.BATCH_SIZE, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=Mnist.BATCH_SIZE, shuffle=True)

    def get_train_data(self):
        return self.train_loader
    def get_test_data(self):
        return self.test_data