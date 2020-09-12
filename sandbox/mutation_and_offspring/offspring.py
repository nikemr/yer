import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 240x240x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 120x120x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 60x60x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 30 * 30 -> 500)
        self.fc1 = nn.Linear(64 * 30 * 30, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 4)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 30 * 30)        
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))        
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

model_A= Net()
model_B= Net()
print(model_A.conv1.weight)