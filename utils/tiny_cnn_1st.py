import numpy as np
import torch
import torch.nn as nn

class TinyCNN(nn.Module):
    """
    TinyCNN: https://arxiv.org/abs/1911.06777v1

    fc1, fc2 should be modified for each image size and class num.

    model=TinyCNN()
    model.fc1 = nn.Linear(in_features=128 * FC_H * FC_W, out_features=N_CLASSES)
    model.fc2 = nn.Linear(in_features=100, out_features=N_CLASSES)
    """
    def __init__(self, DROPOUT=False):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,   32,  3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(32,  64,  3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,  128, 3, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 10)

        self.dropout_control = DROPOUT
        self.fc1_drop = nn.Dropout2d(p=0.5)
        self.fc2_drop = nn.Dropout2d(p=0.4)

    def encode(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        return x

    def classify(self, x):
        x = x.view(x.size(0),-1)

        if(self.dropout_control):
            x = self.relu(self.fc1_drop(self.fc1(x)))
            x = self.relu(self.fc2_drop(self.fc2(x)))
        else:
            x = self.relu(self.fc1(x))
            # For easy replacement of label sizes
            # activation is assigned to forward output
            x = self.fc2(x)

        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.classify(x)

        return self.relu(x)

"""
# for FC layer's input size computation
img_shapes = [224, 224]

FC_H = img_shapes[0]//2**4 # 4 is max plling layer num
FC_W = img_shapes[1]//2**4


# TinyCNN model instance set
model=TinyCNN()

# FC layer input&output size change
model.fc1 = nn.Linear(in_features=128 * FC_H * FC_W, out_features=100)
model.fc2 = nn.Linear(in_features=100, out_features=N_CLASSES)
"""