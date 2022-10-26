import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,nb_classes:int):
        super().__init__()
        size_of_pool = (2, 2)
        
        self.conv1 =  nn.Conv2d(3, 32, 5)
        self.conv2 =  nn.Conv2d(32, 32, 5)
        self.max_pool =  nn.MaxPool2d(size_of_pool)
        self.conv3 =  nn.Conv2d(32, 64, (3,3))
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()
        self.conv4 = nn.Conv2d(64, 64, (3,3))
        self.conv5 = nn.Conv2d(10, 8, (3,3))
        self.conv5_bn=nn.BatchNorm2d(8)
        self.dropout = nn.Dropout(p=0.25)
        
        nb=576
        self.lin1 = nn.Linear(nb, 576)
        self.lin2 = nn.Linear(576, nb_classes)
        self.lin3 = nn.Linear(576, 576)
        self.lin4 = nn.Linear(576, nb_classes)
        self.softmax = nn.Softmax(dim=1)
        self.Lrelu = nn.LeakyReLU()
        
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.lin1(x)
        x = self.tan(x)
        x = self.dropout(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x