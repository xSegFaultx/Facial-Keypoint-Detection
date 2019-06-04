## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # initial shape (1,224,224)
        # first group
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1,bias=False) 
        # shape (32,244,244)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0,bias=False) 
        # shape (32,244,244)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2) # shape (32,112,112)
        
        # second group
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,bias=False) 
        # shape (64,112,112)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0,bias=False) 
        # shape (64,112,112)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2) # shape (64,56,56)
        
        # third group
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,bias=False)
        # shape (128,56,56)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0,bias=False)
        # shape (128,56,56)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2,2) # shape (128,28,28)
        
        # fourth group
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1,bias=False)
        # shape (256,28,28)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0,bias=False)
        # shape (256,28,28)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2,2) # shape (256, 14, 14)
        
        # fifth group
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1,bias=False)
        # shape (512,14,14)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0,bias=False)
        # shape (512,14,14)
        self.pool5 = nn.MaxPool2d(2,2) # shape (512, 7, 7)
        
        #fully connected layers
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 1024)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 1024)
        self.drop3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(1024, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # first group
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(self.bn2(F.relu(self.conv2(x))))
        
        # second group
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool2(self.bn4(F.relu(self.conv4(x))))
        
        # thrid group
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.pool3(self.bn6(F.relu(self.conv6(x))))
        
        # fourth group
        x = self.bn7(F.relu(self.conv7(x)))
        x = self.pool4(self.bn8(F.relu(self.conv8(x))))
        
        # fifth group
        x = self.bn9(F.relu(self.conv9(x)))
        x = self.pool5(F.relu(self.conv10(x)))
        
        # fully connected layer
        x = x.view(x.size(0),-1)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
