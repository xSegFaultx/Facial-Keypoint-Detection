import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # input shape (1, 224, 224)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # shape (64, 112, 112)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # shape (128, 56, 56)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        # shape (256, 28, 28)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        # shape (512, 14, 14)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        # shape (512, 7, 7)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(4096, 136)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)), inplace=True)
        x = F.relu(self.bn6(self.conv6(x)), inplace=True)
        x = F.relu(self.bn7(self.conv7(x)), inplace=True)
        x = self.pool(x)

        x = F.relu(self.bn8(self.conv8(x)), inplace=True)
        x = F.relu(self.bn9(self.conv9(x)), inplace=True)
        x = F.relu(self.bn10(self.conv10(x)), inplace=True)
        x = self.pool(x)

        x = F.relu(self.bn11(self.conv11(x)), inplace=True)
        x = F.relu(self.bn12(self.conv12(x)), inplace=True)
        x = F.relu(self.bn13(self.conv13(x)), inplace=True)
        x = self.pool(x)

        # fully connected layer
        x = x.view(x.size(0), -1)
        x = self.drop1(F.relu(self.fc1(x), inplace=True))
        x = self.drop2(F.relu(self.fc2(x), inplace=True))
        x = self.fc3(x)
        return x
