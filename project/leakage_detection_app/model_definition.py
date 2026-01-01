# leakage_detection_app/model_definition.py

import torch.nn as nn
import torch.nn.functional as F

class LeakCNN(nn.Module):
    def __init__(self):
        super(LeakCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.dp1   = nn.Dropout2d(0.2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.dp2   = nn.Dropout2d(0.3)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.dp3   = nn.Dropout2d(0.4)

        self.conv4 = nn.Conv2d(128, 192, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(192)
        self.dp4   = nn.Dropout2d(0.5)

        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(192 * 8 * 8, 384)
        self.dpfc  = nn.Dropout(0.6)
        self.fc2   = nn.Linear(384, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))); x = self.dp1(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x)))); x = self.dp2(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x)))); x = self.dp3(x)
        x = self.pool(F.relu(self.bn4(self.conv4(x)))); x = self.dp4(x)
        x = x.view(-1, 192*8*8)
        x = self.dpfc(F.relu(self.fc1(x)))
        return self.fc2(x)
