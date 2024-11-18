# models/cnn_model.py
import torch  # Added import statement
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetSpeaker(nn.Module):
    def __init__(self, num_classes=251):  # Update num_classes as per your dataset
        super(ResNetSpeaker, self).__init__()
        # Updated to use 'weights' instead of 'pretrained' as per the warning
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Modify the first convolutional layer to accept single-channel (grayscale) input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        features = self.resnet.conv1(x)
        features = self.resnet.bn1(features)
        features = self.resnet.relu(features)
        features = self.resnet.maxpool(features)

        features = self.resnet.layer1(features)
        features = self.resnet.layer2(features)
        features = self.resnet.layer3(features)
        features = self.resnet.layer4(features)

        features = self.resnet.avgpool(features)
        features = torch.flatten(features, 1)
        logits = self.resnet.fc(features)
        return features, logits
