import torch
import torch.nn as nn
import torch.nn.functional as F

# FOR MNIST
# class SimpleCNN(nn.Module):
#     def __init__(self, input_channels=1, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(64 * 16 * 16, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x_flat = x.view(-1, 64 * 16 * 16)
#         representation = F.relu(self.fc1(x_flat))
#         output = self.fc2(representation)
#         return x_flat, output


# FOR LIBRISPEECH

# models/cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=41):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  # Output: 32 x 64 x 64
        self.pool = nn.MaxPool2d(2, 2)  # Output after pool: 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64 x 32 x 32
        self.pool = nn.MaxPool2d(2, 2)  # Output after second pool: 64 x 16 x 16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 16, 16]
        x_flat = x.view(-1, 64 * 16 * 16)      # [batch, 1024]
        representation = F.relu(self.fc1(x_flat))  # [batch, 128]
        output = self.fc2(representation)          # [batch, num_classes]
        return x_flat, output
