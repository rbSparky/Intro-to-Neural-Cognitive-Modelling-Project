import torch
import torch.nn as nn

class HebbianLearning(nn.Module):
    def __init__(self, input_dim, output_dim, lr=1e-3):
        super(HebbianLearning, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        self.lr = lr

    def forward(self, x, y):
        delta_w = self.lr * torch.matmul(x.T, y)
        self.weights.data += delta_w
        return self.weights

class BcmLearning(nn.Module):
    def __init__(self, input_dim, output_dim, lr=1e-3, theta=0.5):
        super(BcmLearning, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        self.lr = lr
        self.theta = theta

    def forward(self, x, y):
        delta_w = self.lr * torch.matmul(x.T, y * (x * y - self.theta * y))
        self.weights.data += delta_w
        return self.weights


class LPLLearning(nn.Module):
    def __init__(self, input_dim, output_dim, lr=1e-3, alpha=0.1, beta=0.01):
        super(LPLLearning, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        self.lr = lr
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y_pred, y_true):
        delta_w = self.lr * (
            self.alpha * torch.matmul(x.T, y_true) +
            self.beta * torch.matmul(x.T, (y_pred - y_true))
        )
        
        if self.weights.shape != delta_w.shape:
            raise ValueError(f"Shape mismatch: weights {self.weights.shape} vs delta_w {delta_w.shape}")
        
        self.weights.data += delta_w
        return self.weights
