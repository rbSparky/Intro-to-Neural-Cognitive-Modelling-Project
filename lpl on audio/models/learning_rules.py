import torch

class LPLLearning:
    def __init__(self, model, device, lr=1e-3, alpha=0.01, beta=0.001):
        self.model = model
        self.device = device
        self.lr = lr
        self.alpha = alpha
        self.beta = beta

    def update_weights(self, x, y_pred, y_true):
        x = x.to(self.device)
        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)

        y_pred_probs = torch.softmax(y_pred, dim=1)

        hebbian = torch.matmul(y_true.T, x)
        anti_hebbian = torch.matmul((y_pred_probs - y_true).T, x)
        delta_w = self.lr * (self.alpha * hebbian + self.beta * anti_hebbian)

        for name, param in self.model.named_parameters():
            if 'fc2.weight' in name:
                if param.shape != delta_w.shape:
                    raise ValueError(f"Shape mismatch for {name}: expected {param.shape}, got {delta_w.shape}")
                param.data += delta_w.to(self.device)
            elif 'fc2.bias' in name:
                delta_b = self.lr * (
                    self.alpha * y_true.sum(dim=0) +
                    self.beta * (y_pred_probs - y_true).sum(dim=0)
                )
                param.data += delta_b.to(self.device)
        return
