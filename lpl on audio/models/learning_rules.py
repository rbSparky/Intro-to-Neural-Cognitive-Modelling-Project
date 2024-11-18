# models/learning_rules.py
import torch

class LPLLearning:
    def __init__(self, model, device, lr=1e-3, alpha=0.05, beta=0.005):
        """
        Initializes the LPLLearning instance.

        Args:
            model (nn.Module): The neural network model.
            device (torch.device): The device to perform computations on.
            lr (float): Learning rate for the LPL rule.
            alpha (float): Weight for the Hebbian term.
            beta (float): Weight for the anti-Hebbian term.
        """
        self.model = model
        self.device = device
        self.lr = lr
        self.alpha = alpha
        self.beta = beta

    def update_weights(self, x, y_pred, y_true):
        """
        Updates the model's final layer weights and biases based on the LPL rule.

        Args:
            x (torch.Tensor): The input features (output from fc1 or equivalent).
            y_pred (torch.Tensor): The model's predictions (logits).
            y_true (torch.Tensor): The one-hot encoded true labels.
        """
        # Ensure tensors are on the correct device
        x = x.to(self.device)
        y_pred = y_pred.to(self.device)
        y_true = y_true.to(self.device)

        # Compute delta_w based on the LPL learning rule
        delta_w = self.lr * (
            self.alpha * torch.matmul(y_true.T, x) +
            self.beta * torch.matmul((y_pred - y_true).T, x)
        )

        # Apply delta_w to the final linear layer's weights and biases
        for name, param in self.model.named_parameters():
            if 'fc.weight' in name:
                if param.shape != delta_w.shape:
                    raise ValueError(f"Shape mismatch for {name}: expected {param.shape}, got {delta_w.shape}")
                param.data += delta_w.to(self.device)
            elif 'fc.bias' in name:
                # Compute delta_b separately
                delta_b = self.lr * (
                    self.alpha * y_true.sum(dim=0) +
                    self.beta * (y_pred - y_true).sum(dim=0)
                )
                param.data += delta_b.to(self.device)
        return
