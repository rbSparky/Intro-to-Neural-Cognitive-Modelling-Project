import numpy as np

class LPL:
    def __init__(self, learning_rate=0.001, predictive_factor=0.1):
        """
        Initialize the Learning Predictive Learning (LPL) rule.

        Parameters:
        - learning_rate (float): Learning rate for weight updates.
        - predictive_factor (float): Factor for predictive term.
        """
        self.lr = learning_rate
        self.pred_factor = predictive_factor

    def update_weights(self, pre_act, post_act, weights):
        """
        Update synaptic weights based on pre- and post-synaptic activity.

        Parameters:
        - pre_act (ndarray): Pre-synaptic activity vector.
        - post_act (ndarray): Post-synaptic activity vector.
        - weights (ndarray): Current synaptic weights.

        Returns:
        - ndarray: Updated synaptic weights.
        """
        hebbian = np.outer(pre_act, post_act)
        pred_error = post_act - np.dot(weights.T, pre_act)
        pred_term = np.outer(pre_act, pred_error)
        dw = self.lr * (hebbian + self.pred_factor * pred_term)
        return weights + dw
