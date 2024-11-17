import numpy as np

def create_input_signals(T, dt):
    """
    Create input firing rate signals.

    Parameters:
    - T (int): Total simulation time (ms).
    - dt (float): Time step (ms).

    Returns:
    - signals (ndarray): Input firing rates (n_inputs x n_steps).
    """
    t = np.arange(0, T, dt)
    n_steps = len(t)
    baseline_rate = 50

    P0 = np.full(n_steps, baseline_rate)
    freq_P1 = 2.0
    freq_P2 = 3.0
    amplitude = 20
    P1 = baseline_rate + amplitude * np.sin(2 * np.pi * freq_P1 * t / 1000.0)
    P2 = baseline_rate + amplitude * np.sin(2 * np.pi * freq_P2 * t / 1000.0)
    P1_ctl = np.copy(P1)
    P2_ctl = np.copy(P2)
    np.random.shuffle(P1_ctl)
    np.random.shuffle(P2_ctl)

    signals = np.vstack([P0, P1, P1_ctl, P2, P2_ctl])
    return signals

def generate_poisson_input(rates, dt):
    """
    Generate Poisson-distributed spike trains based on input rates.

    Parameters:
    - rates (ndarray): Firing rates (n_inputs x n_steps).
    - dt (float): Time step (ms).

    Returns:
    - spikes (ndarray): Binary spike matrix (n_inputs x n_steps).
    """
    n_steps = rates.shape[1]
    spikes = np.random.rand(*rates.shape) < (rates * dt / 1000.0)
    return spikes.astype(float)

def effective_rank(A):
    """
    Compute the effective rank of a matrix using entropy.

    Parameters:
    - A (ndarray): Input matrix.

    Returns:
    - float: Effective rank.
    """
    s = np.linalg.svd(A, compute_uv=False)
    s = s[s > 0]
    s_norm = s / np.sum(s)
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
    rank = np.exp(entropy)
    return rank
