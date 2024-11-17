import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('plots', exist_ok=True)

def plot_network_activity(input_spikes, exc_spikes, inh_spikes, input_signals, reconstructions, dt, t_start=0, t_end=100000):
    """
    Plot network activity and signal reconstructions.

    Parameters:
    - input_spikes (ndarray): Input spike matrix (n_inputs x n_steps).
    - exc_spikes (ndarray): Excitatory spike matrix (n_exc x n_steps).
    - inh_spikes (ndarray): Inhibitory spike matrix (n_inh x n_steps).
    - input_signals (ndarray): Input firing rates (n_inputs x n_steps).
    - reconstructions (dict): Reconstructed signals.
    - dt (float): Time step (ms).
    - t_start (int): Start time for plotting (ms).
    - t_end (int): End time for plotting (ms).
    """
    fig = plt.figure(figsize=(20, 12))
    t = np.arange(t_start, t_end) * dt

    ax1 = plt.subplot(221)
    for i in range(input_spikes.shape[0]):
        spike_times = np.where(input_spikes[i, t_start:t_end])[0] * dt
        ax1.plot(spike_times, i * np.ones_like(spike_times), '.', markersize=1)
    ax1.set_ylabel('Input Population')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title('Input Spikes')
    ax1.set_xlim([t_start * dt, t_end * dt])

    ax3 = plt.subplot(222)
    ax3.plot(t, input_signals[1, t_start:t_end], label='P1')
    ax3.plot(t, input_signals[3, t_start:t_end], label='P2')
    ax3.set_ylabel('Firing Rate (Hz)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_title('Input Signals')
    ax3.legend()
    ax3.set_xlim([t_start * dt, t_end * dt])

    ax2 = plt.subplot(223)
    n_exc = exc_spikes.shape[0]
    for i in range(n_exc):
        spike_times = np.where(exc_spikes[i, t_start:t_end])[0] * dt
        ax2.plot(spike_times, i * np.ones_like(spike_times), '.', color='black', markersize=1)
    n_inh = inh_spikes.shape[0]
    for i in range(n_inh):
        spike_times = np.where(inh_spikes[i, t_start:t_end])[0] * dt
        ax2.plot(spike_times, (i + n_exc) * np.ones_like(spike_times), '.', color='red', markersize=1)
    ax2.set_ylabel('Neuron Index')
    ax2.set_xlabel('Time (ms)')
    ax2.set_title('Network Activity')
    ax2.set_xlim([t_start * dt, t_end * dt])

    ax4 = plt.subplot(224)
    bin_duration = 100
    bin_indices = np.arange(len(reconstructions['P1'])) * bin_duration + bin_duration / 2
    ax4.plot(bin_indices, reconstructions['P1'], 'b', label='P1 Reconstructed')
    ax4.plot(bin_indices, reconstructions['P1_orig'], 'b--', label='P1 Original')
    ax4.plot(bin_indices, reconstructions['P2'], 'r', label='P2 Reconstructed')
    ax4.plot(bin_indices, reconstructions['P2_orig'], 'r--', label='P2 Original')
    ax4.set_ylabel('Signal')
    ax4.set_xlabel('Time (ms)')
    ax4.set_title('Signal Reconstruction')
    ax4.legend()
    ax4.set_xlim([t_start * dt, t_end * dt])

    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'network_activity.png'))
    plt.close(fig)

def plot_weight_distributions(mean_weights, labels):
    """
    Plot the distribution of synaptic weights.

    Parameters:
    - mean_weights (dict): Mean weights for each input population.
    - labels (list): Labels for the input populations.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [mean_weights[label] for label in labels]
    sns.violinplot(data=data, ax=ax)
    ax.set_ylabel('Mean Synaptic Strength')
    ax.set_title('Mean Afferent Synaptic Strength per Excitatory Neuron')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'weight_distributions.png'))
    plt.close(fig)

def plot_selectivity(selectivity_data, conditions):
    """
    Plot signal selectivity.

    Parameters:
    - selectivity_data (ndarray): Selectivity metrics.
    - conditions (list): Labels for conditions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(conditions))
    width = 0.35
    ax.bar(x - width/2, selectivity_data[:, 0], width, label='P1')
    ax.bar(x + width/2, selectivity_data[:, 1], width, label='P2')
    ax.set_ylabel('Signal Selectivity')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'selectivity.png'))
    plt.close(fig)

def plot_firing_rates(rates, conditions):
    """
    Plot average firing rates.

    Parameters:
    - rates (list): Average firing rates.
    - conditions (list): Labels for conditions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(conditions, rates)
    ax.set_ylabel('Average Firing Rate (Hz)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'firing_rates.png'))
    plt.close(fig)

def plot_dimensionality(dims, conditions):
    """
    Plot network dimensionality.

    Parameters:
    - dims (list): Dimensionality metrics.
    - conditions (list): Labels for conditions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(conditions, dims)
    ax.set_ylabel('Dimensionality')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'dimensionality.png'))
    plt.close(fig)

def plot_weight_selectivity(w_p1, w_p2, title):
    """
    Plot weight selectivity.

    Parameters:
    - w_p1 (ndarray): Weights for P1.
    - w_p2 (ndarray): Weights for P2.
    - title (str): Title for the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    scatter = ax1.scatter(w_p1, w_p2, c=w_p1 - w_p2, cmap='RdBu')
    ax1.set_xlabel('wP1')
    ax1.set_ylabel('wP2')
    ax1.set_title(f'Weight Distribution ({title})')
    plt.colorbar(scatter, ax=ax1, label='wP1 - wP2')
    selectivity = (w_p1 - w_p2) / (w_p1 + w_p2 + 1e-10)
    ax2.hist(selectivity, bins=30, color='gray')
    ax2.set_xlabel('Relative Selectivity')
    ax2.set_ylabel('Count')
    ax2.set_title('Selectivity Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'weight_selectivity_{title}.png'))
    plt.close(fig)
