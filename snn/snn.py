import numpy as np
from neuron import SpikingNeuron

class SNN:
    def __init__(self, n_exc=100, n_inh=25, n_inputs=5):
        """
        Initialize the Spiking Neural Network (SNN).

        Parameters:
        - n_exc (int): Number of excitatory neurons.
        - n_inh (int): Number of inhibitory neurons.
        - n_inputs (int): Number of input populations.
        """
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_inputs = n_inputs

        self.exc_neurons = [SpikingNeuron(is_inhibitory=False) for _ in range(n_exc)]
        self.inh_neurons = [SpikingNeuron(is_inhibitory=True) for _ in range(n_inh)]

        self.w_input = np.abs(np.random.randn(n_inputs, n_exc)) * 5.0
        conn_prob = 0.1

        self.w_ee = np.abs(np.random.randn(n_exc, n_exc)) * 1.0 * (np.random.rand(n_exc, n_exc) < conn_prob)
        np.fill_diagonal(self.w_ee, 0)

        self.w_ei = np.abs(np.random.randn(n_exc, n_inh)) * 1.0 * (np.random.rand(n_exc, n_inh) < conn_prob)

        self.w_ie = -np.abs(np.random.randn(n_inh, n_exc)) * 1.0 * (np.random.rand(n_inh, n_exc) < conn_prob)

    def simulate(self, input_spikes, dt):
        """
        Simulate the SNN over time.

        Parameters:
        - input_spikes (ndarray): Binary spike matrix for inputs (n_inputs x n_steps).
        - dt (float): Time step (ms).

        Returns:
        - exc_spikes (ndarray): Excitatory spike matrix (n_exc x n_steps).
        - inh_spikes (ndarray): Inhibitory spike matrix (n_inh x n_steps).
        """
        n_steps = input_spikes.shape[1]
        exc_spikes = np.zeros((self.n_exc, n_steps))
        inh_spikes = np.zeros((self.n_inh, n_steps))

        tau_syn = 5.0
        I_syn_exc = np.zeros(self.n_exc)
        I_syn_inh = np.zeros(self.n_inh)

        for t in range(n_steps):
            I_syn_exc *= np.exp(-dt / tau_syn)
            I_syn_inh *= np.exp(-dt / tau_syn)

            I_syn_exc += np.dot(self.w_input.T, input_spikes[:, t])

            if t > 0:
                I_syn_exc += np.dot(self.w_ee, exc_spikes[:, t-1])
                I_syn_exc += np.dot(self.w_ie.T, inh_spikes[:, t-1])
                I_syn_inh += np.dot(self.w_ei.T, exc_spikes[:, t-1])

            for i, neuron in enumerate(self.exc_neurons):
                exc_spikes[i, t] = neuron.update(I_syn_exc[i], dt, t*dt)

            for i, neuron in enumerate(self.inh_neurons):
                inh_spikes[i, t] = neuron.update(I_syn_inh[i], dt, t*dt)

        return exc_spikes, inh_spikes
