import numpy as np

class SpikingNeuron:
    def __init__(self, is_inhibitory=False, tau_m=20.0, v_rest=-65.0, v_thresh=-60.0, v_reset=-65.0):
        """
        Initialize a spiking neuron.

        Parameters:
        - is_inhibitory (bool): Whether the neuron is inhibitory.
        - tau_m (float): Membrane time constant (ms).
        - v_rest (float): Resting membrane potential (mV).
        - v_thresh (float): Spike threshold (mV).
        - v_reset (float): Reset potential after a spike (mV).
        """
        self.is_inhibitory = is_inhibitory
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v_rest
        self.spike_times = []

    def update(self, I_syn, dt, t):
        """
        Update the membrane potential and determine if a spike occurs.

        Parameters:
        - I_syn (float): Synaptic input current.
        - dt (float): Time step (ms).
        - t (float): Current time (ms).

        Returns:
        - float: 1.0 if spike occurred, else 0.0.
        """
        dv = (-(self.v - self.v_rest) + I_syn) * (dt / self.tau_m)
        self.v += dv

        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.spike_times.append(t)
            return 1.0
        return 0.0
