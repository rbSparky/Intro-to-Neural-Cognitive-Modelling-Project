import numpy as np
from snn import SNN
from learning import LPL
from utils import create_input_signals, generate_poisson_input, effective_rank
from plotting import (
    plot_network_activity,
    plot_weight_distributions,
    plot_selectivity,
    plot_firing_rates,
    plot_dimensionality,
    plot_weight_selectivity
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    T = 100000
    dt = 1.0
    n_inputs = 5

    input_signals = create_input_signals(T, dt)
    input_spikes = generate_poisson_input(input_signals, dt)

    snn = SNN(n_exc=100, n_inh=25, n_inputs=n_inputs)
    lpl = LPL(learning_rate=0.001, predictive_factor=0.1)

    n_steps = int(T / dt)
    chunk_size = int(1000 / dt)
    exc_spikes = np.zeros((snn.n_exc, n_steps))
    inh_spikes = np.zeros((snn.n_inh, n_steps))

    enable_learning = False

    for start in range(0, n_steps, chunk_size):
        end = min(start + chunk_size, n_steps)
        exc_chunk, inh_chunk = snn.simulate(input_spikes[:, start:end], dt)
        exc_spikes[:, start:end] = exc_chunk
        inh_spikes[:, start:end] = inh_chunk

        if enable_learning:
            t_start = max(0, start - int(100 / dt))
            pre_activity = np.mean(input_spikes[:, t_start:end], axis=1)
            post_activity = np.mean(exc_spikes[:, t_start:end], axis=1)
            snn.w_input = lpl.update_weights(pre_activity, post_activity, snn.w_input)
            snn.w_input = np.clip(snn.w_input, 0.0, 5.0)

    total_exc_spikes = np.sum(exc_spikes)
    print(f'Total excitatory spikes: {total_exc_spikes}')
    if total_exc_spikes == 0:
        print("No excitatory spikes occurred. Adjust the parameters.")

    window_size = 100
    bin_size = int(window_size / dt)
    n_bins = int(n_steps / bin_size)
    exc_rates = np.zeros((snn.n_exc, n_bins))
    for i in range(n_bins):
        start_bin = i * bin_size
        end_bin = start_bin + bin_size
        exc_rates[:, i] = np.sum(exc_spikes[:, start_bin:end_bin], axis=1) * (1000.0 / window_size)

    input_signal_P1 = np.mean(input_signals[1, :].reshape(-1, bin_size), axis=1)
    input_signal_P2 = np.mean(input_signals[3, :].reshape(-1, bin_size), axis=1)

    min_length = min(len(input_signal_P1), exc_rates.shape[1])
    input_signal_P1 = input_signal_P1[:min_length]
    input_signal_P2 = input_signal_P2[:min_length]
    exc_rates = exc_rates[:, :min_length]

    model_P1 = LinearRegression()
    model_P2 = LinearRegression()
    X = exc_rates.T
    y_P1 = input_signal_P1
    y_P2 = input_signal_P2

    if np.var(X, axis=0).any():
        model_P1.fit(X, y_P1)
        model_P2.fit(X, y_P2)
        recon_P1 = model_P1.predict(X)
        recon_P2 = model_P2.predict(X)
        r2_P1 = r2_score(y_P1, recon_P1)
        r2_P2 = r2_score(y_P2, recon_P2)
        print(f'R^2 for P1 reconstruction: {r2_P1:.4f}')
        print(f'R^2 for P2 reconstruction: {r2_P2:.4f}')
        reconstructions = {
            'P1': recon_P1,
            'P2': recon_P2,
            'P1_orig': y_P1,
            'P2_orig': y_P2
        }
    else:
        print("Input features have no variance. Cannot perform regression.")
        reconstructions = {
            'P1': np.zeros_like(y_P1),
            'P2': np.zeros_like(y_P2),
            'P1_orig': y_P1,
            'P2_orig': y_P2
        }

    t_start_plot = 0
    t_end_plot = min(100000, n_steps)
    plot_network_activity(
        input_spikes,
        exc_spikes,
        inh_spikes,
        input_signals,
        reconstructions,
        dt,
        t_start=t_start_plot,
        t_end=t_end_plot
    )

    mean_weights = {}
    labels = ['P0', 'P1', 'P1_ctl', 'P2', 'P2_ctl']
    for i, label in enumerate(labels):
        mean_weights[label] = snn.w_input[i, :]

    plot_weight_distributions(mean_weights, labels)

    selectivity_P1 = (np.mean(snn.w_input[1, :]) - np.mean(snn.w_input[2, :])) / (np.mean(snn.w_input[1, :]) + np.mean(snn.w_input[2, :]) + 1e-10)
    selectivity_P2 = (np.mean(snn.w_input[3, :]) - np.mean(snn.w_input[4, :])) / (np.mean(snn.w_input[3, :]) + np.mean(snn.w_input[4, :]) + 1e-10)
    selectivity_data = np.array([[selectivity_P1, selectivity_P2]])
    conditions = ['SNN_No_Learning']
    plot_selectivity(selectivity_data, conditions)

    avg_exc_rate = np.sum(exc_spikes) / (snn.n_exc * T / 1000.0)
    avg_inh_rate = np.sum(inh_spikes) / (snn.n_inh * T / 1000.0)
    rates = [avg_exc_rate]
    plot_firing_rates(rates, conditions)

    dim_exc = effective_rank(exc_spikes)
    dims = [dim_exc]
    plot_dimensionality(dims, conditions)

    w_p1 = snn.w_input[1, :]
    w_p2 = snn.w_input[3, :]
    plot_weight_selectivity(w_p1, w_p2, title='SNN_No_Learning')

if __name__ == "__main__":
    main()
