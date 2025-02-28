import numpy as np
import matplotlib.pyplot as plt

def hrf_function(t, delay_response, delay_undershoot, dispersion_response, dispersion_undershoot, ratio):
    """Computes the HRF given a set of parameters."""
    peak = (t ** delay_response) * np.exp(-t / dispersion_response)
    peak /= np.max(peak)  # Normalize

    undershoot = (t ** delay_undershoot) * np.exp(-t / dispersion_undershoot)
    undershoot /= np.max(undershoot)  # Normalize

    hrf = peak - (undershoot / ratio)
    return hrf


def plot_hrf(delay_response, delay_undershoot, dispersion_response, dispersion_undershoot, ratio, kernel):
    """Plots the HRF with given parameters."""
    t = np.linspace(0, kernel, 1000)  # Time from 0 to 32s
    hrf = hrf_function(t, delay_response, delay_undershoot, dispersion_response, dispersion_undershoot, ratio)

    plt.figure(figsize=(8, 4))
    plt.plot(t, hrf, label='HRF')
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.title('Hemodynamic Response Function')
    plt.legend()
    plt.grid(True)
    plt.show()