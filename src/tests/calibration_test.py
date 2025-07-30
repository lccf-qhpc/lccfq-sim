"""
Filename: calibration_test.py
Author: Santiago Nunez-Corrales
Date: 2025-07-029
Version: 1.0
Description:
    This file implements testing for low-level device primitives

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np

from lccfq_sim.device import QPUDevice
from lccfq_sim.calibration import resonator_spectroscopy, qubit_spectroscopy, estimate_coupling_strength

def test_qubit_spectroscopy_returns_expected_frequency():
    device = QPUDevice(
        num_qubits=2,
        levels_qubit=2,
        levels_res=5,
        qubit_freqs=[5.0, 5.2],
        anharmonicities=[-0.2, -0.2],
        readout_freqs=[7.0, 7.1],
        bus_freqs=[5.0],
        g_qr=[0.05, 0.05],
        g_qb=[0.02, 0.02]
    )

    freq_range = np.linspace(4.5, 5.5, 20)
    probs = qubit_spectroscopy(device, "Q0", freq_range)
    f01 = freq_range[np.argmax(probs)]
    expected_f01 = np.average(device.q_freqs)  # approximation for transmon frequency

    # Numerical method ~200 MHz
    assert (np.abs(f01 - expected_f01) < 0.4).all(), f"Spectroscopy returned {f01}, expected approx {expected_f01}"

def test_resonator_spectroscopy_returns_expected_frequency():
    device = QPUDevice(
        num_qubits=2,
        levels_qubit=2,
        levels_res=5,
        qubit_freqs=[5.0, 5.2],
        anharmonicities=[-0.2, -0.2],
        readout_freqs=[7.0, 7.1],
        bus_freqs=[5.0],
        g_qr=[0.05, 0.05],
        g_qb=[0.02, 0.02]
    )

    freq_range = np.linspace(6.5, 7.5, 20)
    detected_freq = resonator_spectroscopy(device, "R_ro_0", freq_range)

    expected_freq = 7.0
    print(f"Detected resonator frequency: {detected_freq}")
    assert abs(detected_freq - expected_freq) < 0.7

def test_estimate_coupling_strength_returns_positive_value():
    device = QPUDevice(
        num_qubits=2,
        levels_qubit=2,
        levels_res=5,
        qubit_freqs=[5.0, 5.2],
        anharmonicities=[-0.2, -0.2],
        readout_freqs=[7.0, 7.1],
        bus_freqs=[5.0],
        g_qr=[0.05, 0.05],
        g_qb=[0.02, 0.02]
    )

    g_est = estimate_coupling_strength(
        device,
        qubit_id="Q0",
        resonator_id="R_ro_0",
        drive_duration=50,
        drive_amplitude=0.1,
        freq=5.0,
        tlist=np.linspace(0, 50, 100)
    )

    assert g_est > 0.0, f"Estimated coupling should be positive, got {g_est}"
    assert g_est < 1.0, f"Estimated coupling too large: {g_est}"
