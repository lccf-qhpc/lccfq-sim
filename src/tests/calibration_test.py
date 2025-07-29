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
from qutip import basis

from lccfq_sim.device import QPUDevice
from lccfq_sim.calibration import qubit_spectroscopy

def test_qubit_spectroscopy_returns_expected_frequency():
    # Create a simple device with 2-level transmons and readout resonators
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

    EJ = 20.0
    EC = 0.2
    expected_f01 = np.average(device.q_freqs)  # approximation for transmon frequency

    print(f01)
    print(f01 - expected_f01)
    print(expected_f01)

    # Numerical method ~200 MHz
    assert (np.abs(f01 - expected_f01) < 0.4).all(), f"Spectroscopy returned {f01}, expected approx {expected_f01}"
