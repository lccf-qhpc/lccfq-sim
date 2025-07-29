"""
Filename: calibration.py
Author: Santiago Nunez-Corrales
Date: 2025-07-029
Version: 1.0
Description:
    This file implements low-level device primitives

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np

from qutip import basis, mesolve, sigmax, tensor, Qobj, identity, destroy
from .device import QPUDevice


def qubit_spectroscopy(
    device: QPUDevice,
    qubit_id: str,
    freq_range: np.ndarray,
    drive_amplitude: float = 0.05,
    drive_duration: float = 50.0,
) -> np.ndarray:
    """
    Simulate spectroscopy of a single qubit using a drive sweep.
    """

    idx = device.get_subsystem_index(qubit_id)
    qubit_dim = device.hilbertspace.subsystem_list[idx].truncated_dim

    psi0 = tensor([
        basis(sub.truncated_dim, 0) for sub in device.hilbertspace.subsystem_list
    ])

    H0 = device.get_qutip_hamiltonian()
    X_gate = sigmax()
    drive_operator = device.embed_gate(X_gate, [qubit_id])

    excitation_probs = []

    for freq in freq_range:
        # H(t) = H0 + A * cos(2 PI f t) * sigma_x
        H_t = [
            H0,
            [drive_operator, lambda t, args=None: np.cos(2 * np.pi * freq * t)]
        ]

        result = mesolve(
            H_t,
            psi0,
            tlist=np.linspace(0, drive_duration, 100),
            e_ops=[],
            options=None
        )

        final_state = result.states[-1]

        projector = tensor([
            Qobj(np.diag([0, 1]), dims=[[qubit_dim], [qubit_dim]]) if i == idx
            else identity(sub.truncated_dim)
            for i, sub in enumerate(device.hilbertspace.subsystem_list)
        ])

        prob = abs((final_state.dag() * projector * final_state))
        excitation_probs.append(prob)

    return np.array(excitation_probs)

def resonator_spectroscopy(
    device,
    resonator_id: str,
    freq_range: np.ndarray,
    drive_duration: float = 100.0
) -> float:
    idx = device.get_subsystem_index(resonator_id)
    d = device.hilbertspace.subsystem_list[idx].truncated_dim
    a = destroy(d)
    drive_op = a + a.dag()
    n = destroy(d).dag() * destroy(d)

    n_op = tensor([
        n if i == idx else identity(sub.truncated_dim)
        for i, sub in enumerate(device.hilbertspace.subsystem_list)
    ])

    drive = tensor([
        drive_op if i == idx else identity(sub.truncated_dim)
        for i, sub in enumerate(device.hilbertspace.subsystem_list)
    ])

    print(drive)

    psi0 = tensor([
        basis(sub.truncated_dim, 0) for sub in device.hilbertspace.subsystem_list
    ])

    tlist = np.linspace(0, drive_duration, 200)
    results = []

    for freq in freq_range:
        H0 = device.get_qutip_hamiltonian()
        H_t = [H0, [drive + drive.dag(), lambda t, args=None: np.cos(2 * np.pi * freq * t)]]
        result = mesolve(H_t, psi0, tlist, e_ops=[n_op])
        avg_n = np.mean(result.expect[0])
        results.append(avg_n)

    best_idx = int(np.argmax(results))
    return freq_range[best_idx]

