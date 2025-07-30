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

def estimate_coupling_strength(
    device,
    qubit_id: str,
    resonator_id: str,
    drive_duration: float = 100,
    drive_amplitude: float = 0.1,
    freq: float = None,
    tlist: np.ndarray = None
) -> float:
    hs = device.hilbertspace
    H0 = device.get_qutip_hamiltonian()

    qidx = device.get_subsystem_index(qubit_id)
    qubit = hs.subsystem_list[qidx]
    ridx = device.get_subsystem_index(resonator_id)
    resonator = hs.subsystem_list[ridx]

    if freq is None:
        freq = qubit.get_freq()

    if tlist is None:
        tlist = np.linspace(0, drive_duration, 200)

    psi0 = tensor([basis(sub.truncated_dim, 0) for sub in hs.subsystem_list])

    a_q = destroy(qubit.truncated_dim)
    a_r = destroy(resonator.truncated_dim)

    def embed_single_op(op, target_subsys):
        ops = [
            op if sub is target_subsys else identity(sub.truncated_dim)
            for sub in hs.subsystem_list
        ]
        return tensor(ops)

    drive_op = embed_single_op(a_q + a_q.dag(), qubit)
    n_r = embed_single_op(a_r.dag() * a_r, resonator)

    H_t = [H0, [drive_op, lambda t, args: drive_amplitude * np.cos(2 * np.pi * freq * t)]]


    result = mesolve(
        H_t,
        psi0,
        tlist,
        e_ops=[n_r]
    )

    avg_photons = np.array(result.expect[0])
    peak = np.max(avg_photons)
    g_est = np.sqrt(peak) / drive_duration if drive_duration > 0 else 0.0

    return g_est
