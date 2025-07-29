"""
Filename: device_test.py
Author: Santiago Nunez-Corrales
Date: 2025-07-029
Version: 1.0
Description:
    This file implements tests for a transmon device

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
import numpy as np
from qutip import basis, tensor, sigmax, Qobj
from lccfq_sim.device import QPUDevice


@pytest.fixture
def small_device():
    return QPUDevice(num_qubits=2)


def test_subsystem_ids(small_device):
    ids = small_device.list_subsystems()
    assert ids == ["Q0", "Q1", "R_ro_0", "R_ro_1", "R_bus_0"]
    assert all(isinstance(s, str) for s in ids)


def test_hamiltonian_structure(small_device):
    H = small_device.get_qutip_hamiltonian()
    subsystems = small_device.hilbertspace.subsystem_list

    def get_effective_dim(sub):
        if hasattr(sub, "truncated_dim"):
            return sub.truncated_dim
        return sub.hilbertdim()

    dims = [get_effective_dim(sub) for sub in subsystems]

    print("Effective subsystem dims:", dims)
    expected_dim = np.prod(dims)
    print("Expected total dim:", expected_dim)
    print("Actual Hamiltonian shape:", H.shape)

    assert H.isherm
    assert H.shape[0] == expected_dim


def test_subsystem_indexing(small_device):
    assert small_device.get_subsystem_index("Q0") == 0
    assert small_device.get_subsystem_index("Q1") == 1
    assert small_device.get_subsystem_index("R_bus_0") == 4
    with pytest.raises(ValueError):
        small_device.get_subsystem_index("nonexistent")


def test_embed_single_qubit_gate(small_device):
    X = sigmax()
    U = small_device.embed_gate(X, ["Q0"])
    assert U.shape[0] == small_device.get_qutip_hamiltonian().shape[0]
    assert U.isunitary


def test_apply_gate(small_device):
    # Ensure Transmon uses truncated_dim = 2
    dim_q0 = small_device.hilbertspace.subsystem_list[0].truncated_dim
    assert dim_q0 == 2, f"Test assumes Q0 is a 2-level system -- {dim_q0}."

    psi0 = tensor([
        basis(sub.truncated_dim, 0)
        for sub in small_device.hilbertspace.subsystem_list
    ])

    print(psi0)

    # Apply Pauli-X on Q0 to flip it to |1⟩
    X = sigmax()
    psi1 = small_device.apply_gate(psi0, X, ["Q0"])

    # Confirm normalization
    assert abs(psi1.norm() - 1.0) < 1e-10

    # Check that Q0 flipped to |1⟩ and rest stayed in |0⟩s
    target_state = tensor(
        basis(2, 1),  # Q0 flipped
        basis(2, 0),  # Q1 unchanged
        basis(5, 0),  # R_ro_0
        basis(5, 0),  # R_ro_1
        basis(5, 0)   # R_bus_0
    )
    fidelity = abs(target_state.overlap(psi1))
    assert abs(fidelity - 1.0) < 1e-10


def test_time_evolution_conserves_norm(small_device):
    dims = [sub.truncated_dim if hasattr(sub, "truncated_dim") else sub.hilbertdim()
            for sub in small_device.hilbertspace.subsystem_list]

    psi0 = tensor(*[basis(d, 0) for d in dims])

    # Use proper dimensioned X gate for Q0
    dim_q0 = dims[0]
    X = Qobj(np.diag([1]*(dim_q0-1), 1) + np.diag([1]*(dim_q0-1), -1), dims=[[dim_q0], [dim_q0]])

    psi1 = small_device.apply_gate(psi0, X, ["Q0"])

    tlist = np.linspace(0, 2, 20)
    result = small_device.evolve_state(psi1, tlist)

    norms = [psi.norm() for psi in result.states]
    assert all(abs(n - 1.0) < 1e-10 for n in norms)

def test_apply_named_gate(small_device):
    dim_q0 = small_device.hilbertspace.subsystem_list[0].truncated_dim
    assert dim_q0 == 2, f"Test assumes Q0 is 2-level, got {dim_q0}"

    psi0 = tensor([
        basis(sub.truncated_dim, 0)
        for sub in small_device.hilbertspace.subsystem_list
    ])

    psi1 = small_device.apply_named_gate(psi0, "X", ["Q0"])

    # Confirm normalization
    assert abs(psi1.norm() - 1.0) < 1e-10

    # Fidelity check
    target = tensor(basis(2, 1), basis(2, 0), basis(5, 0), basis(5, 0), basis(5, 0))
    fidelity = abs((target.dag() * psi1))
    assert abs(fidelity - 1.0) < 1e-10
