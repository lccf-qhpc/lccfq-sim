"""
Filename: gates.py
Author: Santiago Nunez-Corrales
Date: 2025-07-029
Version: 1.0
Description:
    This file implements tests for native gates

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
import numpy as np

from qutip import Qobj, identity, tensor, basis, qeye
from lccfq_sim.gates import X_gate, Y_gate, SQISWAP_gate


def is_unitary(U: Qobj, tol=1e-10) -> bool:
    I = qeye(U.dims[0])  # Fix: use compound input dimensions
    return (U.dag() * U - I).norm() < tol


def test_X_gate_unitary_and_action():
    X = X_gate()
    assert is_unitary(X), "X gate is not unitary"
    assert X.shape == (2, 2)
    result = X * Qobj([[1], [0]])  # |0⟩ -> |1⟩
    expected = Qobj([[0], [1]])
    assert (result - expected).norm() < 1e-10


def test_Y_gate_unitary_and_action():
    Y = Y_gate()
    assert is_unitary(Y), "Y gate is not unitary"
    assert Y.shape == (2, 2)
    result = Y * Qobj([[1], [0]])  # |0⟩ -> i|1⟩
    expected = Qobj([[0], [1j]])
    assert (result - expected).norm() < 1e-10


def test_sqrt_iswap_unitary_and_dims():
    S = SQISWAP_gate(2)
    assert is_unitary(S), "√iSWAP gate is not unitary"
    assert S.shape == (4, 4)
    assert S.dims == [[2, 2], [2, 2]]

    ket_01 = tensor(basis(2, 0), basis(2, 1))
    assert ket_01.dims == [[2, 2], [1]]

    out = S * ket_01
    assert out.norm() - 1.0 < 1e-10