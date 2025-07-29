"""
Filename: gates.py
Author: Santiago Nunez-Corrales
Date: 2025-07-029
Version: 1.0
Description:
    This file implements gates in the native set

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np
from qutip import Qobj, sigmax, sigmay


def X_gate(dim: int = 2) -> Qobj:
    if dim != 2:
        raise NotImplementedError("X gate only implemented for 2-level systems.")
    return sigmax()

def Y_gate(dim: int = 2) -> Qobj:
    if dim != 2:
        raise NotImplementedError("Y gate only implemented for 2-level systems.")
    return sigmay()

def SQISWAP_gate(dim: int = 2) -> Qobj:
    if dim != 2:
        raise NotImplementedError("√iSWAP only implemented for 2-level systems.")

    iswap_matrix = np.array([
        [1, 0,     0,    0],
        [0, 0.5+0.5j, 0.5-0.5j, 0],
        [0, 0.5-0.5j, 0.5+0.5j, 0],
        [0, 0,     0,    1]
    ])
    return Qobj(iswap_matrix, dims=[[dim, dim], [dim, dim]])

NATIVE_GATES = {
    "X": X_gate,
    "Y": Y_gate,
    "SQiSWAP": SQISWAP_gate
}
