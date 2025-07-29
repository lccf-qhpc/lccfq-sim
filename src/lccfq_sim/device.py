"""
Filename: device.py
Author: Santiago Nunez-Corrales
Date: 2025-07-029
Version: 1.0
Description:
    This file implements a transmon device

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np
import scqubits as scq

from qutip import Qobj, tensor, identity
from typing import List, Optional, Callable
from dataclasses import dataclass
from .gates import NATIVE_GATES


@dataclass
class NativeGate:
    name: str
    arity: int
    generator: Callable[[List[str]], Qobj]


class QPUDevice:
    def __init__(
        self,
        num_qubits: int,
        levels_qubit: int = 2,
        levels_res: int = 5,
        qubit_freqs: Optional[List[float]] = None,
        anharmonicities: Optional[List[float]] = None,
        readout_freqs: Optional[List[float]] = None,
        bus_freqs: Optional[List[float]] = None,
        g_qr: Optional[List[float]] = None,
        g_qb: Optional[List[float]] = None
    ):
        self.num_qubits = num_qubits
        self.q_freqs = qubit_freqs or [5.0 + 0.1 * i for i in range(num_qubits)]
        self.anh = anharmonicities or [0.2] * num_qubits
        self.r_freqs = readout_freqs or [6.5 + 0.05 * i for i in range(num_qubits)]
        self.b_freqs = bus_freqs or [7.0] * (num_qubits - 1)
        self.g_qr = g_qr or [0.05] * num_qubits
        self.g_qb = g_qb or [0.03] * (num_qubits - 1)

        self.qubits = []
        self.readouts = []
        self.buses = []

        # Physical constants
        self.EC = 0.2  # GHz
        self.hilbertspace = None

        self._build_subsystems(levels_qubit, levels_res)
        self._build_hilbertspace()
        self._add_interactions()

        # Gates
        self.native_gates = {}
        self._register_native_gates()

    def _build_subsystems(self, levels_qubit: int, levels_res: int):
        for i in range(self.num_qubits):
            EJ = (self.q_freqs[i] + self.anh[i] / 2) ** 2 / (8 * self.EC)
            qubit = scq.Transmon(
                EJ=EJ,
                EC=self.EC,
                ng=0.0,
                ncut=10,
                truncated_dim=levels_qubit,
                id_str=f"Q{i}"
            )
            self.qubits.append(qubit)

            readout = scq.Oscillator(
                E_osc=self.r_freqs[i],
                truncated_dim=levels_res,
                id_str=f"R_ro_{i}"
            )
            self.readouts.append(readout)

        for i in range(self.num_qubits - 1):
            bus = scq.Oscillator(
                E_osc=self.b_freqs[i],
                truncated_dim=levels_res,
                id_str=f"R_bus_{i}"
            )
            self.buses.append(bus)

    def _build_hilbertspace(self):
        subsystems = self.qubits + self.readouts + self.buses
        self.hilbertspace = scq.HilbertSpace(subsystem_list=subsystems)

    def _add_interactions(self):
        n = self.num_qubits
        # Access lists of actual subsystem objects
        qubits = self.qubits
        readouts = self.readouts
        buses = self.buses

        # Qubit–Readout interactions
        for i in range(n):
            self.hilbertspace.add_interaction(
                g=self.g_qr[i],
                op1=(qubits[i].n_operator, qubits[i]),
                op2=(readouts[i].creation_operator, readouts[i]),
                add_hc=True
            )

        # Qubit–Bus interactions
        for i in range(n - 1):
            bus = buses[i]
            q1 = qubits[i]
            q2 = qubits[i + 1]

            self.hilbertspace.add_interaction(
                g=self.g_qb[i],
                op1=(q1.n_operator, q1),
                op2=(bus.creation_operator, bus),
                add_hc=True
            )
            self.hilbertspace.add_interaction(
                g=self.g_qb[i],
                op1=(q2.n_operator, q2),
                op2=(bus.creation_operator, bus),
                add_hc=True
            )

    def get_hamiltonian(self) -> np.ndarray:
        return self.hilbertspace.hamiltonian().full()

    def get_qutip_hamiltonian(self):
        return self.hilbertspace.hamiltonian()

    def list_subsystems(self):
        return [obj.id_str for obj in self.hilbertspace.subsystem_list]

    def diagonalize(self, evals_count: int = 10):
        return self.hilbertspace.eigensys(evals_count=evals_count)

    def get_subsystem_index(self, id_str: str) -> int:
        for i, sub in enumerate(self.hilbertspace.subsystem_list):
            if sub.id_str == id_str:
                return i
        raise ValueError(f"Subsystem '{id_str}' not found")

    def embed_gate(self, gate: Qobj, target_ids: List[str]) -> Qobj:
        subsystems = self.hilbertspace.subsystem_list

        # 1. Extract simulation-space dimensions
        dims = [getattr(sub, "truncated_dim", sub.hilbertdim()) for sub in subsystems]

        # 2. Get the indices of target subsystems
        indices = [self.get_subsystem_index(id_str) for id_str in target_ids]

        # 3. Verify the gate dimensions match
        if len(indices) == 1:
            d_target = dims[indices[0]]

            if gate.shape != (d_target, d_target):
                raise ValueError(f"Gate shape {gate.shape} does not match subsystem dim {d_target}")

        elif len(indices) == 2:
            d_i, d_j = dims[indices[0]], dims[indices[1]]
            expected_shape = (d_i * d_j, d_i * d_j)
            if gate.shape != expected_shape:
                raise ValueError(f"Gate shape {gate.shape} does not match dims ({d_i}, {d_j})")

        else:
            raise NotImplementedError("Only 1- or 2-subsystem gates supported.")

        # 4. Build operator list with identities
        ops = [identity(d) for d in dims]

        # 5. For 1 target
        if len(indices) == 1:
            ops[indices[0]] = gate
            return tensor(ops)

        # 6. For 2 targets — embed using reorder and Kronecker trick
        i, j = sorted(indices)
        # Tensor identities before, gate, and after
        left = tensor(ops[:i]) if i > 0 else 1
        right = tensor(ops[j + 1:]) if j + 1 < len(ops) else 1

        # Identity middle space if i+1 < j
        if j - i > 1:
            middle = tensor(ops[i + 1:j])
            embedded = tensor(left, tensor(identity(dims[i]), identity(dims[j])) + 0 * gate, middle, right)
            embedded = embedded + (tensor(left, gate, middle, right) - embedded)
        else:
            embedded = tensor(left, gate, right)

        return embedded

    def apply_gate(self, state: Qobj, gate: Qobj, target_ids: List[str]) -> Qobj:
        """Apply a gate to the given state"""
        U = self.embed_gate(gate, target_ids)
        return U * state

    def evolve_state(self, initial_state: Qobj, tlist: np.ndarray, solver="sesolve") -> Qobj:
        H = self.get_qutip_hamiltonian()
        if solver == "sesolve":
            from qutip import sesolve
            result = sesolve(H, initial_state, tlist)
        elif solver == "mesolve":
            from qutip import mesolve
            result = mesolve(H, initial_state, tlist)
        else:
            raise ValueError(f"Unknown solver: {solver}")
        return result

    def _register_native_gates(self):
        self.native_gates["X"] = NativeGate(
            name="X", arity=1,
            generator=lambda targets: self.embed_gate(NATIVE_GATES["X"](), targets)
        )
        self.native_gates["Y"] = NativeGate(
            name="Y", arity=1,
            generator=lambda targets: self.embed_gate(NATIVE_GATES["X"](), targets)
        )
        self.native_gates["SQiSWAP"] = NativeGate(
            name="SQiSWAP", arity=2,
            generator=lambda targets: self.embed_gate(NATIVE_GATES["SQISWAP_gate"](), targets)
        )

    def apply_named_gate(self, state: Qobj, gate_name: str, targets: List[str]) -> Qobj:
        if gate_name not in self.native_gates:
            raise ValueError(f"Unknown gate: {gate_name}")

        gate = self.native_gates[gate_name]
        U = gate.generator(targets)
        return U * state
