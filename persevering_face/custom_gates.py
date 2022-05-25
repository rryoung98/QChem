import numpy as np

import cirq

"""Define a custom general unitary gatee with a parameter."""
class RotationGate(cirq.Gate):
    def __init__(self, alpha,beta, gamma):
        super(RotationGate, self)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return np.array([
            [np.cos(self.alpha/2), -np.exp(self.gamma*1j)*np.sin(self.alpha/2)],
            [np.exp(self.beta*1j)*np.sin(self.alpha/2), np.exp(1j*(self.gamma+self.beta))*np.cos(self.alpha/2)]
        ])

    def _circuit_diagram_info_(self, args):
        return f"R({self.alpha,self.beta,self.gamma})"