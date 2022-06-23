import numpy as np
import cirq
import tensorflow_quantum as tfq


class Encoder():
  def __init__(self, x_test,x_train):
    self.x_test=x_test,
    self.x_train=x_train
    self.tolerance = 1e-10
    
  def _section_img(img, sections=3):
    section_angles = []
    for row in img:
      section = int(np.floor(len(row)/sections))
      prior = None
      angles = []
      for i in range(sections):
        if prior is None:
          prior = 0
        angle = np.mean(row[prior:prior+section])/np.pi
        angles.append(angle)
        prior += section
      section_angles.append(angles)
    return section_angles

  def section_angle_encoding(self):
    for img in self.x_test:
      angle_list = self._section_img(img)
      qubits = cirq.LineQubit.range(10)
      test_circuit = cirq.Circuit()
      for i, value in enumerate(angle_list):
        r_x,r_y,r_z,  = value
        test_circuit.append(cirq.X(qubits[i])**(r_x))
        test_circuit.append(cirq.Y(qubits[i])**(r_y))
        test_circuit.append(cirq.Z(qubits[i])**(r_z))
    for img in self.x_train:
      angle_list = self._section_img(img)
      qubits = cirq.LineQubit.range(10)
      train_circuit = cirq.Circuit()
      for i, value in enumerate(angle_list):
        r_x,r_y,r_z,  = value
        train_circuit.append(cirq.X(qubits[i])**(r_x))
        train_circuit.append(cirq.Y(qubits[i])**(r_y))
        train_circuit.append(cirq.Z(qubits[i])**(r_z))
    return test_circuit,train_circuit

  # Use Mottonen state preparation to get an amplitude embedding of the states
  # Adapted from https://pennylane.readthedocs.io/en/stable/_modules/pennylane/templates/state_preparations/mottonen.html#MottonenStatePreparation

  def gray_code(self,rank):
      """Generates the Gray code of given rank.

      Args:
          rank (int): rank of the Gray code (i.e. number of bits)
      """

      def gray_code_recurse(g, rank):
          k = len(g)
          if rank <= 0:
              return

          for i in range(k - 1, -1, -1):
              char = "1" + g[i]
              g.append(char)
          for i in range(k - 1, -1, -1):
              g[i] = "0" + g[i]

          gray_code_recurse(g, rank - 1)

      g = ["0", "1"]
      gray_code_recurse(g, rank - 1)

      return g


  def _matrix_M_entry(self, row, col):
      """Returns one entry for the matrix that maps alpha to theta.

      See Eq. (3) in `Möttönen et al. (2004) <https://arxiv.org/pdf/quant-ph/0407010.pdf>`_.

      Args:
          row (int): one-based row number
          col (int): one-based column number

      Returns:
          (float): transformation matrix entry at given row and column
      """
      # (col >> 1) ^ col is the Gray code of col
      b_and_g = row & ((col >> 1) ^ col)
      sum_of_ones = 0
      while b_and_g > 0:
          if b_and_g & 0b1:
              sum_of_ones += 1

          b_and_g = b_and_g >> 1

      return (-1) ** sum_of_ones


  def _compute_theta(self,alpha):
      """Maps the angles alpha of the multi-controlled rotations decomposition of a uniformly controlled rotation
      to the rotation angles used in the Gray code implementation.

      Args:
          alpha (tensor_like): alpha parameters

      Returns:
          (tensor_like): rotation angles theta
      """
      ln = alpha.shape[0]
      k = np.log2(alpha.shape[0])

      M_trans = np.zeros(shape=(ln, ln))
      for i in range(len(M_trans)):
          for j in range(len(M_trans[0])):
              M_trans[i, j] = self._matrix_M_entry(j, i)

      theta = np.dot(M_trans, alpha)

      return theta / 2 ** k


  def _uniform_rotation_dagger(self,gate, alpha, control_wires, target_wire, circuit):
      r"""Applies a uniformly-controlled rotation to the target qubit.

      A uniformly-controlled rotation is a sequence of multi-controlled
      rotations, each of which is conditioned on the control qubits being in a different state.
      For example, a uniformly-controlled rotation with two control qubits describes a sequence of
      four multi-controlled rotations, each applying the rotation only if the control qubits
      are in states :math:`|00\rangle`, :math:`|01\rangle`, :math:`|10\rangle`, and :math:`|11\rangle`, respectively.

      To implement a uniformly-controlled rotation using single qubit rotations and CNOT gates,
      a decomposition based on Gray codes is used. For this purpose, the multi-controlled rotation
      angles alpha have to be converted into a set of non-controlled rotation angles theta.

      For more details, see `Möttönen and Vartiainen (2005), Fig 7a<https://arxiv.org/pdf/quant-ph/0504100.pdf>`_.

      Args:
          gate (.Operation): gate to be applied, needs to have exactly one parameter
          alpha (tensor_like): angles to decompose the uniformly-controlled rotation into multi-controlled rotations
          control_wires (array[int]): wires that act as control
          target_wire (int): wire that acts as target
      """

      theta = self._compute_theta(alpha)

      gray_code_rank = len(control_wires)

      if gray_code_rank == 0:
          if theta[0] != 0.0:
              circuit.append(gate(theta[0])(target_wire))
          return

      code = self.gray_code(gray_code_rank)
      num_selections = len(code)

      control_indices = [
          int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % num_selections], 2)))
          for i in range(num_selections)
      ]

      for i, control_index in enumerate(control_indices):
          if theta[i] != 0.0:
              circuit.append(gate(theta[i])(target_wire))
          circuit.append(cirq.CNOT(control_wires[control_index], target_wire))


  def _get_alpha_z(omega, n, k):
      r"""Computes the rotation angles required to implement the uniformly-controlled Z rotation
      applied to the :math:`k`th qubit.

      The :math:`j`th angle is related to the phases omega of the desired amplitudes via:

      .. math:: \alpha^{z,k}_j = \sum_{l=1}^{2^{k-1}} \frac{\omega_{(2j-1) 2^{k-1}+l} - \omega_{(2j-2) 2^{k-1}+l}}{2^{k-1}}

      Args:
          omega (tensor_like): phases of the state to prepare
          n (int): total number of qubits for the uniformly-controlled rotation
          k (int): index of current qubit

      Returns:
          array representing :math:`\alpha^{z,k}`
      """
      indices1 = [
          [(2 * j - 1) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)]
          for j in range(1, 2 ** (n - k) + 1)
      ]
      indices2 = [
          [(2 * j - 2) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)]
          for j in range(1, 2 ** (n - k) + 1)
      ]

      term1 = np.take(omega, indices=indices1)
      term2 = np.take(omega, indices=indices2)
      diff = (term1 - term2) / 2 ** (k - 1)

      return np.sum(diff, axis=1)


  def _get_alpha_y(a, n, k):
      r"""Computes the rotation angles required to implement the uniformly controlled Y rotation
      applied to the :math:`k`th qubit.

      The :math:`j`-th angle is related to the absolute values, a, of the desired amplitudes via:

      .. math:: \alpha^{y,k}_j = 2 \arcsin \sqrt{ \frac{ \sum_{l=1}^{2^{k-1}} a_{(2j-1)2^{k-1} +l}^2  }{ \sum_{l=1}^{2^{k}} a_{(j-1)2^{k} +l}^2  } }

      Args:
          a (tensor_like): absolute values of the state to prepare
          n (int): total number of qubits for the uniformly-controlled rotation
          k (int): index of current qubit

      Returns:
          array representing :math:`\alpha^{y,k}`
      """
      indices_numerator = [
          [(2 * (j + 1) - 1) * 2 ** (k - 1) + l for l in range(2 ** (k - 1))]
          for j in range(2 ** (n - k))
      ]
      numerator = np.take(a, indices=indices_numerator)
      numerator = np.sum(np.abs(numerator) ** 2, axis=1)

      indices_denominator = [[j * 2 ** k + l for l in range(2 ** k)] for j in range(2 ** (n - k))]
      denominator = np.take(a, indices=indices_denominator)
      denominator = np.sum(np.abs(denominator) ** 2, axis=1)

      # Divide only where denominator is zero, else leave initial value of zero.
      # The equation guarantees that the numerator is also zero in the corresponding entries.

      with np.errstate(divide="ignore", invalid="ignore"):
          division = numerator / denominator

      division = np.where(denominator != 0.0, division, 0.0)

      return 2 * np.arcsin(np.sqrt(division))

  def mottonenStatePrep(self,state_vector, qubits, circuit):

          a = np.abs(state_vector)
          omega = np.angle(state_vector)

          # change ordering of wires, since original code
          # was written for IBM machines
          wires_reverse = qubits[::-1]

          # Apply inverse y rotation cascade to prepare correct absolute values of amplitudes
          for k in range(len(wires_reverse), 0, -1):
              alpha_y_k = self._get_alpha_y(a, len(wires_reverse), k)
              control = wires_reverse[k:]
              target = wires_reverse[k - 1]
              self._uniform_rotation_dagger(cirq.ry, alpha_y_k, control, target, circuit)

          # If necessary, apply inverse z rotation cascade to prepare correct phases of amplitudes
          if not np.allclose(omega, 0):
              for k in range(len(wires_reverse), 0, -1):
                  alpha_z_k = self._get_alpha_z(omega, len(wires_reverse), k)
                  control = wires_reverse[k:]
                  target = wires_reverse[k - 1]
                  if len(alpha_z_k) > 0:
                      self._uniform_rotation_dagger(cirq.rz, alpha_z_k, control, target, circuit)

          
  # The below is borrowed from https://pennylane.readthedocs.io/en/stable/_modules/pennylane/templates/embeddings/amplitude.html#AmplitudeEmbedding
  def _preprocess(self, features, wires, pad_with, normalize):
      """Validate and pre-process inputs as follows:

      * Check that the features tensor is one-dimensional.
      * If pad_with is None, check that the first dimension of the features tensor
        has length :math:`2^n` where :math:`n` is the number of qubits. Else check that the
        first dimension of the features tensor is not larger than :math:`2^n` and pad features with value if necessary.
      * If normalize is false, check that first dimension of features is normalised to one. Else, normalise the
        features tensor.
      """

      shape = features.shape

      # check shape
      if features.ndim != 1:
          raise ValueError(f"Features must be a one-dimensional tensor; got shape {shape}.")

      n_features = shape[0]
      if pad_with is None and n_features != 2 ** len(wires):
          raise ValueError(
              f"Features must be of length {2 ** len(wires)}; got length {n_features}. "
              f"Use the 'pad' argument for automated padding."
          )

      if pad_with is not None and n_features > 2 ** len(wires):
          raise ValueError(
              f"Features must be of length {2 ** len(wires)} or "
              f"smaller to be padded; got length {n_features}."
          )

      # pad
      if pad_with is not None and n_features < 2 ** len(wires):
          padding = [pad_with] * (2 ** len(wires) - n_features)
          features = np.concatenate([features, padding], axis=0)

      # normalize
      norm = np.sum(np.abs(features) ** 2)

      if not np.allclose(norm, 1.0, atol=self.tolerance):
          if normalize or pad_with:
              features = features / np.sqrt(norm)
          else:
              raise ValueError(
                  f"Features must be a vector of length 1.0; got length {norm}."
                  "Use 'normalize=True' to automatically normalize."
              )

      features = features.astype(np.complex128)
      return features

  def cirq_amplitude_embedding(self,features, qubits, circuit, pad_with=None, normalize=False):
      features = self._preprocess(features, qubits, pad_with, normalize)
      self.mottonenStatePrep(features, qubits, circuit)