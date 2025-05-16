#!/usr/bin/env python3
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit import transpile

import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.visualization import plot_state_qsphere
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
from ipywidgets import interact


# Computational basis states |0> and |1> (column vectors)

ket0 = np.array([[1], [0]], dtype=complex)
ket1 = np.array([[0], [1]], dtype=complex)



# Pauli-X gate: flips |0> to |1> and vice versa
X = np.array([[0, 1],
              [1, 0]], dtype=complex)

# Hadamard gate: creates superposition
H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                 [1, -1]], dtype=complex)

# Identity gate: does nothing
I = np.eye(2, dtype=complex)

# 2-qubit CNOT gate: flips target qubit if control is |1>
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)


def tensor(*matrices):
    """Compute the tensor (Kronecker) product of multiple matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


# Oracle for constant function f(x) = 0: returns input unchanged
def oracle_constant_zero():
    return np.eye(4, dtype=complex)

# Oracle for balanced function f(x) = x: applies CNOT
def oracle_balanced():
    return CNOT

# Deutsch-Jozsa circuit 
def deutsch_joza(oracle):
    # Start in |0>|1>
    state = tensor(ket0, ket1)
    print("Initial |0>|1> state:\n", state)

    # Apply Hadamard to both qubits: create superposition
    H2 = tensor(H, H)
    state = H2 @ state
    print("After Hadamards on both qubits:\n", state)

    # Apply oracle gate
    state = oracle @ state
    print("After Oracle application:\n", state)

    # Apply Hadamard to first qubit only (to extract interference)
    H1 = tensor(H, I)
    state = H1 @ state
    print("After Hadamard on input qubit:\n", state)
    

      # Measurement: Check how likely first qubit is in state |0>
    # Index 0 and 1 → states |00> and |01>: first qubit is |0>
    prob_zero = np.sum(np.abs(state[:2])**2)
    print(f"Probability of measuring input qubit as |0>: {prob_zero:.4f}")

    return prob_zero

# run an test both oracles
p_const = deutsch_joza(oracle_constant_zero())
p_balanced = deutsch_joza(oracle_balanced())

print(f"Probability first qubit is |0> for constant f: {p_const:.4f}")
print(f"Probability first qubit is |0> for balanced f: {p_balanced:.4f}")

###################################
#  Qiskit implementation
####################################

# Oracle for constant function f(x)=0: do nothing
def constant_oracle():
    oracle = QuantumCircuit(2, name="Constant Oracle")
    return oracle

# Oracle for balanced function f(x)=x: flip second qubit if first is 1
def balanced_oracle():
    oracle = QuantumCircuit(2, name="Balanced Oracle")
    oracle.cx(0, 1)  # CNOT gate
    return oracle


# Step 1: create a 2-qubit circuit: input (qubit 0), output (qubit 1)
## creating a pond with 2 ripple points
qc = QuantumCircuit(2, 1)  # 2 qubits, 1 classical bit
# Input qubit -> where you drop your pebble (causes ripple), output qubit -> underwater ripple mirror (alters waves)




# Step 2: Prepare the input and output qubits
## drop your pebbles
## ripple mirror output qubit
qc.x(1)         # Flip output qubit to |1>

# dropping two pebbles, both at once
# qc.h(0) and qc.h(1) drop a ripple in every direction -> superposition: explore both inputs

qc.h(0)         # Put input qubit in superposition
qc.h(1)         # Put output qubit in superposition



# Step 3: Oracle – f(x) applied as a quantum gate
#	Insert ripple mirror (oracle gate)	f(x) flips some ripples
oracle = constant_oracle()  # or balanced_oracle()
qc.compose(oracle, inplace=True)

# Step 4: Interference
# qc.h(0) again, combine all ripples at centre	-> interference tells you pattern of f(x)
qc.h(0)         # Hadamard again on input qubit ->  extract global pattern (balanced or constant)
#If the ripples reinforce (constructive), you’ll get a big splash at the centre -> qubit becomes |0>.
# If the ripples cancel out (destructive), it’s still pond -> qubit becomes |1>



# Step 5: Measure input qubit
# qc.measure(0, 0)	1st '0'qubit index you want to measure (in the quantum register), 2nd '0' classical bit index where the result gets stored
# Look at the centre of the pond	-> Splash = constant (0), still pond = balanced (1) 
qc.measure(0, 0)# "measure qubit 0 and put the answer in classical bit 0"

# run the circuit simulation using Aer
simulator = Aer.get_backend('aer_simulator')
result = simulator.run(qc, shots=1024).result()
counts = result.get_counts()
print("Qiskit Deutsch-Jozsa results:", counts)



### simplified functions for vis
def build_dj_circuit(oracle_func, label=""):
    qc = QuantumCircuit(2, 1)
    qc.x(1)
    qc.h(0)
    qc.h(1)
    oracle1 = oracle_func()
    qc.compose(oracle1, inplace=True)
    qc.h(0)
    qc.measure(0, 0)
    return qc

def custom_balanced_oracle():
    oracle = QuantumCircuit(2, name="balanced Oracle")
    oracle.x(0)
    oracle.cx(0, 1)
    oracle.x(0)
    return oracle
# plot with transpiled, plot histogram
def plot_circuit(qc, title=""):
    fig = qc.draw(output='mpl')
    plt.title(title)
    plt.show()
    fig.savefig(f"{title}.png")


def run_and_plot(qc, title=""):
    sim = Aer.get_backend('aer_simulator')
    qc = transpile(qc, sim)
    result = sim.run(qc, shots=1024).result()
    counts = result.get_counts()
    print(f"\nResults for {title}:\n", counts)
    plot_histogram(counts, title=title)
    plt.show()


def show_histogram(shots=1024):
    simulator = Aer.get_backend('aer_simulator')
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    plot_histogram(counts)
    plt.show()

interact(show_histogram, shots=(100, 5000, 100))


qc_no_measure = qc.remove_final_measurements(inplace=False)
state = Statevector.from_instruction(qc_no_measure)
plot_bloch_multivector(state)


qc_const = build_dj_circuit(constant_oracle, "Constant")
qc_bal = build_dj_circuit(balanced_oracle, "Balanced")

run_and_plot(qc_const, title="Deutsch-Jozsa (constant oracle)")
run_and_plot(qc_bal, title="Deutsch-Jozsa (balanced oracle)")


qc_custom = build_dj_circuit(custom_balanced_oracle, "kustom balanced")
run_and_plot(qc_custom, "Deutsch-Jozsa (kustom balanced oracle)")