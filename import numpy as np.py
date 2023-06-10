import numpy as np
from scipy.linalg import expm

def trotterize(operator, time, num_steps):
    """
    Applies Trotterization to simulate the time evolution of a quantum system.
    
    Args:
        operator (np.ndarray): The Hamiltonian operator of the system.
        time (float): Total time of evolution.
        num_steps (int): Number of Trotter steps to perform.
        
    Returns:
        np.ndarray: The final state after time evolution.
    """
    dt = time / num_steps
    unitary = expm(-1j * operator * dt)
    state = np.eye(operator.shape[0], dtype=np.complex128)  # Initial state as identity matrix
    
    for _ in range(num_steps):
        state = unitary @ state  # Apply Trotter step
        
    return state

# Example usage
hamiltonian = np.array([[1, 0], [0, -1]])  # Example 2x2 Hamiltonian matrix
evolution_time = 1.0
num_steps = 100

final_state = trotterize(hamiltonian, evolution_time, num_steps)

print("Final state after time evolution:")
print(final_state)
