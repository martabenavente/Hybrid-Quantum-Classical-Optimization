import torch
from qubosolver import Instance, Solver, SolverConfig, QuantumSolvingConfig


# define QUBO
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
instance = Instance(matrix=Q)

# Create a SolverConfig object to use a quantum backend
config = SolverConfig(solving=QuantumSolvingConfig())

# Instantiate the quantum solver.
solver = Solver(instance, config)

# Solve the QUBO problem.
solution = solver.solve()
print(solution)

# Returns the following
# QUBOSolution(bitstrings=tensor([[0, 0]]), costs=tensor([0.]), counts=None, probabilities=None, solution_status=)