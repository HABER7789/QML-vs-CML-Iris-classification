import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Map labels to binary (0 or 1)
y_bin = np.where(y == 0, 0, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

# Quantum Circuit
def quantum_circuit(params, x):
    circuit = QuantumCircuit(1, 1)
    circuit.rx(params[0], 0)
    circuit.ry(params[1], 0)
    circuit.measure(0, 0)
    
    simulator = Aer.get_backend('aer_simulator')
    compiled_circuit = transpile(circuit, simulator)
    
    result = simulator.run(compiled_circuit).result()
    counts = result.get_counts()
    
    # Return probability of the '1' state
    return counts['1'] / 1024.0

# Objective function for quantum circuit optimization
def objective_function(params):
    predictions = [quantum_circuit(params, x) for x in X_train]
    loss = np.mean((predictions - y_train) ** 2)
    return loss

# Optimize the Quantum Circuit using classical optimization
result = minimize(objective_function, np.random.rand(2), method='COBYLA')
optimal_params = result.x

# Apply the optimized parameters to the quantum circuit for predictions on the test set
test_predictions_quantum = [quantum_circuit(optimal_params, x) for x in X_test]
binary_predictions_quantum = np.where(np.array(test_predictions_quantum) > 0.5, 1, 0)

# Calculate accuracy for Quantum ML
accuracy_quantum = accuracy_score(binary_predictions_quantum, y_test)
print(f"Quantum ML Accuracy: {accuracy_quantum * 100:.2f}%")

# Classical Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
test_predictions_classical = model.predict(X_test)

# Calculate accuracy for Classical ML
accuracy_classical = accuracy_score(test_predictions_classical, y_test)
print(f"Classical ML Accuracy: {accuracy_classical * 100:.2f}%")