import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from typing import Tuple, Dict

class QuantumMLModels:
    """Quantum Machine Learning models for BCI signal prediction"""
    
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.scaler = StandardScaler()
        self.pca = None  # Will be initialized adaptively
    
    def quantum_feature_map(self, x):
        """Enhanced quantum feature map with deeper encoding and entanglement"""
        n_features = min(len(x), self.n_qubits)
        
        # Layer 1: Initial encoding with Hadamards for superposition
        for i in range(n_features):
            qml.Hadamard(wires=i)
            qml.RY(x[i] * np.pi, wires=i)
        
        # Layer 2: Data re-uploading with Z rotations
        for i in range(n_features):
            qml.RZ(x[i] * np.pi, wires=i)
        
        # Layer 3: Circular entanglement for better expressivity
        for i in range(n_features):
            qml.CNOT(wires=[i, (i + 1) % n_features])
        
        # Layer 4: Second data re-uploading
        for i in range(n_features):
            qml.RY((x[i] ** 2) * np.pi / 2, wires=i)
    
    def quantum_kernel(self, x1, x2):
        """Quantum kernel with proper fidelity computation for high accuracy"""
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            self.quantum_feature_map(x1)
            qml.adjoint(self.quantum_feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))
        
        probs = kernel_circuit(x1, x2)
        # Quantum fidelity kernel - probability of measuring |0...0⟩ state
        # This measures overlap between quantum states encoded from x1 and x2
        return float(probs[0])
    
    def quantum_kernel_matrix(self, X1, X2):
        """Compute quantum kernel matrix with optimized caching"""
        n1, n2 = len(X1), len(X2)
        kernel_matrix = np.zeros((n1, n2))
        
        # Use symmetry when X1 == X2 (training kernel)
        is_symmetric = (X1 is X2) or (len(X1) == len(X2) and np.allclose(X1, X2))
        
        for i in range(n1):
            for j in range(n2):
                if is_symmetric and j < i:
                    # Use symmetry: K[i,j] = K[j,i]
                    kernel_matrix[i, j] = kernel_matrix[j, i]
                else:
                    kernel_matrix[i, j] = self.quantum_kernel(X1[i], X2[j])
        
        return kernel_matrix
    
    def train_qsvm(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[SVC, float, Dict]:
        """Train Quantum Support Vector Machine with improved preprocessing"""
        start_time = time.time()
        
        # Preprocess features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize PCA with 8 components for 8-qubit circuit
        n_components = min(self.n_qubits, X_scaled.shape[0], X_scaled.shape[1])
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Pad features to match n_qubits if needed
        if X_reduced.shape[1] < self.n_qubits:
            padding = np.zeros((X_reduced.shape[0], self.n_qubits - X_reduced.shape[1]))
            X_reduced = np.hstack([X_reduced, padding])
        
        # Compute quantum kernel matrix
        K_train = self.quantum_kernel_matrix(X_reduced, X_reduced)
        
        # Add small regularization to kernel matrix for numerical stability
        K_train = K_train + 1e-8 * np.eye(len(K_train))
        
        # Train SVM with optimized hyperparameters for higher accuracy
        qsvm = SVC(kernel='precomputed', C=100.0, class_weight='balanced')
        qsvm.fit(K_train, y_train)
        
        # Training accuracy
        train_score = qsvm.score(K_train, y_train)
        
        processing_time = time.time() - start_time
        
        metrics = {
            'training_accuracy': train_score,
            'processing_time': processing_time,
            'n_support_vectors': len(qsvm.support_),
            'n_qubits': self.n_qubits
        }
        
        return qsvm, train_score, metrics
    
    def predict_qsvm(self, qsvm: SVC, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict using QSVM with optimized preprocessing"""
        start_time = time.time()
        
        # Preprocess
        X_train_scaled = self.scaler.transform(X_train)
        if self.pca is not None:
            X_train_reduced = self.pca.transform(X_train_scaled)
        else:
            X_train_reduced = X_train_scaled
        
        X_test_scaled = self.scaler.transform(X_test)
        if self.pca is not None:
            X_test_reduced = self.pca.transform(X_test_scaled)
        else:
            X_test_reduced = X_test_scaled
        
        # Pad features to match n_qubits if needed
        if X_train_reduced.shape[1] < self.n_qubits:
            padding_train = np.zeros((X_train_reduced.shape[0], self.n_qubits - X_train_reduced.shape[1]))
            X_train_reduced = np.hstack([X_train_reduced, padding_train])
        
        if X_test_reduced.shape[1] < self.n_qubits:
            padding_test = np.zeros((X_test_reduced.shape[0], self.n_qubits - X_test_reduced.shape[1]))
            X_test_reduced = np.hstack([X_test_reduced, padding_test])
        
        # Compute kernel between test and train
        K_test = self.quantum_kernel_matrix(X_test_reduced, X_train_reduced)
        
        # Predict
        predictions = qsvm.predict(K_test)
        
        processing_time = time.time() - start_time
        
        return predictions, processing_time
    
    def variational_quantum_classifier(self, n_layers: int = 2):
        """Create a Variational Quantum Classifier circuit"""
        
        @qml.qnode(self.dev)
        def vqc_circuit(weights, x):
            # Encoding
            self.quantum_feature_map(x)
            
            # Variational layers
            for layer in range(n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return qml.expval(qml.PauliZ(0))
        
        return vqc_circuit
    
    def train_vqc(self, X_train: np.ndarray, y_train: np.ndarray, 
                  n_layers: int = 2, epochs: int = 50) -> Tuple[np.ndarray, float, Dict]:
        """Train Variational Quantum Classifier"""
        start_time = time.time()
        
        # Preprocess
        X_scaled = self.scaler.fit_transform(X_train)
        # Initialize PCA adaptively if not already set
        if self.pca is None:
            n_components = min(self.n_qubits, X_scaled.shape[0], X_scaled.shape[1])
            self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Convert labels to -1, 1
        y_train_quantum = 2 * y_train - 1
        
        # Initialize weights
        weights = np.random.randn(n_layers, self.n_qubits, 2)
        weights = pnp.array(weights, requires_grad=True)
        
        # Create circuit
        circuit = self.variational_quantum_classifier(n_layers)
        
        # Optimizer
        opt = qml.GradientDescentOptimizer(stepsize=0.1)
        
        # Cost function
        def cost(weights, X, y):
            predictions = [circuit(weights, x) for x in X]
            return np.mean((predictions - y) ** 2)
        
        # Training loop
        for epoch in range(epochs):
            weights = opt.step(lambda w: cost(w, X_reduced, y_train_quantum), weights)
        
        # Final accuracy
        predictions = [circuit(weights, x) for x in X_reduced]
        predictions_binary = [1 if p > 0 else 0 for p in predictions]
        train_accuracy = np.mean(np.array(predictions_binary) == y_train)
        
        processing_time = time.time() - start_time
        
        metrics = {
            'training_accuracy': train_accuracy,
            'processing_time': processing_time,
            'n_layers': n_layers,
            'epochs': epochs,
            'n_qubits': self.n_qubits
        }
        
        return weights, train_accuracy, metrics
    
    def predict_vqc(self, weights: np.ndarray, X_test: np.ndarray, n_layers: int = 2) -> Tuple[np.ndarray, float]:
        """Predict using VQC with optimized preprocessing"""
        start_time = time.time()
        
        # Preprocess
        X_scaled = self.scaler.transform(X_test)
        if self.pca is not None:
            X_reduced = self.pca.transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        # Create circuit
        circuit = self.variational_quantum_classifier(n_layers)
        
        # Predict
        predictions = [circuit(weights, x) for x in X_reduced]
        predictions_binary = np.array([1 if p > 0 else 0 for p in predictions])
        
        processing_time = time.time() - start_time
        
        return predictions_binary, processing_time
    
    def extract_quantum_features(self, X: np.ndarray) -> np.ndarray:
        """Extract quantum features from classical data"""
        
        @qml.qnode(self.dev)
        def feature_circuit(x):
            self.quantum_feature_map(x)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        # Preprocess
        X_scaled = self.scaler.fit_transform(X)
        # Initialize PCA adaptively if not already set
        if self.pca is None:
            n_components = min(self.n_qubits, X_scaled.shape[0], X_scaled.shape[1])
            self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Extract features
        quantum_features = []
        for x in X_reduced:
            features = feature_circuit(x)
            quantum_features.append(features)
        
        return np.array(quantum_features)
