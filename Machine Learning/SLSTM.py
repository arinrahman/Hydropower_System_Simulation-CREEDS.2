import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initWeights(rows, cols):
    return np.random.randn(rows, cols) * 0.01  # Small random initialization

class SLSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weight matrices and biases
        self.wf_x = initWeights(hidden_size, input_size)
        self.wf_h = initWeights(hidden_size, hidden_size)
        self.wi_x = initWeights(hidden_size, input_size)
        self.wi_h = initWeights(hidden_size, hidden_size)
        self.wo_x = initWeights(hidden_size, input_size)
        self.wo_h = initWeights(hidden_size, hidden_size)
        self.wc_x = initWeights(hidden_size, input_size)
        self.wc_h = initWeights(hidden_size, hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

        self.wy = initWeights(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

        self.reset()

    def reset(self):
        self.hidden_states = {-1: np.zeros((self.hidden_size, 1))}
        self.cell_states = {-1: np.zeros((self.hidden_size, 1))}
        self.forget_gates = {}
        self.input_gates = {}
        self.candidate_gates = {}
        self.output_gates = {}

    def forward(self, X):
        T = X.shape[1]  # Time steps
        self.outputs = {}

        for t in range(T):
            x_t = X[:, t].reshape(-1, 1)

            # Ensure dictionary entries exist for t
            if t not in self.hidden_states:
                self.hidden_states[t] = np.zeros((self.hidden_size, 1))
                self.cell_states[t] = np.zeros((self.hidden_size, 1))

            prev_h = self.hidden_states[t - 1]
            prev_c = self.cell_states[t - 1]

            # Compute gate activations
            self.forget_gates[t] = sigmoid(np.dot(self.wf_x, x_t) + np.dot(self.wf_h, prev_h) + self.bf)
            self.input_gates[t] = sigmoid(np.dot(self.wi_x, x_t) + np.dot(self.wi_h, prev_h) + self.bi)
            self.candidate_gates[t] = np.tanh(np.dot(self.wc_x, x_t) + np.dot(self.wc_h, prev_h) + self.bc)
            self.output_gates[t] = sigmoid(np.dot(self.wo_x, x_t) + np.dot(self.wo_h, prev_h) + self.bo)

            # Update cell state and hidden state
            self.cell_states[t] = self.forget_gates[t] * prev_c + self.input_gates[t] * self.candidate_gates[t]
            self.hidden_states[t] = self.output_gates[t] * np.tanh(self.cell_states[t])

            # Compute output
            self.outputs[t] = np.dot(self.wy, self.hidden_states[t]) + self.by

        return self.outputs

    def backward(self, X, Y, outputs):
        T = X.shape[1]
        dWy = np.zeros_like(self.wy)
        dBy = np.zeros_like(self.by)

        dH_next = np.zeros((self.hidden_size, 1))
        dC_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            x_t = X[:, t].reshape(-1, 1)
            y_t = Y[:, t].reshape(-1, 1)
            output = outputs[t]

            if t not in self.hidden_states:
                self.hidden_states[t] = np.zeros((self.hidden_size, 1))

            # Compute output gradients
            dY = output - y_t
            dWy += np.dot(dY, self.hidden_states[t].T)
            dBy += dY

            # Compute hidden state error
            dH = np.dot(self.wy.T, dY) + dH_next
            dO = dH * np.tanh(self.cell_states[t]) * sigmoid_derivative(self.output_gates[t])
            dC = dH * self.output_gates[t] * (1 - np.tanh(self.cell_states[t]) ** 2) + dC_next
            dF = dC * self.cell_states[t - 1] * sigmoid_derivative(self.forget_gates[t])
            dI = dC * self.candidate_gates[t] * sigmoid_derivative(self.input_gates[t])
            dC_bar = dC * self.input_gates[t] * (1 - self.candidate_gates[t] ** 2)

            # Compute weight and bias gradients
            dWf_x = np.dot(dF, x_t.T)
            dWf_h = np.dot(dF, self.hidden_states[t - 1].T)
            dbf = dF

            dWi_x = np.dot(dI, x_t.T)
            dWi_h = np.dot(dI, self.hidden_states[t - 1].T)
            dbi = dI

            dWo_x = np.dot(dO, x_t.T)
            dWo_h = np.dot(dO, self.hidden_states[t - 1].T)
            dbo = dO

            dWc_x = np.dot(dC_bar, x_t.T)
            dWc_h = np.dot(dC_bar, self.hidden_states[t - 1].T)
            dbc = dC_bar

            # Update next state errors
            dH_next = np.dot(self.wf_h.T, dF) + np.dot(self.wi_h.T, dI) + np.dot(self.wo_h.T, dO) + np.dot(self.wc_h.T, dC_bar)
            dC_next = dC * self.forget_gates[t]

        return dWy, dBy  # Add other weight updates as needed


    
# Define model parameters
input_size = 3
hidden_size = 5
output_size = 1

# Instantiate the LSTM model (without num_epochs and learning_rate)
lstm = SLSTM(input_size, hidden_size, output_size)

# Generate dummy data (time series of length T)
T = 10
X_train = np.random.randn(input_size, T)  # Shape: (input_size, T)
Y_train = np.random.randn(output_size, T) # Shape: (output_size, T)

# Run forward and backward passes
outputs = lstm.forward(X_train)
lstm.backward(X_train, Y_train, outputs)

# Generate dummy test data
X_test = np.random.randn(input_size, T)  # New time series data

# Get predictions
predictions = lstm.forward(X_test)

# Print predictions
print("Predicted Outputs:")
print(predictions)