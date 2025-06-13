# ---------------------- Imports ---------------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Ingestor import Ingestor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# ---------------------- Optimizer ---------------------- #
class AdamOptimizer:
    def __init__(self, lr=0.001, beta_1=0.97, beta_2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, param, grad, key):
        if key not in self.m:
            self.m[key] = np.zeros_like(grad)
            self.v[key] = np.zeros_like(grad)

        self.t += 1
        self.m[key] = self.beta_1 * self.m[key] + (1 - self.beta_1) * grad
        self.v[key] = self.beta_2 * self.v[key] + (1 - self.beta_2) * (grad ** 2)

        m_hat = self.m[key] / (1 - self.beta_1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta_2 ** self.t)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param

# ---------------------- Reproducibility ---------------------- #
np.random.seed(78) #5,40
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---------------------- Load and Preprocess Data ---------------------- #
amnistadRelease = 'DataSetExport-Discharge Total.Last-24-Hour-Change-in-Storage@08450800-Instantaneous-TCM-20240622194957.csv'
data = Ingestor(amnistadRelease).data

df = pd.DataFrame(data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df.set_index('Timestamp', inplace=True)
df['Value'] = pd.to_numeric(df['Value'], errors='coerce').ffill()

# Additional time-based features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['day'] = df.index.day

# Scaling features
scaler_value = MinMaxScaler()
scaler_time = MinMaxScaler()

features = df[['Value', 'hour', 'dayofweek', 'month', 'day']].copy()
scaled_values = scaler_value.fit_transform(features[['Value']])
scaled_time = scaler_time.fit_transform(features.drop(columns='Value'))

combined_scaled = np.hstack([scaled_values, scaled_time])

# ---------------------- Sequence Generation ---------------------- #
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(seq_length, len(data)):
        sequences.append(data[i - seq_length:i])
        targets.append(data[i, 0])
    return np.array(sequences), np.array(targets).reshape(-1, 1)

seq_length = 10
X, y = create_sequences(combined_scaled, seq_length)

train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# ---------------------- NumPy LSTM Model ---------------------- #
class NumPyLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001, loss_fn='rmse'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.loss_fn = loss_fn
        self.optimizer = AdamOptimizer(lr=learning_rate)

        def xavier_init(shape):
            return np.random.randn(*shape) * np.sqrt(1. / shape[1])

        self.W = {gate: xavier_init((hidden_dim, hidden_dim + input_dim + 1)) for gate in ['f', 'i', 'c', 'o']}
        self.b = {gate: np.zeros((hidden_dim, 1)) for gate in ['f', 'i', 'c', 'o']}

        self.W['y'] = xavier_init((output_dim, hidden_dim))
        self.b['y'] = np.zeros((output_dim, 1))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def tanh(x):
        return np.tanh(np.clip(x, -500, 500))

    def forward(self, x):
        T, _ = x.shape
        h_t = np.zeros((self.hidden_dim, 1))
        c_t = np.zeros((self.hidden_dim, 1))

        for t in range(T):
            x_t = x[t].reshape(-1, 1)
            combined = np.vstack((h_t, np.ones((1, 1)), x_t))
            f_t = self.sigmoid(np.dot(self.W['f'], combined) + self.b['f'])
            i_t = self.sigmoid(np.dot(self.W['i'], combined) + self.b['i'])
            o_t = self.sigmoid(np.dot(self.W['o'], combined) + self.b['o'])
            c_t_candidate = self.tanh(np.dot(self.W['c'], combined) + self.b['c'])
            c_t = f_t * c_t + i_t * c_t_candidate
            h_t = o_t * self.tanh(c_t)

        y_t = np.dot(self.W['y'], h_t) + self.b['y']
        return y_t.flatten(), h_t

    def compute_loss(self, pred, target):
        if self.loss_fn == 'mae':
            return np.mean(np.abs(pred - target))
        return np.mean((pred - target) ** 2)

    def train(self, X_train, y_train, epochs=50):
        print("\nTraining NumPy LSTM model with Adam Optimizer...")
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X_train)):
                x, y_true = X_train[i], y_train[i].reshape(-1, 1)
                y_pred, h_t = self.forward(x)
                error = y_pred.reshape(-1, 1) - y_true
                loss = self.compute_loss(y_pred, y_true)
                total_loss += loss

                dW_y = np.dot(error, h_t.T)
                db_y = error

                self.W['y'] = self.optimizer.update(self.W['y'], dW_y, 'W_y')
                self.b['y'] = self.optimizer.update(self.b['y'], db_y, 'b_y')

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X_train):.6f}")

# ---------------------- Train and Evaluate ---------------------- #
lstm_model = NumPyLSTM(input_dim=5, hidden_dim=64, output_dim=1, learning_rate=0.0005, loss_fn='rmse')
lstm_model.train(X_train, y_train, epochs=200)

# Prediction
lstm_predictions = [lstm_model.forward(x)[0] for x in X_test]
lstm_predictions = np.array(lstm_predictions).reshape(-1, 1)
lstm_predictions = scaler_value.inverse_transform(lstm_predictions)
y_test = scaler_value.inverse_transform(y_test)

# Save
np.save('numpy_lstm_predictions.npy', lstm_predictions)
np.save('y_test.npy', y_test)

# Evaluate
mse = mean_squared_error(y_test, lstm_predictions)
mae = mean_absolute_error(y_test, lstm_predictions)
print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df.index[train_size + seq_length:], y_test, label='Actual', color='black')
plt.plot(df.index[train_size + seq_length:], lstm_predictions, label='Predicted', color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Discharge Value')
plt.title('LSTM Prediction vs Actual Values')
plt.legend()
plt.grid(True)
plt.show()

