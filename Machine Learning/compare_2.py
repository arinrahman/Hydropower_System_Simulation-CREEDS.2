import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Ingestor import Ingestor

# —— Load saved arrays —— #
y_test_numpy = np.load('y_test.npy')
numpy_preds  = np.load('numpy_lstm_predictions.npy')
keras_lstm   = np.load('keras_lstm_preds.npy')
keras_rnn    = np.load('keras_rnn_preds.npy')
keras_bidir  = np.load('keras_bidir_preds.npy')
keras_dense  = np.load('keras_dense_preds.npy')

# —— Reconstruct index —— #
df = pd.DataFrame(Ingestor('DataSetExport-Discharge Total.Last-24-Hour-Change-in-Storage@08450800-Instantaneous-TCM-20240622194957.csv').data)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

seq_length = 10
train_size = int(len(df) * 0.8)
timestamps = df.index[train_size + seq_length:]

# —— Align lengths —— #
all_arrays = [
    y_test_numpy.flatten(), numpy_preds.flatten(),
    keras_lstm.flatten(), keras_rnn.flatten(),
    keras_bidir.flatten(), keras_dense.flatten()
]
common_len = min(map(len, all_arrays))
idx = timestamps[-common_len:]

actual = y_test_numpy.flatten()[-common_len:]
predictions = {
    'NumPy LSTM': numpy_preds.flatten()[-common_len:],
    'Keras LSTM': keras_lstm.flatten()[-common_len:],
    'Keras RNN': keras_rnn.flatten()[-common_len:],
    'Keras Bidirectional': keras_bidir.flatten()[-common_len:],
    'Keras Dense': keras_dense.flatten()[-common_len:]
}

# —— Compute & print metrics —— #
print("Model Performance Metrics:")
for name, preds in predictions.items():
    mse = mean_squared_error(actual, preds)
    mae = mean_absolute_error(actual, preds)
    print(f"{name:>18}: MSE = {mse:.4f}, MAE = {mae:.4f}")

df_plot = pd.DataFrame({'Actual': actual}, index=idx)
for name, preds in predictions.items():
    df_plot[name] = preds



plt.figure(figsize=(16, 8))
for col in df_plot.columns:
    plt.plot(df_plot.index, df_plot[col], label=col)
plt.title('Actual vs Predicted — All Models')
plt.xlabel('Timestamp')
plt.ylabel('Discharge Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

