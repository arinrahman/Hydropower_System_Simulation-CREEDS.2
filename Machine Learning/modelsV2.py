import tensorflow as tf
import matplotlib.pyplot as plt
from Ingestor import Ingestor
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load and preprocess data
amnistadRelease = 'DataSetExport-Discharge Total.Last-24-Hour-Change-in-Storage@08450800-Instantaneous-TCM-20240622194957.csv'
data = Ingestor(amnistadRelease).data
df = pd.DataFrame(data)

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# Ensure 'Timestamp' column is set as index
df.set_index('Timestamp', inplace=True)

# Extract time-based features if index is DatetimeIndex
if isinstance(df.index, pd.DatetimeIndex):
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute
else:
    raise TypeError("Index is not a DatetimeIndex. Please check the conversion of the 'Timestamp' column.")

# Create a custom time-based feature To convert 
df['CustomTimeFeature'] = (
    df['Year'] * 10000 +         # Year * 10000 (e.g., 2024 -> 20240000)
    df['Month'] * 100 +          # Month * 100 (e.g., 09 -> 900)
    df['Day'] +                  # Day (e.g., 21)
    df['Hour'] / 24 +            # Hour as a fraction of the day (e.g., 15 -> 0.625)
    df['Minute'] / 1440          # Minute as a fraction of the day (e.g., 30 -> 0.0208)
)

# Scale the custom time feature
scaler_time = MinMaxScaler()
scaled_time_features = scaler_time.fit_transform(df[['CustomTimeFeature']])

# Scale numerical values
scaler_values = MinMaxScaler()
scaled_values = scaler_values.fit_transform(df[['Value']])

# Combine the scaled features
combined_features = np.hstack([scaled_time_features, scaled_values])

def predict_with_custom_value(timestamp, value, scaler, model, seq_length):
    timestamp = pd.to_datetime(timestamp)
    
    scaled_value = scaler.transform(np.array([[value]]))

    last_sequence = scaled_data[-(seq_length - 1):]

    custom_sequence = np.append(last_sequence, scaled_value).reshape((1, seq_length, 1))

    prediction = model.predict(custom_sequence)

    actual_prediction = scaler.inverse_transform(prediction)
    
    print(f"Prediction for the next time step after {timestamp} with input value {value}: {actual_prediction[0][0]}")
    return actual_prediction[0][0]

def create_sequences_with_targets(data, seq_length):
    sequences = []
    targets = []
    
    # Loop through the data to create sequences
    for i in range(seq_length, len(data)):
        
        # Extract the current timestamp and features
        timestamp = data[i][0]

        # The target is the value at the current timestamp
        target = data[i][1]

        # Extract the current and prior values (seq_length total)
        prior_values = [data[j] for j in range(i - seq_length, i)]
        
        # Create a sequence where data[0] = timestamp and the rest are prior values 
        sequence = [timestamp] + [item for sublist in prior_values for item in sublist]
        
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)
    
# Define sequence length and shift
seq_length = 1
shift = 1 

# Create X and Y
X,y = create_sequences_with_targets(combined_features, seq_length)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]
print(X_train.size)
print(X_train)

# Reshape for LSTM, RNN, and Bidirectional RNN
num_features = X_train.size // (X_train.shape[0] * seq_length)
X_train = X_train.reshape((X_train.shape[0], (seq_length * 2) + 1, 1))
X_test = X_test.reshape((X_test.shape[0],(seq_length * 2)  + 1, 1))

# Define and train LSTM model
lstm_model = Sequential([
    Input(shape=((seq_length * 2) + 1 , 1)), 
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

optimizer = Adam(learning_rate=0.001)
lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

lstm_model.fit(X_train, y_train, epochs=75, batch_size=1, validation_data=(X_test, y_test))

# Define and train RNN model
rnn_model = Sequential([
    Input(shape=((seq_length * 2) + 1 , 1)), 
    SimpleRNN(50, return_sequences=True),
    Dropout(0.2),
    SimpleRNN(50),
    Dropout(0.2),
    Dense(1)
])

rnn_optimizer = Adam(learning_rate=0.001)
rnn_model.compile(optimizer=rnn_optimizer, loss='mean_squared_error')

rnn_model.fit(X_train, y_train, epochs=75, batch_size=1, validation_data=(X_test, y_test))

# Define and train Bidirectional RNN model
bidirectional_rnn_model = Sequential([
    Input(shape=((seq_length * 2) + 1 , 1)),  
    Bidirectional(SimpleRNN(50, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(SimpleRNN(50)),
    Dropout(0.2),
    Dense(1)
])

bidirectional_rnn_optimizer = Adam(learning_rate=0.001)
bidirectional_rnn_model.compile(optimizer=bidirectional_rnn_optimizer, loss='mean_squared_error')

bidirectional_rnn_model.fit(X_train, y_train, epochs=75, batch_size=1, validation_data=(X_test, y_test))

# Flatten X_train and X_test for Dense model
X_train_dense = X_train.reshape((X_train.shape[0], -1))
X_test_dense = X_test.reshape((X_test.shape[0], -1))

dense_model = Sequential([
    Input(shape=(X_train_dense.shape[1], )), 
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

dense_optimizer = Adam(learning_rate=0.001)
dense_model.compile(optimizer=dense_optimizer, loss='mean_squared_error')

dense_model.fit(X_train_dense, y_train, epochs=75, batch_size=1, validation_data=(X_test_dense, y_test))

# Make predictions for the test set with LSTM model
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = lstm_predictions.reshape(-1, 1)
lstm_predictions = scaler_values.inverse_transform(lstm_predictions)

# Make predictions for the test set with RNN model
rnn_predictions = rnn_model.predict(X_test)
rnn_predictions = rnn_predictions.reshape(-1, 1)
rnn_predictions = scaler_values.inverse_transform(rnn_predictions)

# Make predictions for the test set with Bidirectional RNN model
bidirectional_rnn_predictions = bidirectional_rnn_model.predict(X_test)
bidirectional_rnn_predictions = bidirectional_rnn_predictions.reshape(-1, 1)
bidirectional_rnn_predictions = scaler_values.inverse_transform(bidirectional_rnn_predictions)

# Make predictions for the test set with Dense model
dense_predictions = dense_model.predict(X_test_dense)
dense_predictions = dense_predictions.reshape(-1, 1)
dense_predictions = scaler_values.inverse_transform(dense_predictions)

# Inverse transform y_test back to original values
#y_test = scaler_values.inverse_transform(y_test.reshape(-1, 1))
y_test = scaler_values.inverse_transform(y_test.reshape(-1,1))
print(f"y_test shape: {y_test.shape}")
print(f"LSTM predictions shape: {lstm_predictions.shape}")
print(f"RNN predictions shape: {rnn_predictions.shape}")
print(f"Bidirectional RNN predictions shape: {bidirectional_rnn_predictions.shape}")
print(f"Dense predictions shape: {dense_predictions.shape}")
# Calculate and print metrics for LSTM model
lstm_mse = mean_squared_error(y_test, lstm_predictions)
print(f'LSTM Mean Squared Error: {lstm_mse}')

lstm_mae = mean_absolute_error(y_test, lstm_predictions)
print(f'LSTM Mean Absolute Error: {lstm_mae}')

# Calculate and print metrics for RNN model
rnn_mse = mean_squared_error(y_test, rnn_predictions)
print(f'RNN Mean Squared Error: {rnn_mse}')

rnn_mae = mean_absolute_error(y_test, rnn_predictions)
print(f'RNN Mean Absolute Error: {rnn_mae}')

# Calculate and print metrics for Bidirectional RNN model
bidirectional_rnn_mse = mean_squared_error(y_test, bidirectional_rnn_predictions)
print(f'Bidirectional RNN Mean Squared Error: {bidirectional_rnn_mse}')

bidirectional_rnn_mae = mean_absolute_error(y_test, bidirectional_rnn_predictions)
print(f'Bidirectional RNN Mean Absolute Error: {bidirectional_rnn_mae}')

# Calculate and print metrics for Dense model
dense_mse = mean_squared_error(y_test, dense_predictions)
print(f'Dense Mean Squared Error: {dense_mse}')

dense_mae = mean_absolute_error(y_test, dense_predictions)
print(f'Dense Mean Absolute Error: {dense_mae}')

# Calculate average gap for all models
lstm_absolute_errors = np.abs(y_test - lstm_predictions)
lstm_average_gap = np.mean(lstm_absolute_errors)

rnn_absolute_errors = np.abs(y_test - rnn_predictions)
rnn_average_gap = np.mean(rnn_absolute_errors)

bidirectional_rnn_absolute_errors = np.abs(y_test - bidirectional_rnn_predictions)
bidirectional_rnn_average_gap = np.mean(bidirectional_rnn_absolute_errors)

dense_absolute_errors = np.abs(y_test - dense_predictions)
dense_average_gap = np.mean(dense_absolute_errors)

print(f'LSTM Average Gap: {lstm_average_gap}')
print(f'RNN Average Gap: {rnn_average_gap}')
print(f'Bidirectional RNN Average Gap: {bidirectional_rnn_average_gap}')
print(f'Dense Average Gap: {dense_average_gap}')

# Plot results
test_timestamps = df.index[train_size + (seq_length):]

y_test = y_test.reshape(-1)

plt.figure(figsize=(14, 7))

plt.plot(test_timestamps, y_test, label='Actual', color='black')
plt.plot(test_timestamps, lstm_predictions, label='LSTM Predicted', color='blue')
plt.plot(test_timestamps, rnn_predictions, label='RNN Predicted', color='orange')
plt.plot(test_timestamps, bidirectional_rnn_predictions, label='Bidirectional RNN Predicted', color='purple')
plt.plot(test_timestamps, dense_predictions, label='Dense Predicted', color='green')

plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Prediction vs Actual Values')
plt.legend()
plt.grid(True)
plt.show()

# Ensure residuals (errors) for each model are calculated as 1D arrays
lstm_errors = y_test - lstm_predictions.reshape(-1)
rnn_errors = y_test - rnn_predictions.reshape(-1)
bidirectional_rnn_errors = y_test - bidirectional_rnn_predictions.reshape(-1)
dense_errors = y_test - dense_predictions.reshape(-1)

# Check that test_timestamps matches the number of residuals
if len(test_timestamps) != len(lstm_errors):
    raise ValueError(f"Mismatch between number of timestamps ({len(test_timestamps)}) and errors ({len(lstm_errors)})")

# Plot the errors (residuals) over time
plt.figure(figsize=(14, 7))

plt.text(
    0.02, 0.95, 
    f'LSTM Average Gap: {lstm_average_gap:.2f}', 
    transform=plt.gca().transAxes, 
    fontsize=12,
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black')
)

plt.text(
    0.02, 0.90, 
    f'RNN Average Gap: {rnn_average_gap:.2f}', 
    transform=plt.gca().transAxes, 
    fontsize=12,
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black')
)

plt.text(
    0.02, 0.85, 
    f'Bidirectional RNN Average Gap: {bidirectional_rnn_average_gap:.2f}', 
    transform=plt.gca().transAxes, 
    fontsize=12,
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black')
)

plt.text(
    0.02, 0.80, 
    f'Dense Average Gap: {dense_average_gap:.2f}', 
    transform=plt.gca().transAxes, 
    fontsize=12,
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black')
)

plt.plot(test_timestamps, lstm_errors, label='LSTM Error', color='blue')
plt.plot(test_timestamps, rnn_errors, label='RNN Error', color='orange')
plt.plot(test_timestamps, bidirectional_rnn_errors, label='Bidirectional RNN Error', color='purple')
plt.plot(test_timestamps, dense_errors, label='Dense Error', color='green')

plt.axhline(0, color='black', linestyle='--')  # Add a horizontal line at x=0 for reference

plt.xlabel('Timestamp')
plt.ylabel('Error (Actual - Predicted)')
plt.title('Prediction Errors Over Time')
plt.legend()
plt.grid(True)
plt.show()
