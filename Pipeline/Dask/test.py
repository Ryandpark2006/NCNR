#%conda install optuna dask dask-jobqueue

STORAGE = "sqlite:///dasktest.sqlite"
DEBUG_FILE = "debug_dask.txt"

import socket
import time
import random

from dask.distributed import Client
from dask.distributed import wait
import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader
import pandas as pd

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.metrics import smape
from darts.models import TCNModel
from darts.utils.likelihood_models import GaussianLikelihood
import dask.distributed as dd

# Define the self-supervised CPC model
class CPCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.2):
        super(CPCModel, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # The input sequence 'x' will have shape (batch_size, sequence_length, input_dim)
        # Encode the input sequence using the transformer encoder
        encoded = self.encoder(x.transpose(0, 1))[0].transpose(0, 1)  # (batch_size, seq_len, d_model)
        # Apply the transformer encoder
        transformed = self.transformer_encoder(encoded)
        # Project back to the input dimension using a linear layer
        reconstructed = self.linear(transformed)
        return reconstructed

# Train the CPC model
def train_cpc_model(model, data_loader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # Reconstruction loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")

# Use the trained model for anomaly detection
def detect_anomalies(model, data_loader, threshold):
    model.eval()
    anomalies = []

    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs)
            loss = torch.mean(torch.square(outputs - inputs), dim=1)  # Mean squared error
            for idx, error in enumerate(loss):
                if error > threshold:
                    anomalies.append((idx, error.item()))

    return anomalies

if __name__ == "__main__":
    # Hyperparameters
    input_dim = 128  # Input dimension of the time series data
    hidden_dim = 256  # Hidden dimension for the transformer model
    num_layers = 3  # Number of transformer layers
    num_heads = 4  # Number of attention heads
    num_epochs = 10
    learning_rate = 1e-4
    threshold = 0.1  # Anomaly detection threshold

    # Load and preprocess your unlabelled multivariate time series dataset
    # ...

    # Create a data loader for training
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the CPC model
    model = CPCModel(input_dim, hidden_dim, num_layers, num_heads)

    # Train the CPC model
    train_cpc_model(model, data_loader, num_epochs, learning_rate)

    # Use the trained model for anomaly detection
    anomalies = detect_anomalies(model, data_loader, threshold)

    print("Detected Anomalies:")
    for idx, error in anomalies:
        print(f"Time Step: {idx}, Reconstruction Error: {error}")

# Define objective function for hyperparameter tuning
def objective(trial):
    # Load the dataset
    series = AirPassengersDataset().load().astype(np.float32)

    # Split the data into train and validation sets
    VAL_LEN = 36
    train, val = series[:-VAL_LEN], series[-VAL_LEN:]

    # Scale the data
    scaler = Scaler(MaxAbsScaler())
    train = scaler.fit_transform(train)
    val = scaler.transform(val)

    # Select hyperparameters for the anomaly detection model (CPCModel)
    in_len = trial.suggest_int("in_len", 12, 36)
    out_len = trial.suggest_int("out_len", 1, in_len-1)
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    num_filters = trial.suggest_int("num_filters", 1, 5)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)

    # Create a Dask cluster
    cluster = dd.LocalCluster(n_workers=4, threads_per_worker=1)
    client = dd.Client(cluster)

    # Scatter the dataset to Dask workers
    train = client.scatter(train)

    # Initialize the CPC model on the workers
    model = CPCModel(input_dim, hidden_dim, num_layers, num_heads)
    models = client.scatter(model)

    # Train the CPC model on each worker using Dask's `client.run` function
    client.run(train_cpc_model_on_worker, models, train, num_epochs=10, learning_rate=lr)

    # Use the trained model for anomaly detection on each worker using Dask's `client.run` function
    anomalies = client.run(detect_anomalies_on_worker, models, train, threshold=0.5)

    # Gather the results from all workers
    all_anomalies = anomalies.gather()

    # Compute the sMAPE score for anomaly detection performance (use val set for evaluation)
    preds = model.predict(series=train, n=VAL_LEN)
    smapes = smape(val, preds, n_jobs=-1, verbose=True)
    smape_val = np.mean(smapes)

    # Return the sMAPE score for optimization
    return smape_val if smape_val != np.nan else float("inf")

# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

# optimize hyperparameters by minimizing the sMAPE on the validation set
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[print_callback])
