# 1. Imports
from data_pipeline import load_data, scale_data, prepare_sequences
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 2. Data prep
ts = load_data("data/SN_ms_tot_V2.0.csv")
ts = ts.loc["1750-01":"2019-12"]
scaled_series, scaler = scale_data(ts)
SEQ_LEN = 18
X, y = prepare_sequences(scaled_series, seq_len=SEQ_LEN)

# 3. Save scaler
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# 4. Split
dates = ts.index[12:]
train_end = dates.get_loc("2008-12-01")
val_end = dates.get_loc("2019-12-01")
X_train, y_train = X[:train_end+1], y[:train_end+1]
X_val, y_val = X[train_end+1:val_end+1], y[train_end+1:val_end+1]


# 5. Loaders
train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float()), batch_size=32)


# 6. NN architecture
class DeepNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32, 32]):
        """
        input_dim    – number of inputs (here SEQ_LEN=12)
        hidden_dims  – list of hidden‐layer sizes; length = depth
        """
        super().__init__()
        layers = []
        in_dim = input_dim
        # build hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        # final output layer
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 7. Hyperparams, Layer initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 18
model = DeepNN(input_dim=SEQ_LEN, hidden_dims=[16,64,8]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# 8. Training Loop
import copy

# --- before training loop ---
train_losses = []
val_losses   = []
best_val_loss = float('inf')
best_epoch    = -1
best_model_wts = copy.deepcopy(model.state_dict())

# --- training loop with checkpointing ---
EPOCHS = 200
for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    running_train = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running_train += loss.item() * xb.size(0)
    epoch_train_loss = running_train / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            running_val += criterion(model(xb), yb).item() * xb.size(0)
    epoch_val_loss = running_val / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    # Checkpoint if this is the best model so far
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_epoch    = epoch
        best_model_wts = copy.deepcopy(model.state_dict())

    # Print progress
    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train: {epoch_train_loss:.6f} | Val: {epoch_val_loss:.6f} "
              f"{'(best)' if epoch == best_epoch else ''}")

print(f"\nBest validation loss of {best_val_loss:.4f} at epoch {best_epoch}")

# --- restore best weights ---
model.load_state_dict(best_model_wts)

# Train & Validation Loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training & Validation Loss per Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Save best model
torch.save(model.state_dict(), "model/model_2.pth")

import joblib
joblib.dump(scaler, "model/scaler_2.pkl")