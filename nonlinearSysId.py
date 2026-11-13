"""
Neural Network training for lettuce greenhouse benchmark data.

Target : dy/dt  (numerical derivative of y1..y4, computed per scenario)
Features: u1, u2, u3  (control inputs)
          d1, d2, d3, d4  (disturbances)
          y1, y2, y3, y4  (outputs / states)

Split   : train/validation scenarios  → training + validation
          test scenario               → held-out test set
"""
# import user-defined help functions
import library as lib 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ── 0. Config ────────────────────────────────────────────────────────────────
DATA_PATH   = "data/lettuceGreenhouseBenchmarkData.csv"
DT          = 900.0          # seconds per time step
VAL_RATIO   = 0.2            # fraction of train/val scenarios used for val
HIDDEN_DIMS = [16,16]        # hidden layer sizes
EPOCHS      = 200
BATCH_SIZE  = 128
LR          = 1e-3
SEED        = 23

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df           = pd.read_csv(DATA_PATH)
Y_COLS       = ["y1", "y2", "y3", "y4"]
U_COLS       = ["u1", "u2", "u3"]
D_COLS       = ["d1", "d2", "d3", "d4"]
FEATURE_COLS = U_COLS + D_COLS + Y_COLS   # 11 features total

# ── 2. Compute dy/dt per scenario (avoids cross-scenario finite differences) ──
def compute_dydt(group: pd.DataFrame) -> pd.DataFrame:
    """Forward difference dy/dt; last row of each scenario is dropped."""
    dy = group[Y_COLS].diff().shift(-1) / DT   # shape (N, 4)
    group = group.copy()
    for col in Y_COLS:
        group[f"d{col}_dt"] = dy[col]
    return group

df = (df.groupby("scenario", group_keys=False)[df.columns].apply(compute_dydt))
df = df.dropna()   # removes last row of every scenario

TARGET_COLS = [f"d{c}_dt" for c in Y_COLS]   # ["dy1_dt", ..., "dy4_dt"]

# ── 3. Train / test split (by casus label) ───────────────────────────────────
train_val_df = df[df["casus"] == "train/validation"].copy()
test_df      = df[df["casus"] == "test"].copy()

# Further split train_val by scenario index
scenarios       = train_val_df["scenario"].unique()
n_val           = max(1, int(len(scenarios) * VAL_RATIO))
val_scenarios   = scenarios[-n_val:]
train_scenarios = scenarios[:-n_val]

train_df = train_val_df[train_val_df["scenario"].isin(train_scenarios)]
val_df   = train_val_df[train_val_df["scenario"].isin(val_scenarios)]

print(f"Train rows: {len(train_df)}  |  Val rows: {len(val_df)}  |  Test rows: {len(test_df)}")

# ── 4. Scale features and targets ────────────────────────────────────────────
feat_scaler   = StandardScaler()
target_scaler = StandardScaler()
X_train       = feat_scaler.fit_transform(train_df[FEATURE_COLS].values)
y_train       = target_scaler.fit_transform(train_df[TARGET_COLS].values)
X_val         = feat_scaler.transform(val_df[FEATURE_COLS].values)
y_val         = target_scaler.transform(val_df[TARGET_COLS].values)
X_test        = feat_scaler.transform(test_df[FEATURE_COLS].values)
y_test        = target_scaler.transform(test_df[TARGET_COLS].values)

def to_tensors(*arrays):
    return [torch.tensor(a, dtype=torch.float32) for a in arrays]

X_tr, y_tr = to_tensors(X_train, y_train)
X_v,  y_v  = to_tensors(X_val,   y_val)
X_te, y_te = to_tensors(X_test,  y_test)

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

# ── 5. Model ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: list[int]):
        super().__init__()
        dims   = [in_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

in_dim  = len(FEATURE_COLS)   # 11
out_dim = len(TARGET_COLS)    # 4
model   = MLP(in_dim, out_dim, HIDDEN_DIMS)


print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── 6. Training loop ──────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.MSELoss()

train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    # -- train --
    model.train()
    batch_losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = np.mean(batch_losses)

    # -- validate --
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_v), y_v).item()

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}  train MSE={train_loss:.6f}  val MSE={val_loss:.6f}")

# ── 7. Test evaluation ────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_te).numpy()

y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_true = target_scaler.inverse_transform(y_te.numpy())

rmse_per_output = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
for i, col in enumerate(TARGET_COLS):
    print(f"Test RMSE  {col}: {rmse_per_output[i]:.6f}")

# ── 8. Euler simulation on the test scenario ─────────────────────────────────
#
#  At every time step k:
#    dy/dt  = NN(y_k, u_k, d_k)          ← neural network
#    y_{k+1} = y_k + dt * dy/dt           ← Euler step
#
#  u and d come from the recorded test data (open-loop replay).
#  Only the initial condition y[0] is taken from the test data.

model.eval()

# Raw (unscaled) test inputs and true outputs
U_test = test_df[U_COLS].values          # (N, 3)
D_test = test_df[D_COLS].values          # (N, 4)
Y_true = test_df[Y_COLS].values          # (N, 4)  – ground truth
t_sim  = test_df["t"].values / 3600.0    # hours

N      = len(Y_true)
Y_sim  = np.zeros_like(Y_true)
Y_sim[0] = Y_true[0]                     # initial condition from data

for k in range(N - 1):
    y_k = Y_sim[k]                       # current simulated state  (4,)
    u_k = U_test[k]                      # control input at step k  (3,)
    d_k = D_test[k]                      # disturbance at step k    (4,)

    # Build feature vector [u, d, y] and scale
    feat = np.concatenate([u_k, d_k, y_k]).reshape(1, -1)   # (1, 11)
    feat_scaled = feat_scaler.transform(feat)

    # Predict dy/dt (scaled), then invert scaling
    with torch.no_grad():
        dydt_scaled = model(torch.tensor(feat_scaled, dtype=torch.float32)).numpy()
    dydt = target_scaler.inverse_transform(dydt_scaled).flatten()  # (4,)

    # Euler step
    Y_sim[k + 1] = y_k + DT * dydt

# ── 9. Simulation RMSE ───────────────────────────────────────────────────────
sim_rmse = np.sqrt(np.mean((Y_sim - Y_true) ** 2, axis=0))
for i, col in enumerate(Y_COLS):
    print(f"Simulation RMSE  {col}: {sim_rmse[i]:.6f}")

# ── 10. Plots ─────────────────────────────────────────────────────────────────
OUTPUT_LABELS = ["$y_1$", "$y_2$", "$y_3$", "$y_4$"]
Y_UNITS       = ["(g/m²)", r"(ppm) $\cdot 10^2$", "(°C)", "(%)"]   # adjust if known

# ── 10a. Loss curves ──────────────────────────────────────────────────────────
fig_loss, ax_loss = plt.subplots(figsize=(7, 3))
ax_loss.plot(train_losses, label="Train MSE")
ax_loss.plot(val_losses,   label="Val MSE")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("MSE (scaled)")
ax_loss.set_title("Training / Validation Loss")
ax_loss.legend()
ax_loss.grid(True)
fig_loss.tight_layout()
#fig_loss.savefig("loss_curves.png", dpi=150)

# ── 10b. Parity plots ─────────────────────────────────────────────────────────
fig_par, axes_par = plt.subplots(1, 4, figsize=(14, 4))
fig_par.suptitle("Parity Plots – Euler Simulation vs. Test Data", fontsize=13)

for i, ax in enumerate(axes_par):
    ax.scatter(Y_true[:, i], Y_sim[:, i], s=4, alpha=0.35, color="steelblue")
    lo = min(Y_true[:, i].min(), Y_sim[:, i].min())
    hi = max(Y_true[:, i].max(), Y_sim[:, i].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="ideal")
    ax.set_xlabel(f"True {OUTPUT_LABELS[i]}")
    ax.set_ylabel(f"Simulated {OUTPUT_LABELS[i]}")
    ax.set_title(f"RMSE = {sim_rmse[i]:.2e}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

fig_par.tight_layout()
#fig_par.savefig("parity_plots.png", dpi=150)

plt.show()

###############################################################################
###############################################################################
ops       = {}
ops['nx'] = 4    # #of states
ops['nu'] = 3    # #controllable inputs
ops['nd'] = 4    # #disturbance inputs
c         = 86400 

ops['t']  = test_df['t'].to_numpy()
d         = D_test.T
u         = U_test.T
x         = test_df[['x1','x2','x3','x4']].to_numpy().T
y         = Y_true.T
yhat      = Y_sim.T

lib.plotTestValidation(d,u,x,y,ops['t'],yhat,ops)

fit_i = 100*(1 - np.linalg.norm(y - yhat, axis=1) /
                 np.linalg.norm(y - y.mean(axis=1, keepdims=True), axis=1))
rmse_i = np.sqrt(np.mean((y - yhat)**2, axis=1))

print(np.array2string(rmse_i, precision=2, suppress_small=True))
print(np.array2string(fit_i, precision=2, suppress_small=True))