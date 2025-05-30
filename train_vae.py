import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, TensorDataset
import matplotlib.pyplot as plt
from vae_model_pytorch import AE  

# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = SEED) -> None:
    """Ensure deterministic behaviour across NumPy and PyTorch (CPU & CUDA)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------------------------------
# utility functions
# 1) Import data from data.npz and create torch tensor for X train and test values
# 2) He initialisation function through Kaming-normal method to every linear layer in the AE model
# -----------------------------------------------------------------------------------------------------

def load_data(filename: str = "data.npz"):
    """Load train/val tensor and test tensor from an *npz* file.

    The file **must** contain at least the key ``"train_data"``.
    If a ``"test_data"`` key is present we treat it as the held‑out test set;
    otherwise the function raises.
    """
    data = np.load(filename)

    if "train_data" not in data:
        raise KeyError("'train_data' array not found in" f" {filename!r}")

    if "test_data" not in data:
        raise KeyError("'test_data' array not found – provide a held‑out set")

    x_train = torch.from_numpy(data["train_data"].astype(np.float32))
    x_test  = torch.from_numpy(data["test_data"].astype(np.float32))
    return x_train, x_test


def apply_he_init(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


# -----------------------------------------------------------------------------------------------
# K‑Fold helpers 
# train one epoch on the Kfold split
# evaluate the epoch for the remaining validation set
# Running loss is normalised on the data set to be easily intepretated 
# -----------------------------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: optim.Optimizer,
    criterion: nn.Module,
):
    model.train()
    running = 0.0
    for (x,) in loader:
        x = x.to(DEVICE)
        optimiser.zero_grad()
        out, _ = model(x)
        loss = criterion(out, x)
        loss.backward()
        optimiser.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


def evaluate(
    model = nn.Module,
    loader = DataLoader,
    criterion = nn.Module,
):
    
    model.eval()
    running = 0.0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(DEVICE)
            out, _ = model(x)
            running += criterion(out, x).item() * x.size(0)
    return running / len(loader.dataset)

#----------------------------------------------------------------------------------------
# Runs the training step for one fold over the range of epochs 
# output the training losses to the terminal
#----------------------------------------------------------------------------------------

def run_fold(data: torch.Tensor,
             train_idx: np.ndarray,
             val_idx: np.ndarray,
             epochs: int = 250,
             batch_size: int = 256,
             lr: float = 1e-3
) :
    

    model = AE().to(DEVICE)
    apply_he_init(model)

    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(Subset(TensorDataset(data), train_idx), batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(TensorDataset(data), val_idx),   batch_size)

    train_loss_hist, val_loss_hist = [], []

  
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion)
        val_loss   = evaluate(model, val_loader, criterion)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        if epoch % 100 == 0 or epoch == epochs:
            print(f"Epoch {epoch:4d}/{epochs} │ train {train_loss:.6f} │ val {val_loss:.6f}")

    return train_loss_hist, val_loss_hist, model.state_dict()


# -----------------------------------------------------------------------------------
# Main training routine over the full range of K folds 
# The best epoch with lowest validation MSE is determined and used for the full
# re train step the weights are saved to be loaded to the optimisation routine
# Data from the training process is saved for future processing
# Test set is used to evaluate the model performance after the full training step
# -----------------------------------------------------------------------------------

def kfold_train(
    k = 5,
    epochs= 200,
    batch_size =64,
    lr = 0.001,
    filename = "data.npz",
) :

    # 1 – Load data 
    x_train, x_test = load_data(filename)
    full_ds = TensorDataset(x_train)

    # 2 – K‑Fold cross‑validation 
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)

    fold_train_curves, fold_val_curves = [], []
    print("\n=== K‑Fold CV ===")
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train), start=1):
        print(f"\n── FOLD {fold}/{k} ──────────────────────────────────────")
        train_hist, val_hist, _ = run_fold(
            data=x_train,
            train_idx=train_idx,
            val_idx=val_idx,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        fold_train_curves.append(train_hist)
        fold_val_curves.append(val_hist)

    # 3 – Best epoch across folds
    min_len = min(len(c) for c in fold_val_curves)
    avg_val = np.mean([c[:min_len] for c in fold_val_curves], axis=0)
    best_epoch = int(avg_val.argmin()) + 1  # 1‑based
    print(f"\n Best epoch averaged over folds = {best_epoch}/{min_len}")

    # 4 full training data training step 
    final_model = AE().to(DEVICE)
    apply_he_init(final_model)

    optimiser = optim.Adam(final_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
    full_train_curve: List[float] = []

    print("\n=== Final re‑train on full train+val set ===")
    for epoch in range(1, best_epoch + 1):
        epoch_loss = train_one_epoch(final_model, full_loader, optimiser, criterion)
        full_train_curve.append(epoch_loss)

        if epoch % 100 == 0 or epoch == best_epoch:
            print(f"Epoch {epoch:4d}/{best_epoch} │ train {epoch_loss:.6f}")

    torch.save(final_model.state_dict(), "ae_final.pt")

    # 5 – Test evaluation 
    test_loader = DataLoader(TensorDataset(x_test), batch_size=200)
    test_mse = evaluate(final_model, test_loader, criterion)
    print(f"\n Final test MSE = {test_mse:.6f}\n")

    # 6 – Plot curves
    plot_training_progress(fold_train_curves, fold_val_curves, best_epoch, full_train_curve)

    return final_model, test_mse


# -----------------------------------------------------------------------------
# Plotting utils
# -----------------------------------------------------------------------------

def plot_training_progress(
    train_curves: float,
    val_curves: float,
    best_epoch: int,
    full_train_curve: float
): 

    k = len(train_curves)
    min_len = min(len(c) for c in train_curves)
    xs = np.arange(1, min_len + 1)

    plt.figure(figsize=(9, 5))

    # per‑fold curves 
    for i in range(k):
        plt.plot(xs, val_curves[i][:min_len], alpha=0.4, linestyle="--", label=f"val fold {i+1}")

    # mean curves
    mean_train = np.mean([c[:min_len] for c in train_curves], axis=0)
    mean_val   = np.mean([c[:min_len] for c in val_curves], axis=0)
    plt.plot(xs, mean_train, label="mean train", linewidth=2)
    plt.plot(xs, mean_val,   label="mean val",   linewidth=2)

    # final re‑train curve 
    xs_full = np.arange(1, len(full_train_curve) + 1)
    plt.plot(xs_full, full_train_curve, label="full re‑train", linewidth=2)

    # best epoch marker 
    plt.axvline(best_epoch, color="k", linestyle=":", label=f"best epoch = {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="small")
    plt.tight_layout()

    out = Path("training_progress.png")
    plt.savefig(out, dpi=150)
    print(f" Saved plot → {out.resolve()}")


# -----------------------------------------------------------------------------
# Running the training steps
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    kfold_train(
    k = 5,
    epochs= 1000,
    batch_size = 128,
    lr = 0.001,
    filename = "data.npz",
    )

