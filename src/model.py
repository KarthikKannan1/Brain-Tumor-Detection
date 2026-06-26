"""
model.py — TumorMLP with swappable binary / multiclass output head.

Architecture:
    Input → [Linear → BatchNorm1d → ReLU → Dropout] × n_hidden → Linear head

Usage:
    model = make_model(input_dim=44, task='binary')   # num_classes=2
    history = fit(model, X_train, y_train, X_val, y_val)
    results = evaluate(model, X_test, y_test)

    model.swap_head(num_classes=4)                    # reuse backbone for multiclass
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model ─────────────────────────────────────────────────────────────────────

class TumorMLP(nn.Module):
    """
    Fully-connected MLP with a replaceable output head.

    Args:
        input_dim:   Number of input features.
        hidden_dims: Sequence of hidden layer widths.
        num_classes: Output neurons (2 for binary, 4 for 4-class).
        dropout:     Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        num_classes: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def swap_head(self, num_classes: int) -> None:
        """Replace the classification head while keeping backbone weights."""
        device = next(self.parameters()).device
        self.head = nn.Linear(self.head.in_features, num_classes).to(device)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_model(
    input_dim: int,
    task: str = "binary",
    hidden_dims: Tuple[int, ...] = (256, 128, 64),
    dropout: float = 0.4,
) -> TumorMLP:
    """Return a TumorMLP sized for the given task ('binary' or 'multiclass')."""
    num_classes = 2 if task == "binary" else 4
    return TumorMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def _to_tensors(
    X: np.ndarray, y: np.ndarray, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.long).to(device),
    )


def fit(
    model: TumorMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 20,
    weight_decay: float = 1e-4,
    device: str = DEVICE,
) -> Dict:
    """
    Train with Adam + cross-entropy; early-stop on validation accuracy.

    Returns:
        history dict with lists: train_loss, val_loss, train_acc, val_acc.
    """
    model.to(device)
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=5, factor=0.5
    )
    criterion  = nn.CrossEntropyLoss()

    X_tr, y_tr = _to_tensors(X_train, y_train, device)
    X_vl, y_vl = _to_tensors(X_val,   y_val,   device)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    history: Dict = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best_val_acc, best_state, wait = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        total_loss, correct = 0.0, 0
        for xb, yb in loader:
            optimiser.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(xb)
            correct    += (logits.argmax(1) == yb).sum().item()

        train_loss = total_loss / len(X_train)
        train_acc  = correct    / len(X_train)

        # ── validate ──
        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl)
            val_loss   = criterion(val_logits, y_vl).item()
            val_acc    = (val_logits.argmax(1) == y_vl).float().mean().item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ── early stopping ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    print(f"Stopped at epoch {epoch:3d}  |  best val acc: {best_val_acc:.4f}")
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model: TumorMLP,
    X: np.ndarray,
    y: np.ndarray,
    device: str = DEVICE,
) -> Dict:
    """
    Run inference and return a results dict.

    Returns:
        {
            'accuracy':     float,
            'predictions':  ndarray (N,)   — argmax class,
            'probabilities':ndarray (N, C) — softmax scores,
            'true_labels':  ndarray (N,),
        }
    """
    model.eval()
    model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

    return {
        "accuracy":      float((preds == y).mean()),
        "predictions":   preds,
        "probabilities": probs,
        "true_labels":   np.asarray(y),
    }
