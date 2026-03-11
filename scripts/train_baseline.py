import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "outputs/fingerprints"
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset
# -----------------------------
class FingerprintDataset(Dataset):
    def __init__(self, x_drug, x_patient, y):
        self.x = np.concatenate([x_drug, x_patient], axis=1).astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


# -----------------------------
# Model
# -----------------------------
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Metrics
# -----------------------------
def precision_at_k(logits, targets, k=10):
    probs = torch.sigmoid(logits)
    topk = torch.topk(probs, k=k, dim=1).indices

    correct = 0
    total = targets.size(0)

    for i in range(total):
        true_labels = set(torch.where(targets[i] == 1)[0].cpu().numpy())
        pred_labels = set(topk[i].cpu().numpy())
        if len(true_labels & pred_labels) > 0:
            correct += 1

    return correct / total


# -----------------------------
# Train / Eval
# -----------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    all_prec10 = []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        logits = model(xb)
        loss = criterion(logits, yb)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        all_prec10.append(precision_at_k(logits.detach(), yb, k=10))

    avg_loss = total_loss / len(loader.dataset)
    avg_prec10 = sum(all_prec10) / len(all_prec10)

    return avg_loss, avg_prec10


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading arrays...")

    X_drug_train = np.load(f"{DATA_DIR}/X_drug_train.npy")
    X_patient_train = np.load(f"{DATA_DIR}/X_patient_train.npy")
    Y_train = np.load(f"{DATA_DIR}/Y_train.npy")

    X_drug_val = np.load(f"{DATA_DIR}/X_drug_val.npy")
    X_patient_val = np.load(f"{DATA_DIR}/X_patient_val.npy")
    Y_val = np.load(f"{DATA_DIR}/Y_val.npy")

    # normalize patient features using train stats
    mean = X_patient_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = (X_patient_train.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)    

    X_patient_train = (X_patient_train - mean) / std
    X_patient_val = (X_patient_val - mean) / std

    train_ds = FingerprintDataset(X_drug_train, X_patient_train, Y_train)
    val_ds = FingerprintDataset(X_drug_val, X_patient_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_drug_train.shape[1] + X_patient_train.shape[1]
    output_dim = Y_train.shape[1]

    print("Input dim:", input_dim)
    print("Output dim:", output_dim)

    # positive class weighting trick
    pos_counts = Y_train.sum(axis=0)
    neg_counts = len(Y_train) - pos_counts
    pos_weight = np.where(pos_counts > 0, neg_counts / (pos_counts + 1e-8), 1.0)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

    model = BaselineMLP(input_dim, output_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_p10 = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_p10 = run_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train P@10: {train_p10:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val P@10: {val_p10:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                },
                f"{DATA_DIR}/baseline_model.pt"
            )
            print("Saved best model.")

    print("Training complete.")


if __name__ == "__main__":
    main()