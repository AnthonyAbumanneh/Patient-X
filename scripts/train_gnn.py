import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from models.gnn_model import GNNDrugSideEffectModel

# -----------------------------
# Config
# -----------------------------
TRAIN_FILE = "outputs/gnn/graph_cache/train_graphs.pt"
VAL_FILE = "outputs/gnn/graph_cache/val_graphs.pt"
OUT_FILE = "outputs/gnn/gnn_model.pt"

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Helpers
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


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    all_p10 = []

    for batch in loader:
        batch = batch.to(DEVICE)

        # y was saved as shape [1, output_dim] per sample
        y = batch.y.view(batch.num_graphs, -1)

        logits = model(batch)
        loss = criterion(logits, y)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_p10.append(precision_at_k(logits.detach(), y, k=10))

    avg_loss = total_loss / len(loader.dataset)
    avg_p10 = sum(all_p10) / len(all_p10)

    return avg_loss, avg_p10


def normalize_patient_features(train_data, val_data):
    # stack train patient features to compute train-only normalization
    train_pf = torch.cat([d.patient_features for d in train_data], dim=0).numpy()

    mean = train_pf.mean(axis=0, keepdims=True).astype(np.float32)
    std = (train_pf.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

    for dataset in [train_data, val_data]:
        for d in dataset:
            arr = d.patient_features.numpy()
            arr = (arr - mean) / std
            d.patient_features = torch.tensor(arr, dtype=torch.float)

    return mean, std


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading graph datasets...")

    train_data = torch.load(TRAIN_FILE, weights_only=False)
    val_data = torch.load(VAL_FILE, weights_only=False)

    print("Train samples:", len(train_data))
    print("Val samples:", len(val_data))

    # normalize patient features using train split only
    mean, std = normalize_patient_features(train_data, val_data)

    node_dim = train_data[0].x.shape[1]
    output_dim = train_data[0].y.shape[1]

    print("Node feature dim:", node_dim)
    print("Output dim:", output_dim)

    # positive class weighting trick
    Y_train = torch.cat([d.y for d in train_data], dim=0).numpy()
    pos_counts = Y_train.sum(axis=0)
    neg_counts = len(Y_train) - pos_counts
    pos_weight = np.where(pos_counts > 0, neg_counts / (pos_counts + 1e-8), 1.0)
    pos_weight = np.clip(pos_weight, 1.0, 50.0)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = GNNDrugSideEffectModel(
        node_dim=node_dim,
        patient_dim=3,
        gnn_hidden=128,
        patient_hidden=64,
        output_dim=output_dim
    ).to(DEVICE)

    print(model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

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
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "mean": mean,
                    "std": std,
                    "node_dim": node_dim,
                    "output_dim": output_dim,
                },
                OUT_FILE
            )
            print("Saved best GNN model.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()