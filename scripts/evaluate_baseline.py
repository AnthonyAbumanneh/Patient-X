import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "outputs/fingerprints"
BATCH_SIZE = 128
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


def recall_at_k(logits, targets, k=10):
    probs = torch.sigmoid(logits)
    topk = torch.topk(probs, k=k, dim=1).indices

    recalls = []

    for i in range(targets.size(0)):
        true_labels = set(torch.where(targets[i] == 1)[0].cpu().numpy())
        pred_labels = set(topk[i].cpu().numpy())

        if len(true_labels) == 0:
            continue

        recalls.append(len(true_labels & pred_labels) / len(true_labels))

    return sum(recalls) / len(recalls) if recalls else 0.0


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_p10 = []
    all_r10 = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            all_p10.append(precision_at_k(logits, yb, k=10))
            all_r10.append(recall_at_k(logits, yb, k=10))

    avg_loss = total_loss / len(loader.dataset)
    avg_p10 = sum(all_p10) / len(all_p10)
    avg_r10 = sum(all_r10) / len(all_r10)

    return avg_loss, avg_p10, avg_r10


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading saved model checkpoint...")
    checkpoint = torch.load(
    f"{DATA_DIR}/baseline_model.pt",
    map_location=DEVICE,
    weights_only=False
)

    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    print("Input dim:", input_dim)
    print("Output dim:", output_dim)

    print("\nLoading test arrays...")
    X_drug_test = np.load(f"{DATA_DIR}/X_drug_test.npy")
    X_patient_test = np.load(f"{DATA_DIR}/X_patient_test.npy")
    Y_test = np.load(f"{DATA_DIR}/Y_test.npy")

    print("X_drug_test shape:", X_drug_test.shape)
    print("X_patient_test shape:", X_patient_test.shape)
    print("Y_test shape:", Y_test.shape)

    # normalize patient features using saved train stats
    X_patient_test = (X_patient_test - mean) / std

    test_ds = FingerprintDataset(X_drug_test, X_patient_test, Y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BaselineMLP(input_dim, output_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()

    test_loss, test_p10, test_r10 = evaluate(model, test_loader, criterion)

    print("\n===== TEST RESULTS =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Precision@10: {test_p10:.4f}")
    print(f"Test Recall@10: {test_r10:.4f}")


if __name__ == "__main__":
    main()