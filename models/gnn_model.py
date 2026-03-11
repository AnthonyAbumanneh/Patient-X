import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class PatientEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class DrugGNN(nn.Module):
    def __init__(self, node_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        # graph-level embedding
        x = global_mean_pool(x, batch)
        return x


class GNNDrugSideEffectModel(nn.Module):
    def __init__(self, node_dim, patient_dim=3, gnn_hidden=128, patient_hidden=64, output_dim=1500):
        super().__init__()

        self.drug_encoder = DrugGNN(node_dim=node_dim, hidden_dim=gnn_hidden)
        self.patient_encoder = PatientEncoder(input_dim=patient_dim, hidden_dim=patient_hidden)

        fusion_dim = gnn_hidden + patient_hidden

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, data):
        # graph inputs
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # patient inputs
        patient = data.patient_features

        drug_emb = self.drug_encoder(x, edge_index, batch)
        patient_emb = self.patient_encoder(patient)

        fused = torch.cat([drug_emb, patient_emb], dim=1)
        logits = self.classifier(fused)
        return logits