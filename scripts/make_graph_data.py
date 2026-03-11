import os
import ast
import torch
import pandas as pd
import numpy as np

from rdkit import Chem
from torch_geometric.data import Data

OUT_DIR = "outputs/gnn/graph_cache"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_FILE = "data/processed/train_dataset.csv"
VAL_FILE = "data/processed/val_dataset.csv"
TEST_FILE = "data/processed/test_dataset.csv"


def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetTotalNumHs(),
    ]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # node features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # edges
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return x, edge_index


def encode_sex(sex):
    sex = str(sex).strip().upper()
    if sex == "M":
        return 0.0
    elif sex == "F":
        return 1.0
    return 0.0


def load_and_process(csv_file, split_name):
    print(f"\nProcessing {split_name}...")
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"{split_name} rows before processing: {len(df)}")

    df["labels"] = df["labels"].apply(ast.literal_eval)

    # keep only needed rows
    df = df.dropna(subset=["smiles", "age_years"]).copy()

    # fill missing weight with median
    if "weight_kg" in df.columns:
        median_weight = df["weight_kg"].median()
        df["weight_kg"] = df["weight_kg"].fillna(median_weight)
    else:
        df["weight_kg"] = 70.0

    # normalize patient features using split stats for now
    # later, training script will re-normalize using train stats if needed
    age = df["age_years"].astype(float).to_numpy()
    sex = df["sex"].apply(encode_sex).to_numpy()
    weight = df["weight_kg"].astype(float).to_numpy()

    patient_features = np.stack([age, sex, weight], axis=1).astype(np.float32)

    data_list = []
    bad = 0

    for i, row in enumerate(df.itertuples(index=False)):
        graph = smiles_to_graph(row.smiles)
        if graph is None:
            bad += 1
            continue

        x, edge_index = graph

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(row.labels, dtype=torch.float).unsqueeze(0)
        )

        pf = torch.tensor(patient_features[i], dtype=torch.float).unsqueeze(0)
        data.patient_features = pf

        data.primaryid = str(row.primaryid)
        data.drug = str(row.drug)
        data.drug_clean = str(row.drug_clean)

        data_list.append(data)

    print(f"{split_name} valid graph samples: {len(data_list)}")
    print(f"{split_name} failed graph conversions: {bad}")

    torch.save(data_list, os.path.join(OUT_DIR, f"{split_name}_graphs.pt"))
    print(f"Saved -> {os.path.join(OUT_DIR, f'{split_name}_graphs.pt')}")

    if len(data_list) > 0:
        sample = data_list[0]
        print(f"{split_name} sample node feature shape: {sample.x.shape}")
        print(f"{split_name} sample edge_index shape: {sample.edge_index.shape}")
        print(f"{split_name} sample patient feature shape: {sample.patient_features.shape}")
        print(f"{split_name} sample label shape: {sample.y.shape}")


def main():
    load_and_process(TRAIN_FILE, "train")
    load_and_process(VAL_FILE, "val")
    load_and_process(TEST_FILE, "test")
    print("\nDone building graph datasets.")


if __name__ == "__main__":
    main()