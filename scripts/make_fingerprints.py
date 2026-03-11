import pandas as pd
import numpy as np
import ast
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

TRAIN_FILE = "outputs/train_dataset.csv"
VAL_FILE   = "outputs/val_dataset.csv"
TEST_FILE  = "outputs/test_dataset.csv"

OUT_DIR = "outputs/fingerprints"
os.makedirs(OUT_DIR, exist_ok=True)

N_BITS = 2048
RADIUS = 2


def smiles_to_morgan(smiles, n_bits=N_BITS, radius=RADIUS):
    """Convert a SMILES string to a Morgan fingerprint numpy array."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def parse_sex(sex):
    sex = str(sex).strip().upper()
    if sex == "M":
        return 0
    elif sex == "F":
        return 1
    return -1


def process_file(input_file, split_name):
    print(f"\nProcessing {split_name}...")
    df = pd.read_csv(input_file, low_memory=False)

    print(f"{split_name} rows before filtering: {len(df)}")

    # parse labels back from string to list
    df["labels"] = df["labels"].apply(ast.literal_eval)

    # keep rows with valid patient values
    df = df.dropna(subset=["smiles", "age_years"]).copy()

    # sex encoding
    df["sex_encoded"] = df["sex"].apply(parse_sex)
    df = df[df["sex_encoded"].isin([0, 1])].copy()

    # weight: keep missing weights, fill with median of this split
    if "weight_kg" in df.columns:
        median_weight = df["weight_kg"].median()
        df["weight_kg"] = df["weight_kg"].fillna(median_weight)
    else:
        df["weight_kg"] = 70.0

    # create fingerprints
    fps = []
    keep_indices = []

    for i, smiles in enumerate(df["smiles"].tolist()):
        fp = smiles_to_morgan(smiles)
        if fp is not None:
            fps.append(fp)
            keep_indices.append(i)

    df = df.iloc[keep_indices].copy()

    print(f"{split_name} rows after fingerprinting: {len(df)}")

    X_drug = np.stack(fps)

    # patient feature matrix
    X_patient = df[["age_years", "sex_encoded", "weight_kg"]].to_numpy(dtype=np.float32)

    # labels matrix
    Y = np.array(df["labels"].tolist(), dtype=np.float32)

    # metadata
    meta = df[["primaryid", "drug", "drug_clean"]].copy()

    print(f"{split_name} X_drug shape: {X_drug.shape}")
    print(f"{split_name} X_patient shape: {X_patient.shape}")
    print(f"{split_name} Y shape: {Y.shape}")

    # save
    np.save(os.path.join(OUT_DIR, f"X_drug_{split_name}.npy"), X_drug)
    np.save(os.path.join(OUT_DIR, f"X_patient_{split_name}.npy"), X_patient)
    np.save(os.path.join(OUT_DIR, f"Y_{split_name}.npy"), Y)
    meta.to_csv(os.path.join(OUT_DIR, f"meta_{split_name}.csv"), index=False)

    print(f"Saved {split_name} arrays and metadata.")


if __name__ == "__main__":
    process_file(TRAIN_FILE, "train")
    process_file(VAL_FILE, "val")
    process_file(TEST_FILE, "test")

    print("\nDone. Saved all fingerprint datasets to outputs/fingerprints/")