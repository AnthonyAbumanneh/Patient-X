import pandas as pd
import numpy as np

INPUT_FILE = "outputs/training_dataset.csv"

TRAIN_FILE = "outputs/train_dataset.csv"
VAL_FILE   = "outputs/val_dataset.csv"
TEST_FILE  = "outputs/test_dataset.csv"

RANDOM_SEED = 42

print("Loading training dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Total samples:", len(df))
print("Unique drugs:", df["drug_clean"].nunique())

# --------------------------------------------------
# 1. Get unique drugs
# --------------------------------------------------
unique_drugs = df["drug_clean"].dropna().unique()
print("Number of unique drugs to split:", len(unique_drugs))

# --------------------------------------------------
# 2. Shuffle drugs
# --------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(unique_drugs)

# --------------------------------------------------
# 3. Split drugs: 70 / 15 / 15
# --------------------------------------------------
n_total = len(unique_drugs)
n_train = int(0.70 * n_total)
n_val   = int(0.15 * n_total)

train_drugs = unique_drugs[:n_train]
val_drugs   = unique_drugs[n_train:n_train + n_val]
test_drugs  = unique_drugs[n_train + n_val:]

print("\nDrug split sizes:")
print("Train drugs:", len(train_drugs))
print("Val drugs:", len(val_drugs))
print("Test drugs:", len(test_drugs))

# --------------------------------------------------
# 4. Build row splits using drug membership
# --------------------------------------------------
train_df = df[df["drug_clean"].isin(train_drugs)].copy()
val_df   = df[df["drug_clean"].isin(val_drugs)].copy()
test_df  = df[df["drug_clean"].isin(test_drugs)].copy()

print("\nSample split sizes:")
print("Train samples:", len(train_df))
print("Val samples:", len(val_df))
print("Test samples:", len(test_df))

# --------------------------------------------------
# 5. Sanity checks: no leakage
# --------------------------------------------------
train_set = set(train_df["drug_clean"].unique())
val_set   = set(val_df["drug_clean"].unique())
test_set  = set(test_df["drug_clean"].unique())

print("\nLeakage checks:")
print("Train ∩ Val:", len(train_set & val_set))
print("Train ∩ Test:", len(train_set & test_set))
print("Val ∩ Test:", len(val_set & test_set))

# --------------------------------------------------
# 6. Save files
# --------------------------------------------------
train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("\nSaved files:")
print(TRAIN_FILE)
print(VAL_FILE)
print(TEST_FILE)