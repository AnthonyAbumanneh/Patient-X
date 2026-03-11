import pandas as pd
from rdkit import Chem

INPUT_FILE = "outputs/faers_clean.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Rows:", len(df))
print("Unique drugs:", df["drug_clean"].nunique())

# unique drug -> smiles pairs
pairs = df[["drug_clean", "smiles"]].drop_duplicates().copy()

print("Unique drug/SMILES pairs:", len(pairs))

def is_valid_smiles(smiles):
    if pd.isna(smiles):
        return False
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except Exception:
        return False

pairs["valid_smiles"] = pairs["smiles"].apply(is_valid_smiles)

valid_count = pairs["valid_smiles"].sum()
invalid_count = len(pairs) - valid_count

print("Valid SMILES pairs:", valid_count)
print("Invalid SMILES pairs:", invalid_count)
print("Percent valid:", round(100 * valid_count / len(pairs), 2))

if invalid_count > 0:
    print("\nExamples of invalid SMILES:")
    print(pairs.loc[~pairs["valid_smiles"], ["drug_clean", "smiles"]].head(10).to_string(index=False))