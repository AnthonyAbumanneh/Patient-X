import pandas as pd
import re

FAERS_FILE = "outputs/faers_combined.csv"
SMILES_FILE = "outputs/drug_to_smiles.csv"
OUT_FILE = "outputs/faers_with_smiles.csv"


def clean_drug_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip().upper()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\(.*?\)", "", name).strip()
    return name


print("Loading files...")

faers = pd.read_csv(FAERS_FILE, low_memory=False)
smiles_map = pd.read_csv(SMILES_FILE, low_memory=False)

print("FAERS rows:", len(faers))
print("SMILES rows:", len(smiles_map))

# clean FAERS drug names the same way we cleaned them for PubChem lookup
faers["drug_clean"] = faers["drug"].map(clean_drug_name)

# remove duplicate drug mappings just in case
smiles_map = smiles_map.drop_duplicates(subset=["drug_clean"])

print("Merging SMILES into FAERS...")

merged = faers.merge(smiles_map, on="drug_clean", how="left")

print("Merged rows:", len(merged))
print("Rows with SMILES:", merged["smiles"].notna().sum())
print("Rows missing SMILES:", merged["smiles"].isna().sum())

merged.to_csv(OUT_FILE, index=False)

print(f"Saved -> {OUT_FILE}")


df = pd.read_csv("outputs/faers_with_smiles.csv")

print("Total rows:", len(df))
print("Rows with smiles:", df["smiles"].notna().sum())
print("Percent with smiles:", 100 * df["smiles"].notna().mean())

print("\nUnique drugs:", df["drug"].nunique())
print("Unique drugs with smiles:", df.dropna(subset=["smiles"])["drug"].nunique())