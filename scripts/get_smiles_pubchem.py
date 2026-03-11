import pandas as pd
from tqdm import tqdm
import pubchempy as pcp
import re

INPUT_CSV = "outputs/faers_combined.csv"
OUT_MAP   = "outputs/drug_to_smiles.csv"

def clean_drug_name(name: str) -> str:
    """Basic cleanup for FAERS drugname strings."""
    if not isinstance(name, str):
        return ""
    name = name.strip().upper()
    # remove weird extra spaces
    name = re.sub(r"\s+", " ", name)
    # common junk patterns (keep this minimal for now)
    # e.g., "IBUPROFEN (ADVIL)" -> "IBUPROFEN"
    name = re.sub(r"\(.*?\)", "", name).strip()
    return name

def fetch_smiles(drug_name: str) -> str | None:
    """Query PubChem by name and return isomeric SMILES if found."""
    try:
        hits = pcp.get_compounds(drug_name, "name")
        if not hits:
            return None
        c = hits[0]
        return c.isomeric_smiles or c.canonical_smiles
    except Exception:
        return None

def main():
    df = pd.read_csv(INPUT_CSV, usecols=["drug"])
    df["drug_clean"] = df["drug"].map(clean_drug_name)

    # unique drug names (start small if you want by slicing head())
    drugs = sorted(df["drug_clean"].dropna().unique().tolist())
    print("Unique drug names:", len(drugs))

    rows = []
    for d in tqdm(drugs):
        if not d:
            continue
        smiles = fetch_smiles(d)
        rows.append({"drug_clean": d, "smiles": smiles})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_MAP, index=False)
    print(f"Saved mapping -> {OUT_MAP}")
    print("Found SMILES for:", out["smiles"].notna().sum(), "/", len(out))

if __name__ == "__main__":
    main()/