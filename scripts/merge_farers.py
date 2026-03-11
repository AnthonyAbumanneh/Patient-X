import pandas as pd

print("Loading FAERS data...")

demo = pd.read_csv("data/DEMO25Q4.txt", delimiter="$", low_memory=False)
drug = pd.read_csv("data/DRUG25Q4.txt", delimiter="$", low_memory=False)
reac = pd.read_csv("data/REAC25Q4.txt", delimiter="$", low_memory=False)

print("Rows loaded:")
print("DEMO:", len(demo))
print("DRUG:", len(drug))
print("REAC:", len(reac))

# ---- Keep only useful columns ----
demo = demo[["primaryid", "age", "age_cod", "sex", "wt", "wt_cod"]].copy()
drug = drug[["primaryid", "drugname", "role_cod"]].copy()
reac = reac[["primaryid", "pt"]].copy()

# ---- Keep only primary suspect drugs (reduces noise) ----
drug = drug[drug["role_cod"] == "PS"].copy()

print("Merging tables on primaryid...")

df = demo.merge(drug, on="primaryid", how="inner")
df = df.merge(reac, on="primaryid", how="inner")

# ---- Rename columns to nice names ----
df = df.rename(columns={
    "drugname": "drug",
    "pt": "side_effect",
    "age": "age",
    "sex": "sex",
    "wt": "weight",
    "age_cod": "age_unit",
    "wt_cod": "weight_unit"
})

print("Final dataset size:", len(df))

df.to_csv("outputs/faers_combined.csv", index=False)
print("Saved -> outputs/faers_combined.csv")