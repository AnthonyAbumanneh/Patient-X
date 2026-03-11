import pandas as pd
import numpy as np

INPUT_FILE = "outputs/faers_with_smiles.csv"
OUT_FILE = "outputs/faers_clean.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Initial rows:", len(df))

# --------------------------------------------------
# 1. Keep only rows with smiles
# --------------------------------------------------
df = df.dropna(subset=["smiles"]).copy()
print("After dropping missing smiles:", len(df))

# --------------------------------------------------
# 2. Normalize age into years
# --------------------------------------------------
def convert_age_to_years(age, unit):
    if pd.isna(age) or pd.isna(unit):
        return np.nan

    try:
        age = float(age)
    except:
        return np.nan

    unit = str(unit).strip().upper()

    if unit == "YR":
        return age
    elif unit == "MON":
        return age / 12
    elif unit == "WK":
        return age / 52
    elif unit == "DY":
        return age / 365
    elif unit == "HR":
        return age / (24 * 365)
    elif unit == "DEC":
        return age * 10
    else:
        return np.nan

df["age_years"] = df.apply(lambda row: convert_age_to_years(row["age"], row["age_unit"]), axis=1)

# --------------------------------------------------
# 3. Normalize weight into kg
# --------------------------------------------------
def convert_weight_to_kg(weight, unit):
    if pd.isna(weight) or pd.isna(unit):
        return np.nan

    try:
        weight = float(weight)
    except:
        return np.nan

    unit = str(unit).strip().upper()

    if unit == "KG":
        return weight
    elif unit == "LBS":
        return weight * 0.453592
    elif unit == "GMS":
        return weight / 1000
    else:
        return np.nan

df["weight_kg"] = df.apply(lambda row: convert_weight_to_kg(row["weight"], row["weight_unit"]), axis=1)

# --------------------------------------------------
# 4. Clean sex
# --------------------------------------------------
df["sex"] = df["sex"].astype(str).str.strip().str.upper()
df = df[df["sex"].isin(["M", "F"])].copy()

# --------------------------------------------------
# 5. Drop missing critical values
# --------------------------------------------------
df = df.dropna(subset=["drug", "side_effect", "age_years"]).copy()

# --------------------------------------------------
# 6. Remove unrealistic values
# --------------------------------------------------
df = df[(df["age_years"] >= 0) & (df["age_years"] <= 120)].copy()

# weight is optional for now, so do not force it
df.loc[(df["weight_kg"] <= 0) | (df["weight_kg"] > 400), "weight_kg"] = np.nan

# --------------------------------------------------
# 7. Keep only useful columns
# --------------------------------------------------
df = df[[
    "primaryid",
    "drug",
    "drug_clean",
    "smiles",
    "sex",
    "age_years",
    "weight_kg",
    "side_effect"
]].copy()

print("Final cleaned rows:", len(df))
print("Unique drugs:", df["drug"].nunique())
print("Unique side effects:", df["side_effect"].nunique())

df.to_csv(OUT_FILE, index=False)
print(f"Saved -> {OUT_FILE}")



counts = df["side_effect"].value_counts()

print("Total side effects:", len(counts))
print("Top 20:")
print(counts.head(20))

print("\nMedian frequency:", counts.median())
print("Side effects with <10 cases:", (counts < 10).sum())


total_rows = len(df)

coverage = counts.cumsum() / total_rows

print("\nCoverage by top K side effects:")
for k in [50,100,200,300,500,750,1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500]:
    if k < len(coverage):
        print(k, round(coverage.iloc[k],3))


import matplotlib.pyplot as plt

coverage = counts.cumsum() / total_rows

plt.plot(coverage.values[:3500])
plt.xlabel("Number of side effects")
plt.ylabel("Dataset coverage")
plt.title("Side Effect Coverage Curve")
plt.show()