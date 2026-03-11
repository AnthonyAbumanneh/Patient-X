import pandas as pd
import json

INPUT_FILE = "outputs/faers_clean.csv"
OUT_FILE = "outputs/training_dataset.csv"
LABEL_MAP_FILE = "outputs/side_effect_label_map.json"

TOP_K = 1500

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Initial rows:", len(df))

# --------------------------------------------------
# 1. Keep only top-K most frequent side effects
# --------------------------------------------------
top_effects = df["side_effect"].value_counts().head(TOP_K).index.tolist()
top_effects_set = set(top_effects)

df = df[df["side_effect"].isin(top_effects_set)].copy()

print("Rows after top-K filter:", len(df))
print("Unique side effects kept:", df["side_effect"].nunique())

# --------------------------------------------------
# 2. Build label map
# --------------------------------------------------
label_map = {effect: i for i, effect in enumerate(top_effects)}

with open(LABEL_MAP_FILE, "w") as f:
    json.dump(label_map, f, indent=2)

print(f"Saved label map -> {LABEL_MAP_FILE}")

# --------------------------------------------------
# 3. Group by one sample = (primaryid + drug)
# --------------------------------------------------
group_cols = ["primaryid", "drug", "drug_clean", "smiles", "sex", "age_years", "weight_kg"]

grouped = df.groupby(group_cols)["side_effect"].apply(list).reset_index()

print("Grouped samples:", len(grouped))

# --------------------------------------------------
# 4. Convert side effects into multi-hot vectors
# --------------------------------------------------
def make_multihot(effect_list):
    vec = [0] * TOP_K
    for effect in effect_list:
        idx = label_map.get(effect)
        if idx is not None:
            vec[idx] = 1
    return vec

grouped["labels"] = grouped["side_effect"].apply(make_multihot)

# optional: keep count of how many positive labels each sample has
grouped["num_side_effects"] = grouped["labels"].apply(sum)

# drop raw side effect list after encoding
grouped = grouped.drop(columns=["side_effect"])

print("Final training samples:", len(grouped))
print("Average labels per sample:", grouped["num_side_effects"].mean())

# --------------------------------------------------
# 5. Save
# --------------------------------------------------
grouped.to_csv(OUT_FILE, index=False)

print(f"Saved -> {OUT_FILE}")