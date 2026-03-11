import pandas as pd
import ast

INPUT_FILE = "outputs/training_dataset.csv"

df = pd.read_csv(INPUT_FILE, low_memory=False)

# labels were saved as strings like "[0, 1, 0, ...]"
df["labels"] = df["labels"].apply(ast.literal_eval)

num_samples = len(df)
num_labels = len(df["labels"].iloc[0])

positive_counts = [sum(row[i] for row in df["labels"]) for i in range(num_labels)]
total_positives = sum(positive_counts)

avg_labels_per_sample = total_positives / num_samples
label_prevalences = [c / num_samples for c in positive_counts]

print("Number of samples:", num_samples)
print("Number of labels:", num_labels)
print("Average positive labels per sample:", round(avg_labels_per_sample, 3))
print("Most common label prevalence:", round(max(label_prevalences), 4))
print("Median label prevalence:", round(pd.Series(label_prevalences).median(), 6))
print("Labels with prevalence < 0.001:", sum(p < 0.001 for p in label_prevalences))