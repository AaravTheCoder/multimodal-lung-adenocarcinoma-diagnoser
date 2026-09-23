import pandas as pd

df = pd.read_csv("clinical.tsv", sep="\t")

# Print columns that contain "submitter" or "case" or "id"
matching_cols = [col for col in df.columns if any(x in col.lower() for x in ["submitter", "case", "id"])]
print("Potential ID columns found:")
for col in matching_cols[:15]:
    print(f" - {col}")