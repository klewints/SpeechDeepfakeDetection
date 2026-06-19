import pandas as pd

df = pd.read_csv("dataset/metadata.csv")

print(df["label"].value_counts())