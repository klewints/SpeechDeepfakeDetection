import os
import pandas as pd

data = []

base_dir = "dataset/audio"

for label_name in ["real", "fake"]:

    folder = os.path.join(base_dir, label_name)

    label = 0 if label_name == "real" else 1

    for file in os.listdir(folder):

        if file.endswith(".wav"):

            path = os.path.join(folder, file)

            data.append([path, label])

df = pd.DataFrame(
    data,
    columns=["path", "label"]
)

df.to_csv(
    "dataset/metadata.csv",
    index=False
)

print(df.head())

print("Total samples:", len(df))