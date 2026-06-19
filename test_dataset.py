from src.dataset_loader import AudioDataset

dataset = AudioDataset("dataset/metadata.csv")

print("Dataset size:", len(dataset))

x, y = dataset[0]

print("Input shape:", x.shape)

print("Label:", y)