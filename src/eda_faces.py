import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path

# Load embeddings and labels
DATA_DIR = Path("models")

embeddings = np.load(DATA_DIR / "embeddings.npy")
labels = np.load(DATA_DIR / "labels.npy")
meta = pd.read_csv(DATA_DIR / "meta.csv")

print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels.shape)
print(meta.head())

# --- 1️⃣ Basic class distribution ---
df = pd.DataFrame({'label': labels})
count_per_class = df['label'].value_counts()
print("\n📊 Samples per person:\n", count_per_class)

plt.figure(figsize=(6,4))
sns.barplot(x=count_per_class.index, y=count_per_class.values, palette='viridis')
plt.title("Samples per Person")
plt.xlabel("Person")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --- 2️⃣ PCA visualization ---
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

pca_df = pd.DataFrame({
    'PC1': embeddings_2d[:, 0],
    'PC2': embeddings_2d[:, 1],
    'label': labels
})

plt.figure(figsize=(6,5))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', s=80, palette='Set2')
plt.title("PCA Visualization of Face Embeddings")
plt.tight_layout()
plt.show()
