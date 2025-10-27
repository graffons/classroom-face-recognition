import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# --- Load embeddings and labels ---
DATA_DIR = Path("models")
X = np.load(DATA_DIR / "embeddings.npy")
y = np.load(DATA_DIR / "labels.npy")

print(f"Loaded {X.shape[0]} embeddings with {X.shape[1]} features each.")
print(f"Unique classes: {np.unique(y)}")

# --- Split into train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --- Define models ---
models = {
    "SVM (linear)": SVC(kernel='linear', probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3)
}

results = {}

# --- Train and evaluate each model ---
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# --- Compare models ---
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
print("\n📊 Model Comparison:\n", results_df)

# --- Save best model ---
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
model_path = DATA_DIR / f"best_model_{best_model_name.replace(' ', '_')}.joblib"
joblib.dump(best_model, model_path)
print(f"\n✅ Best model: {best_model_name} (Accuracy={results[best_model_name]:.3f})")
print(f"Saved model to: {model_path}")
