# train_momentum_model old.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load labeled dataset
df = pd.read_csv("../ml_dataset.csv", low_memory=False)

# Drop any remaining NaNs
df.dropna(inplace=True)

# Feature columns used for training
features = [
    "ema_gap",
    "rsi",
    "volume_ratio",
    "bb_width",
    "above_ema_50"
]

# Inspect for debugging (optional)
print(df.dtypes)
print(df["above_ema_50"].unique())

# Convert 'above_ema_50' to int, safely
df["above_ema_50"] = pd.to_numeric(df["above_ema_50"], errors='coerce').fillna(0).astype(int)

# Input and target
X = df[features]
y = df["label"]

# Keep only labels that appear at least twice
label_counts = y.value_counts()
valid_labels = label_counts[label_counts >= 2].index
mask = y.isin(valid_labels)

X = X[mask]
y = y[mask]

# Now safe to stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train model
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",  # 👈 This is the key fix
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Save the model
model_path = "../momentum_model.pkl"
joblib.dump(clf, model_path)
print("Label distribution:")
print(y.value_counts(normalize=True))
print(f"\nModel saved to {model_path}")

