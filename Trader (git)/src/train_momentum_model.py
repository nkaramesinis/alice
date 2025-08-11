# train_momentum_model.py

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

# Ensure 'above_ema_50' is numeric
df["above_ema_50"] = pd.to_numeric(df["above_ema_50"], errors='coerce').fillna(0).astype(int)

# Input and target
X = df[features]
y = df["label"]

# Drop labels that appear less than twice (for safety)
label_counts = y.value_counts()
valid_labels = label_counts[label_counts >= 2].index
mask = y.isin(valid_labels)
X = X[mask]
y = y[mask]

# Combine into a single DataFrame for balancing
df_balanced = X.copy()
df_balanced["label"] = y

# Split into winners and losers
winners = df_balanced[df_balanced["label"] == 1]
losers = df_balanced[df_balanced["label"] == 0].sample(n=len(winners), random_state=42)

# Combine balanced set
df_final = pd.concat([winners, losers]).sample(frac=1, random_state=42)
X = df_final[features]
y = df_final["label"]

# Print label distribution after balancing
print("\n✅ Rebalanced label distribution:")
print(y.value_counts(normalize=True))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Initialize and train model with class balancing
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Save the model
model_path = "../momentum_model.pkl"
joblib.dump(clf, model_path)
print(f"\nModel saved to {model_path}")