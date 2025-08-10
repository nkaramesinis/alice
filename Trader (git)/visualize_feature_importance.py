# visualize_feature_importance.py

import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load model
model = joblib.load("momentum_model.pkl")

# Define feature names (must match training order)
feature_names = ["ema_gap", "rsi", "volume_ratio", "bb_width", "above_ema_50"]

# Get feature importances
importances = model.feature_importances_

# Create DataFrame for sorting and plotting
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(8, 5))
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importances - ML Momentum Strategy")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
