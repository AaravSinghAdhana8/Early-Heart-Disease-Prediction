from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATHS = [
    PROJECT_ROOT / "heart.csv" / "heart.csv",
    PROJECT_ROOT / "heart.csv",
]

DATASET_PATH = next((path for path in DATASET_PATHS if path.exists() and path.is_file()), None)
if DATASET_PATH is None:
    raise FileNotFoundError(
        "Could not find heart.csv. Make sure the dataset exists in the project root or inside the heart.csv folder."
    )

MODEL_PATH = PROJECT_ROOT / "model.pkl"

# Load dataset
data = pd.read_csv(DATASET_PATH)

# Show first 5 rows
print(data.head())

# Features and Target
X = data.drop("target", axis=1)
y = data["target"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Save Model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully as {MODEL_PATH}")