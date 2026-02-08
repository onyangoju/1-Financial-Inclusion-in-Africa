# =========================
# STEP 1: LOAD & EXPLORE DATA
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")
sample_submission = pd.read_csv("SampleSubmission.csv")
variable_definitions = pd.read_csv("VariableDefinitions.csv")

# Quick look
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)

print("\nMissing values in train:")
print(train.isnull().sum())

# Target distribution
sns.countplot(x="bank_account", data=train)
plt.title("Distribution of Bank Account Ownership")
plt.tight_layout()
plt.savefig("bank_account_distribution.png")
plt.close()


# =========================
# STEP 2: DATA PREPROCESSING
# =========================

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Drop ID column (not predictive)
train = train.drop(columns=["uniqueid"])
test_ids = test["uniqueid"]  # save original IDs for reference
test = test.drop(columns=["uniqueid"])

# Separate features and target
y = train["bank_account"]
X = train.drop(columns=["bank_account"])

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "string"]).columns
numerical_cols = X.select_dtypes(exclude=["object", "string"]).columns

print("\nCategorical columns:", list(categorical_cols))
print("Numerical columns:", list(numerical_cols))

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)
print("\nPreprocessing pipeline created successfully!")


# =========================
# STEP 3: MODEL TRAINING
# =========================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# Full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train model
pipeline.fit(X_train, y_train)

# Validate
y_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {accuracy:.4f}")


# =========================
# STEP 4: CREATE ZINDI-SAFE SUBMISSION
# =========================

import pandas as pd

# Load full test data (to get country and other features)
test_full = pd.read_csv("Test.csv")

# Keep only the features used in training (all except 'uniqueid')
X_test_full = test_full.drop(columns=["uniqueid"])

# Predict on test set
test_predictions = pipeline.predict(X_test_full)

# Convert predictions to 0/1 if they are 'Yes'/'No'
if test_predictions.dtype.kind in {'U', 'O'}:  # string type
    test_predictions_int = (test_predictions == 'Yes').astype(int)
else:
    test_predictions_int = test_predictions.astype(int)

# Build submission with 'uniqueid x country'
submission = pd.DataFrame({
    "uniqueid": test_full["uniqueid"] + " x " + test_full["country"],
    "bank_account": test_predictions_int
})

# Save submission
submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv with countries created successfully!")
print(submission.head())
print("Submission shape:", submission.shape)
