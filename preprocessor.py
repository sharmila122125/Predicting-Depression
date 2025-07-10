# preprocessor.py
import pandas as pd
import cloudpickle
from preprocessor_module import DepressionPreprocessor  # Ensure this matches the filename exactly

# Load data
df = pd.read_csv("train.csv")
y = df["Depression"]
X = df.drop(columns=["Depression"])

# Fit preprocessor
preprocessor = DepressionPreprocessor()
X_processed = preprocessor.fit_transform(X)
X_processed["Depression"] = y.values

# Save processed data (optional)
X_processed.to_csv("processed_train.csv", index=False)

# Save preprocessor object
with open("preprocessor.pkl", "wb") as f:
    cloudpickle.dump(preprocessor, f)

print(" preprocessor.pkl saved using Python 3.9")
