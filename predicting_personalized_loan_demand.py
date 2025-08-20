import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# that should be the google drive
from google.colab import drive
drive.mount('/content/drive')

DATA_DIR =" top give the path of the file where file located."

train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "Submission.csv"))

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_sub.shape)

target_col = None
for cand in ["RequestedLoanAmount", "requestedloanamount", "Target", "target"]:
    if cand in train.columns:
        target_col = cand
        break
if target_col is None:
    raise ValueError("Target column not found!")


feature_cols = [c for c in train.columns if c != target_col]
num_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

X_train_full = train[num_cols]
X_test_full = test[num_cols]
y_train_full = train[target_col]

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train_full)
X_test_imputed = imputer.transform(X_test_full)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_imputed, y_train_full, test_size=0.2, random_state=42
)

model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=400,
    min_samples_leaf=20,
    random_state=42
)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print("Validation RMSE:", rmse)

model.fit(X_train_imputed, y_train_full)

test_preds = model.predict(X_test_imputed)
test_preds = np.maximum(test_preds, 0)

submission = sample_sub.copy()
pred_col = submission.columns[-1]
submission[pred_col] = test_preds

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
submission.head(20)