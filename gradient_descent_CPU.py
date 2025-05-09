import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ──────────────────────────────────────────────────────────────
# 1. Load data and one‑hot encode
# ──────────────────────────────────────────────────────────────
CSV = "insurance[1].csv"
df = pd.read_csv(CSV)

df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

df["x0"] = 1  # intercept term

feature_cols = [
    "x0", "age", "bmi", "children",
    "sex_male", "smoker_yes",
    "region_northwest", "region_southeast", "region_southwest",
]
X = df[feature_cols].copy()
Y = df["charges"].copy()

# ──────────────────────────────────────────────────────────────
# 2. Save stats for later inverse‑transform / single‑predict
# ──────────────────────────────────────────────────────────────
numeric_cols = ["age", "bmi", "children"]
num_means = {c: X[c].mean() for c in numeric_cols}
num_stds  = {c: X[c].std()  for c in numeric_cols}

target_mean = Y.mean()
target_std  = Y.std()

# z‑score the target
Y_std = (Y - target_mean) / target_std

# z‑score numeric features
for col in numeric_cols:
    X[col] = (X[col] - num_means[col]) / num_stds[col]

# ensure dummy types
for col in ["sex_male", "smoker_yes",
            "region_northwest", "region_southeast", "region_southwest"]:
    X[col] = X[col].astype(int)

# ──────────────────────────────────────────────────────────────
# 3. Train‑test split (70/30)
# ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X.values, Y_std.values.reshape(-1, 1), test_size=0.30, random_state=42, shuffle=True)

m, n = X_train.shape

# ──────────────────────────────────────────────────────────────
# 4. Hyper‑params & initialisation
# ──────────────────────────────────────────────────────────────
np.random.seed(42)
alpha = 0.01
num_iterations = 1338

theta = np.random.randn(n, 1)  # parameter vector
cost_history = []
r2_history   = []

# ──────────────────────────────────────────────────────────────
# 5. Gradient‑descent loop (full batch)
# ──────────────────────────────────────────────────────────────
for it in range(1, num_iterations + 1):
    y_hat = X_train @ theta            # predictions
    error = y_hat - y_train

    mse  = np.mean(error ** 2)
    cost_history.append(mse)
    r2   = r2_score(y_train, y_hat)
    r2_history.append(r2)

    grad = (2 / m) * X_train.T @ error  # gradient of MSE
    theta -= alpha * grad               # update

    print(f"Iter {it:}/{num_iterations} | MSE={mse:} | R²={r2:}")

# ──────────────────────────────────────────────────────────────
# 6. Training & test metrics
# ──────────────────────────────────────────────────────────────
train_rmse = np.sqrt(cost_history[-1])
print("\nTRAIN metrics:")
print(f"MSE  = {cost_history[-1]:}")
print(f"RMSE = {train_rmse:}")
print(f"R²   = {r2_history[-1]:}")

# Test set
y_test_pred = X_test @ theta

test_mse  = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2   = r2_score(y_test, y_test_pred)

print("\nTEST metrics:")
print(f"MSE  = {test_mse:}")
print(f"RMSE = {test_rmse:}")
print(f"R²   = {test_r2:}")

# ──────────────────────────────────────────────────────────────
# 7. Plots
# ──────────────────────────────────────────────────────────────
plt.figure()
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("MSE loss")
plt.title("Gradient Descent Convergence (NumPy)")
plt.tight_layout()
plt.savefig("gradient_descent_insurance.png")
print("Saved 'gradient_descent_insurance.png'")

fig, ax1 = plt.subplots()
ax1.plot(cost_history, color="tab:blue")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("MSE loss", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(r2_history, color="tab:red")
ax2.set_ylabel("R²", color="tab:red")
ax2.set_ylim(0, 1)
plt.title("Training Loss vs Accuracy (NumPy)")
plt.tight_layout()
plt.savefig("loss_and_accuracy.png")
print("Saved 'loss_and_accuracy.png'")

# ──────────────────────────────────────────────────────────────
# 8. Single‑record predictor
# ──────────────────────────────────────────────────────────────

def predict_single(record: dict) -> float:
    row = pd.DataFrame([record])
    row = pd.get_dummies(row, columns=["sex", "smoker", "region"], drop_first=True)
    for col in ["sex_male", "smoker_yes", "region_northwest",
                "region_southeast", "region_southwest"]:
        if col not in row:
            row[col] = 0
    for col in numeric_cols:
        row[col] = (row[col] - num_means[col]) / num_stds[col]
    row["x0"] = 1
    row = row[feature_cols]
    pred_std = float(row.values @ theta)
    return pred_std * target_std + target_mean

# quick demo
example = {
    "age": 28,
    "sex": "female",
    "bmi": 26.0,
    "children": 1,
    "smoker": "no",
    "region": "southeast"
}
print("\nExample record:", example)
print("Predicted insurance charges: $", round(predict_single(example), 2)) 
