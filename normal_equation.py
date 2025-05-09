import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    """Simple Normal‑Equation Linear Regression with preprocessing."""

    def __init__(self):
        self.theta          = None
        self.cost_history   = []
        self.feature_names  = None
        self.scaler         = StandardScaler()

    # ──────────────────────────────────────────────────────────────
    # Data handling helpers
    # ──────────────────────────────────────────────────────────────

    def preprocess_data(self, X):
        """Standardise numeric cols + one‑hot categorical; return NumPy array."""
        numeric   = ["age", "bmi", "children"]
        categorical = ["sex", "smoker", "region"]

        X_numeric      = self.scaler.fit_transform(X[numeric])
        X_categorical  = pd.get_dummies(X[categorical], drop_first=True)
        X_processed    = np.hstack([X_numeric, X_categorical.values])

        # Store the column names for later printout
        self.feature_names = numeric + list(X_categorical.columns)
        return X_processed

    # ──────────────────────────────────────────────────────────────
    # Core linear algebra
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def compute_cost(X, y, theta):
        m = len(y)
        residuals = X @ theta - y
        return (residuals.T @ residuals)[0, 0] / (2 * m)

    def fit(self, X, y):
        """Solve theta once via the Normal Equation."""
        X_proc = self.preprocess_data(X)
        X_b    = np.c_[np.ones((X_proc.shape[0], 1)), X_proc]  # add bias

        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        final_cost = self.compute_cost(X_b, y, self.theta)
        self.cost_history = [final_cost]
        print(f"Final Cost (MSE/2) = {final_cost:.4f}")

    def predict(self, X):
        X_proc = self.preprocess_data(X)
        X_b    = np.c_[np.ones((X_proc.shape[0], 1)), X_proc]
        return X_b @ self.theta

    # ──────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────

    def plot_cost_history(self, out_path="cost_history.png"):
        plt.figure(figsize=(6, 4))
        plt.plot(range(len(self.cost_history)), self.cost_history, "bo-")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Cost History (Normal Equation)")
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Cost plot saved to '{out_path}'")

    def print_equation(self):
        if self.theta is None:
            print("Model has not been trained!")
            return
        print("\nRegression Equation (standardised charges):\ncharges = ", end="")
        # Intercept
        print(f"{self.theta[0, 0]:.4f}", end="")
        # Coefficients
        for name, coef in zip(self.feature_names, self.theta[1:].ravel()):
            sign = "+" if coef >= 0 else "-"
            print(f" {sign} {abs(coef):.4f}*{name}", end="")
        print()

    def print_metrics(self, X, y):
        preds = self.predict(X)
        mse   = np.mean((preds - y) ** 2)
        rmse  = np.sqrt(mse)
        print("\nModel Performance:")
        print(f"MSE  = {mse:.4f}")
        print(f"RMSE = {rmse:.4f}")

# ────────────────────────────────────────────────────────────────────
# Main script entry
# ────────────────────────────────────────────────────────────────────

def find_csv(filename="insurance[1].csv"):
    """Return path to CSV located in the same directory as this script."""
    script_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(script_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{filename}' in {script_dir}. Place the file next to the script.")
    return path

if __name__ == "__main__":
    try:
        csv_path = find_csv()
        df = pd.read_csv(csv_path)
        print("Data loaded from", csv_path)

        X = df.drop("charges", axis=1)
        y = df["charges"].values.reshape(-1, 1)
        y = (y - y.mean()) / y.std()         # standardise target for stability

        model = LinearRegression()
        model.fit(X, y)

        model.print_equation()
        model.print_metrics(X, y)
        model.plot_cost_history()

    except Exception as exc:
        print("Error:", exc)
