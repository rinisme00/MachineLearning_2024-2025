import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# GPU Check
def check_gpu():
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print("GPU is available! Using CUDA.")
        device = torch.device("cuda")
    else:
        print("GPU is not available, using CPU.")
        device = torch.device("cpu")
    return device
device = check_gpu()

# Load dataset
df = pd.read_csv('insurance[1].csv')

df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first = True)

df['x0'] = 1

feature_cols = ['x0', 'age', 'bmi', 
                'children', 'sex_male', 'smoker_yes', 
                'region_northwest', 'region_southeast', 'region_southwest']
X = df[feature_cols]
Y = df['charges']

# Save statistics for later use:
numeric_cols = ['age', 'bmi', 'children']
num_means = {c: X[c].mean() for c in numeric_cols}
num_stds = {c: X[c].std() for c in numeric_cols}
target_mean = Y.mean()
target_std = Y.std()

# Standardize the target variable
Y = (Y - Y.mean()) / Y.std()

# Standardize features (excluding the intercept and binary variables)
for col in ['age', 'bmi', 'children']:
    X[col] = (X[col] - X[col].mean()) / X[col].std()

# Ensure intercept and binary variables remain unchanged
X['x0'] = 1
for col in ['sex_male', 'smoker_yes', 'region_northwest',
            'region_southeast', 'region_southwest']:
    X[col] = X[col].astype(int)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42, shuffle=True)

# Keep the training set for gradient descent
X = X_train.reset_index(drop=True)
Y = Y_train.reset_index(drop=True)

# Convert features and target to torch tensors and move to the selected device
X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y.values, dtype=torch.float32).to(device).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).to(device).unsqueeze(1)

# Initialize theta as a (n_features x 1) tensor with gradients enabled
theta = torch.randn((X_tensor.shape[1], 1), device = device, requires_grad=True)

# Set hyperparameters
alpha = 0.01 # Learning rate
num_iterations = 1338
cost_history = []
r2_history = []

# Gradient Descent Loop
for iteration in range(num_iterations):
    Y_hat = torch.matmul(X_tensor, theta)
    
    # Training loss (MSE)
    cost = torch.mean((Y_hat - Y_tensor) ** 2)
    cost_history.append(cost.item())

    # Training accuracy (R²)
    with torch.no_grad():
        r2 = r2_score(Y_tensor.cpu().numpy(), Y_hat.cpu().numpy())
    r2_history.append(r2)
    
    cost.backward()
    with torch.no_grad():
        theta -= alpha * theta.grad
        theta.grad.zero_()
    print(f"Iteration {iteration + 1}/{num_iterations}, Cost: {cost.item():}, R²: {r2:}")

iterations = list(range(num_iterations))
gd_iterations_df = pd.DataFrame({'iteration': iterations, 'cost': cost_history})

print("Final estimate of theta:")
print(theta)

# Evaluating
with torch.no_grad():
    Y_pred = torch.matmul(X_tensor, theta).cpu().numpy()
Y_truth = Y_tensor.cpu().numpy()

# Compute evaluation metrics
mse = mean_squared_error(Y_truth, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_truth, Y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:}")
print(f"Root Mean Squared Error (RMSE): {rmse:}")
print(f"R² Score: {r2:}")

# Test set metrics
with torch.no_grad():
    Y_test_pred = torch.matmul(X_test_tensor, theta).cpu().numpy()
test_mse = mean_squared_error(Y_test_tensor.cpu().numpy(), Y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(Y_test_tensor.cpu().numpy(), Y_test_pred)

print("\nTest Set Evaluation:")
print(f"Test Mean Squared Error (MSE): {test_mse:}")
print(f"Test Root Mean Squared Error (RMSE): {test_rmse:}")
print(f"Test R² Score: {test_r2:}")

# Plot the cost history to visualize convergence
plt.plot(gd_iterations_df['iteration'], gd_iterations_df['cost'])
plt.xlabel('Number of Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent Convergence')
plt.savefig('gradient_descent_insurance.png')
print("Plot saved as 'gradient_descent_insurance.png'")

# Plot training loss and accuracy
fig, ax1 = plt.subplots()
ax1.plot(cost_history, color="tab:blue")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("MSE loss", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(r2_history, color="tab:red")
ax2.set_ylabel("R² accuracy", color="tab:red")
plt.title("Training Loss vs Accuracy")
plt.tight_layout()
plt.savefig("loss_and_accuracy.png")
print("Plot saved as 'loss_and_accuracy.png'")

# Demo
def predict_single(record: dict) -> float:
    """Predict the insurance charges for a single record."""
    row = pd.DataFrame([record])

    # One-hot encode
    row = pd.get_dummies(row, columns = ['sex', 'smoker', 'region'], drop_first = True)
    for col in ['sex_male', 'smoker_yes', 
                'region_northwest','region_southeast', 'region_southwest']:
        if col not in row.columns:
            row[col] = 0
    
    # Standardize the features
    for col in numeric_cols:
        row[col] = (row[col] - num_means[col]) / num_stds[col]

    # Intercept + reorder
    row['x0'] = 1
    row = row[feature_cols]

    tensor = torch.tensor(row.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_std = (tensor @ theta).item() # Standardized unit
    return pred_std * target_std + target_mean # Charges

# Quick demo (Comment out if not needed)
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
