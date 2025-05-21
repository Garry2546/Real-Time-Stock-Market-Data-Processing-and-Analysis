import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Data_preprocessing import X_test, y_test, target_scaler
from Training import BayesianLSTM

# Set model parameters.
n_features = X_test.shape[2]
output_length = 1
batch_size = 128

# Load the best saved model.
model = BayesianLSTM(n_features=n_features, output_length=output_length, batch_size=batch_size)
model.load_state_dict(torch.load("/Users/Garry/Year3/Third-Year-Project/model/bayesian_lstm_model.pth"))
model.eval()

# Convert test data to torch tensor.
X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)

# Obtain predictions via MC dropout.
n_experiments = 100
all_predictions = []
with torch.no_grad():
    for _ in range(n_experiments):
        preds = model(X_test_tensor).detach().numpy()  # shape: [n_samples, 1]
        all_predictions.append(preds)

all_predictions = np.array(all_predictions)  # shape: (n_experiments, n_samples, 1)
pred_mean = all_predictions.mean(axis=0).squeeze()  # Average prediction over experiments

# Inverse transform predictions.
pred_log = target_scaler.inverse_transform(pred_mean.reshape(-1, 1)).squeeze()
pred_price = np.exp(pred_log)

# Get actual prices.
y_test_np = np.array(y_test)
y_log = target_scaler.inverse_transform(y_test_np.reshape(-1, 1)).squeeze()
actual_price = np.exp(y_log)

# ----- Additional Performance Metrics -----
rmse = math.sqrt(np.mean((actual_price - pred_price)**2))
mae = np.mean(np.abs(actual_price - pred_price))
mape = np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# ----- Uncertainty Calibration Assessment -----
lower_bounds = np.percentile(all_predictions, 2.5, axis=0).squeeze()
upper_bounds = np.percentile(all_predictions, 97.5, axis=0).squeeze()
lower_bounds_log = target_scaler.inverse_transform(lower_bounds.reshape(-1, 1)).squeeze()
upper_bounds_log = target_scaler.inverse_transform(upper_bounds.reshape(-1, 1)).squeeze()
lower_bounds_price = np.exp(lower_bounds_log)
upper_bounds_price = np.exp(upper_bounds_log)

coverage = np.mean((actual_price >= lower_bounds_price) & (actual_price <= upper_bounds_price))
print(f"Coverage Probability (95% CI): {coverage*100:.2f}%")

# ----- Plotting Price Predictions with Confidence Interval -----
plt.figure(figsize=(12, 6))
plt.plot(actual_price, label="Actual Price", color="blue", linewidth=2)
plt.plot(pred_price, label="Predicted Price", color="red", linewidth=2)  # Solid red line
plt.fill_between(range(len(pred_price)), lower_bounds_price, upper_bounds_price, 
                 color='gray', alpha=0.3, label='95% Confidence Interval')
plt.title("Actual vs. Predicted Tesla Stock Price Trend with 95% CI")
plt.xlabel("Time Steps")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.savefig("/Users/Garry/Year3/Third-Year-Project/Figures/price_prediction_S.png")  # Save the plot to file
plt.close()

# ---------------------------------------------------
# Trading Simulation: Active Buy/Sell Strategy with $10,000
# ---------------------------------------------------
# Binary trading strategy:
# - If predicted price > actual price, then go fully long (invest all cash).
# - Otherwise, exit position (sell all shares to hold cash).
initial_cash = 10000.0
cash = initial_cash
shares = 0
position = "cash"  # either "cash" or "long"

portfolio_values = []

for t in range(len(actual_price)):
    current_price = actual_price[t]
    current_pred = pred_price[t]
    
    if current_pred > current_price:
        if position != "long":
            shares = cash / current_price
            cash = 0
            position = "long"
    else:
        if position == "long":
            cash = shares * current_price
            shares = 0
            position = "cash"
    
    portfolio_value = cash + shares * current_price
    portfolio_values.append(portfolio_value)

final_value = portfolio_values[-1]
profit = final_value - initial_cash

print(f"\nFinal Portfolio Value: ${final_value:.2f}")
print(f"Total Profit/Loss: ${profit:.2f}")

# ----- Plot Portfolio Value Over Time -----
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label="Portfolio Value", color="purple", linewidth=2)
plt.title("Portfolio Value Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.grid(True)
plt.savefig("/Users/Garry/Year3/Third-Year-Project/Figures/portfolio_value_S.png")  
plt.close()
