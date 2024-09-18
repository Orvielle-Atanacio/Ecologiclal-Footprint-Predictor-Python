import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import tkinter as tk
from tkinter import ttk

# Load the data
file_path = r"D:\\cpe\\InterSem\\ELECTIVE 3\\Crop_recommendation.xlsx"
data = pd.read_excel(file_path)

# Check for missing values
missing_values = data.isnull().sum()
print(f"Missing values:\n{missing_values}")

# Extract features and target variable
X = data.drop(columns=['yield'])
y = data['yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Multiple Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# 2. Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# 3. Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)


# Generate random values for prediction
def generate_random_values():
    random_values = {}
    for column in X.columns:
        random_values[column] = np.random.uniform(X[column].min(), X[column].max())
    return random_values


# Predict yield based on random inputs and measure prediction time
def predict_yield(input_values):
    user_data = pd.DataFrame([input_values])

    # Standardize the input data
    user_data_scaled = scaler.transform(user_data)

    # Measure prediction times
    start_time = time.time()
    yield_lr = lr_model.predict(user_data_scaled)[0]
    lr_pred_time = time.time() - start_time

    start_time = time.time()
    yield_rf = rf_model.predict(user_data_scaled)[0]
    rf_pred_time = time.time() - start_time

    start_time = time.time()
    yield_dt = dt_model.predict(user_data_scaled)[0]
    dt_pred_time = time.time() - start_time

    return (yield_lr, lr_pred_time), (yield_rf, rf_pred_time), (yield_dt, dt_pred_time)


# Tkinter UI setup
root = tk.Tk()
root.title("Crop Yield Prediction")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(frame, text="Crop Yield Prediction").grid(row=0, column=0, columnspan=2)

results = {
    "Input Values": tk.StringVar(),
    "Linear Regression": tk.StringVar(),
    "Random Forest": tk.StringVar(),
    "Decision Tree": tk.StringVar(),
    "LR Accuracy": tk.StringVar(),
    "RF Accuracy": tk.StringVar(),
    "DT Accuracy": tk.StringVar(),
    "LR Time": tk.StringVar(),
    "RF Time": tk.StringVar(),
    "DT Time": tk.StringVar(),
    "Actual Yield": tk.StringVar(),
    "Overall Accuracy": tk.StringVar()
}

# Labels for Input Values and Predictions
ttk.Label(frame, text="Input Values:").grid(row=1, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["Input Values"]).grid(row=1, column=1, sticky=tk.W)

ttk.Label(frame, text="Predicted Yield by Linear Regression:").grid(row=2, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["Linear Regression"]).grid(row=2, column=1, sticky=tk.W)

ttk.Label(frame, text="Predicted Yield by Random Forest:").grid(row=3, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["Random Forest"]).grid(row=3, column=1, sticky=tk.W)

ttk.Label(frame, text="Predicted Yield by Decision Tree:").grid(row=4, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["Decision Tree"]).grid(row=4, column=1, sticky=tk.W)

# Labels for Model Accuracy
ttk.Label(frame, text="Linear Regression Accuracy (MSE, R²):").grid(row=5, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["LR Accuracy"]).grid(row=5, column=1, sticky=tk.W)

ttk.Label(frame, text="Random Forest Accuracy (MSE, R²):").grid(row=6, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["RF Accuracy"]).grid(row=6, column=1, sticky=tk.W)

ttk.Label(frame, text="Decision Tree Accuracy (MSE, R²):").grid(row=7, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["DT Accuracy"]).grid(row=7, column=1, sticky=tk.W)

# Labels for Prediction Times
ttk.Label(frame, text="Linear Regression Prediction Time (seconds):").grid(row=8, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["LR Time"]).grid(row=8, column=1, sticky=tk.W)

ttk.Label(frame, text="Random Forest Prediction Time (seconds):").grid(row=9, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["RF Time"]).grid(row=9, column=1, sticky=tk.W)

ttk.Label(frame, text="Decision Tree Prediction Time (seconds):").grid(row=10, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["DT Time"]).grid(row=10, column=1, sticky=tk.W)

# Label for Actual Yield
ttk.Label(frame, text="Actual Yield:").grid(row=11, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["Actual Yield"]).grid(row=11, column=1, sticky=tk.W)

# Label for Overall Accuracy
ttk.Label(frame, text="Overall Accuracy:").grid(row=12, column=0, sticky=tk.W)
ttk.Label(frame, textvariable=results["Overall Accuracy"]).grid(row=12, column=1, sticky=tk.W)

# Dropdown menu to select method
method = tk.StringVar(value="Random")
ttk.Label(frame, text="Select Input Method:").grid(row=13, column=0, sticky=tk.W)
ttk.OptionMenu(frame, method, "Random", "Sequential").grid(row=13, column=1, sticky=tk.W)


# Button to switch modes
def switch_mode():
    if method.get() == "Random":
        method.set("Sequential")
    else:
        method.set("Random")


ttk.Button(frame, text="Switch Mode", command=switch_mode).grid(row=14, column=0, columnspan=2)

# Global index for sequential traversal
index = 0
actual_yields = []
correct_predictions_lr = 0
correct_predictions_rf = 0
correct_predictions_dt = 0


# Function to update predictions
def update_predictions():
    global index, correct_predictions_lr, correct_predictions_rf, correct_predictions_dt

    if method.get() == "Random":
        input_values = generate_random_values()
        actual_yield = None
    else:
        input_values = X.iloc[index].to_dict()
        actual_yield = y.iloc[index]
        index += 1

    lr_result, rf_result, dt_result = predict_yield(input_values)

    results["Input Values"].set(str(input_values))
    results["Linear Regression"].set(f"{lr_result[0]:.2f}")
    results["Random Forest"].set(f"{rf_result[0]:.2f}")
    results["Decision Tree"].set(f"{dt_result[0]:.2f}")

    # Update accuracy metrics
    results["LR Accuracy"].set(f"MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}")
    results["RF Accuracy"].set(f"MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")
    results["DT Accuracy"].set(f"MSE: {mse_dt:.2f}, R²: {r2_dt:.2f}")

    # Update prediction times
    results["LR Time"].set(f"{lr_result[1]:.4f}")
    results["RF Time"].set(f"{rf_result[1]:.4f}")
    results["DT Time"].set(f"{dt_result[1]:.4f}")

    # Update actual yield if in sequential mode
    if actual_yield is not None:
        results["Actual Yield"].set(f"{actual_yield:.2f}")
        actual_yields.append(actual_yield)

        correct_predictions_lr += int(np.isclose(lr_result[0], actual_yield, atol=0.1))
        correct_predictions_rf += int(np.isclose(rf_result[0], actual_yield, atol=0.1))
        correct_predictions_dt += int(np.isclose(dt_result[0], actual_yield, atol=0.1))

        accuracy_display = (
            f"Traverse {index}:\n"
            f"LR: {correct_predictions_lr}/{len(actual_yields)} ({(correct_predictions_lr / len(actual_yields)) * 100:.2f}%)\n"
            f"RF: {correct_predictions_rf}/{len(actual_yields)} ({(correct_predictions_rf / len(actual_yields)) * 100:.2f}%)\n"
            f"DT: {correct_predictions_dt}/{len(actual_yields)} ({(correct_predictions_dt / len(actual_yields)) * 100:.2f}%)"
        )
        results["Overall Accuracy"].set(accuracy_display)
    else:
        results["Actual Yield"].set("N/A")

    # Pause for 10 seconds after reaching the end of the dataset in sequential mode
    if method.get() == "Sequential" and index >= 2200:
        index = 0
        correct_predictions_lr = 0
        correct_predictions_rf = 0
        correct_predictions_dt = 0
        actual_yields.clear()
        root.after(10000, update_predictions)
    else:
        root.after(1, update_predictions)


# Initial call to update predictions
update_predictions()

# Start Tkinter main loop
root.mainloop()
