from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from forecasting.baselines.dataset import (
    build_transition_dataset,
)
from forecasting.baselines.random_forest import (
    NetworkStateRandomForest,
)
from forecasting.baselines.synthetic_data import (
    generate_synthetic_states,
)


# Generate development data
states = generate_synthetic_states(
    num_states=1000,
    seed=42,
)

# Build S(t) -> S(t+1) transitions
X, y = build_transition_dataset(states)

# Time-aware split
split_index = int(len(X) * 0.8)

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create model
model = NetworkStateRandomForest(
    n_estimators=100,
    random_state=42,
)

# Train
model.train(X_train, y_train)

print("Model training complete.")


# Predict
predictions = model.predict(X_test)

print("Prediction complete.")


# Evaluate
mae = mean_absolute_error(
    y_test,
    predictions,
)

mse = mean_squared_error(
    y_test,
    predictions,
)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")


# Show one prediction
print("\nActual next state:")
print(y_test[0])

print("\nPredicted next state:")
print(predictions[0])