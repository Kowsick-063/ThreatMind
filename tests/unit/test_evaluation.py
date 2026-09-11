from forecasting.baselines.dataset import build_transition_dataset
from forecasting.baselines.random_forest import NetworkStateRandomForest
from forecasting.baselines.synthetic_data import generate_synthetic_states
from forecasting.baselines.evaluation import (
    evaluate_predictions,
    print_evaluation_report,
)


states = generate_synthetic_states(
    num_states=1000,
    seed=42,
)

X, y = build_transition_dataset(states)

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]


model = NetworkStateRandomForest(
    n_estimators=100,
    random_state=42,
)

model.train(X_train, y_train)

predictions = model.predict(X_test)

results = evaluate_predictions(
    y_test,
    predictions,
)

print_evaluation_report(results)