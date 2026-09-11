from forecasting.baselines.synthetic_data import (
    generate_synthetic_states,
)
from forecasting.baselines.dataset import (
    build_transition_dataset,
)


states = generate_synthetic_states(
    num_states=1000,
    seed=42,
)

X, y = build_transition_dataset(states)

print("States:", len(states))
print("Training samples:", len(X))
print("Features:", len(X[0]))
print("Targets:", len(y[0]))