from forecasting.baselines.synthetic_data import (
    generate_synthetic_states,
)


states = generate_synthetic_states(
    num_states=1000,
    seed=42,
)

print("States generated:", len(states))

print("\nFirst state:")
print(states[0])

print("\nSecond state:")
print(states[1])

print("\nLast state:")
print(states[-1])