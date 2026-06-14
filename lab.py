import matplotlib.pyplot as plt
import numpy as np

import main


# Current single-run experiment.
num = 20000
m = 1
USE_REAL_PS = False
results = {}

a, b, c, d, e, f, g = main.main(0.01, 100, num, m, 42, 1, 0.01, USE_REAL_PS=USE_REAL_PS)

results[m] = {
    "N_tilde": c,
    "Pi": d,
    "Loads": a,
    "SuccessRate": b,
    "Reward": f,
    "TruePi": e,
    "EpisodeHistory": g,
}


plt.figure(figsize=(10, 6))
plt.axhline(y=num, color="black", linestyle="--", label=f"True N ({num})", alpha=0.6)
plt.plot(
    range(len(results[m]["N_tilde"])),
    results[m]["N_tilde"],
    label="MODE 1: RL + Backoff",
    color="blue",
    linewidth=1.5,
)

plt.title("Population Estimation Convergence (N_tilde) - MODE 1 Only")
plt.xlabel("Time Slot (n)")
plt.ylabel("Estimated Population")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(
    range(len(results[m]["Reward"])),
    results[m]["Reward"],
    label="Reward (Negative Variance)",
    color="blue",
    linewidth=1.2,
)

plt.title("Reward Variation over Time Slots")
plt.xlabel("Time Slot (n)")
plt.ylabel("Reward (-Var)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# State distribution over time for all pi_n states.
pi_history = np.asarray(results[m]["Pi"], dtype=float)
estimated_active_pi = np.asarray(results[m]["TruePi"], dtype=float)
estimated_pi = np.concatenate(([max(0.0, 1.0 - np.sum(estimated_active_pi))], estimated_active_pi))

if pi_history.size > 0:
    if pi_history.ndim == 1:
        pi_history = pi_history.reshape(1, -1)

    record_interval = 10
    time_slots = np.arange(pi_history.shape[0]) * record_interval
    state_count = pi_history.shape[1]

    plt.figure(figsize=(10, 6))
    for state_idx in range(state_count):
        state_label = "Idle" if state_idx == 0 else f"State {state_idx}"
        line = plt.plot(
            time_slots,
            pi_history[:, state_idx],
            label=state_label,
            linewidth=1.5,
        )[0]

        if state_idx < estimated_pi.size:
            plt.axhline(
                y=estimated_pi[state_idx],
                color=line.get_color(),
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                label=f"Estimated {state_label}",
            )

    plt.title(r"Convergence of State Distributions ($\pi_n$)")
    plt.xlabel("Time Slot (n)")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No pi history was recorded; skip state distribution plot.")


# Precomputed p_s error over time.
ps_history = results[m]["EpisodeHistory"].get("ps_history", [])
if len(ps_history) > 0:
    ps_time = np.array([item["time_slot"] for item in ps_history])
    ps_error = np.array([item["error"] for item in ps_history])
    real_ps = np.array([item["real"] for item in ps_history])
    precomputed_ps = np.array([item["precomputed"] for item in ps_history])

    plt.figure(figsize=(10, 6))
    plt.axhline(y=0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.plot(ps_time, ps_error, label=r"Error: real $p_s$ - precomputed $p_s$", color="red", linewidth=1.3)
    plt.title(r"Precomputed $p_s$ Error Over Time")
    plt.xlabel("Time Slot (n)")
    plt.ylabel(r"$p_s$ Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(ps_time, real_ps, label=r"Real $p_s$", linewidth=1.3)
    plt.plot(ps_time, precomputed_ps, label=r"Precomputed $p_s$", linewidth=1.3)
    plt.title(r"Real vs. Precomputed $p_s$")
    plt.xlabel("Time Slot (n)")
    plt.ylabel(r"$p_s$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No p_s history was recorded; skip p_s error plots.")

print("--- Test Complete ---")
print(f"Packet Loss Rate: {results[m]['SuccessRate']:.4f}")


"""
# Epsilon sweep for convex group satellite selection.
import main as simulator_main

EPSILON_VALUES = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
EPSILON_RHO = 0.005
EPSILON_SECONDS = 30
EPSILON_NUM_UE = 10000
EPSILON_MODE = 1
EPSILON_SEED = 42
EPSILON_EPOCHS = 1

epsilon_results = []

for eps in EPSILON_VALUES:
    print(f"\nRunning epsilon sweep: epsilon={eps}")
    avg_throughput, plr, n_history, actual_pi, observe_pi, reward_history, epo_history = simulator_main.main(
        EPSILON_RHO,
        EPSILON_SECONDS,
        EPSILON_NUM_UE,
        EPSILON_MODE,
        EPSILON_SEED,
        EPSILON_EPOCHS,
        IMBALANCE_EPSILON=eps,
    )
    epsilon_results.append({
        "epsilon": eps,
        "plr": plr,
        "throughput": avg_throughput,
        "reward": np.mean(reward_history) if len(reward_history) > 0 else np.nan,
    })

epsilon_labels = [str(item["epsilon"]) for item in epsilon_results]
epsilon_plr = [item["plr"] for item in epsilon_results]
epsilon_throughput = [item["throughput"] for item in epsilon_results]
epsilon_reward = [item["reward"] for item in epsilon_results]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Convex Selection Epsilon Sweep", fontsize=16)

axes[0].plot(epsilon_labels, epsilon_plr, marker="o", color="#3498db")
axes[0].set_title("Packet Loss Rate")
axes[0].set_xlabel("Imbalance epsilon")
axes[0].set_ylabel("PLR")
axes[0].grid(True, linestyle=":", alpha=0.6)

axes[1].plot(epsilon_labels, epsilon_throughput, marker="o", color="#27ae60")
axes[1].set_title("Average Throughput")
axes[1].set_xlabel("Imbalance epsilon")
axes[1].set_ylabel("Packets / Second")
axes[1].grid(True, linestyle=":", alpha=0.6)

axes[2].plot(epsilon_labels, epsilon_reward, marker="o", color="#f39c12")
axes[2].set_title("Average Reward")
axes[2].set_xlabel("Imbalance epsilon")
axes[2].set_ylabel("Reward")
axes[2].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\n--- Epsilon Sweep Complete ---")
for item in epsilon_results:
    print(
        f"epsilon={item['epsilon']}: "
        f"PLR={item['plr']:.4f}, "
        f"throughput={item['throughput']:.2f}, "
        f"reward={item['reward']:.4f}"
    )
"""
