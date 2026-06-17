import matplotlib.pyplot as plt
import numpy as np

import main

RUN_RHO_SWEEP = True
epsilon_sweep = False

if RUN_RHO_SWEEP:
    NUM_UE = 10000
    SECONDS = 100
    SEED = 42
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    RHO_VALUES = np.array([0.005, 0.01, 0.015, 0.02, 0.025])
    MODES = [
        ([1, 1], "Proposed"),
        ([1, 2], "Dynamic ACB"),
        ([1, 3], "State-aware P-ACB"),
    ]

    # Backoff settings 2 and 3 are ACB baselines; all other experiment parameters are
    # kept identical to the proposed setting so the PLR curves isolate the backoff controller.
    rho_results = {label: [] for _, label in MODES}
    for mode, label in MODES:
        for rho in RHO_VALUES:
            print(f"\nRunning PLR rho sweep: {label}, rho={rho}")
            avg_throughput, plr, n_history, actual_pi, observe_pi, load_imbalance_history, run_history = main.main(
                rho,
                SECONDS,
                NUM_UE,
                mode,
                SEED,
                IMBALANCE_EPSILON,
                USE_REAL_PS=USE_REAL_PS,
            )
            final_n_estimate = n_history[-1] if len(n_history) > 0 else np.nan
            rho_results[label].append({
                "rho": rho,
                "plr": plr,
                "throughput": avg_throughput,
                "average_delay_ms": run_history.get("average_delay_ms", np.nan),
                "final_n_estimate": final_n_estimate,
            })

    plt.figure(figsize=(10, 6))
    for _, label in MODES:
        rho_axis = np.array([item["rho"] for item in rho_results[label]])
        plr_values = np.array([item["plr"] for item in rho_results[label]])
        plt.plot(rho_axis, plr_values, marker="o", linewidth=1.6, label=label)
    plt.title(r"PLR Comparison under Different $\rho$")
    plt.xlabel(r"Arrival probability $\rho$")
    plt.ylabel("Packet Loss Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for _, label in MODES:
        rho_axis = np.array([item["rho"] for item in rho_results[label]])
        throughput_values = np.array([item["throughput"] for item in rho_results[label]])
        plt.plot(rho_axis, throughput_values, marker="o", linewidth=1.6, label=label)
    plt.title(r"Throughput Comparison under Different $\rho$")
    plt.xlabel(r"Arrival probability $\rho$")
    plt.ylabel("Average Throughput (packets/second)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for _, label in MODES:
        rho_axis = np.array([item["rho"] for item in rho_results[label]])
        delay_values = np.array([item["average_delay_ms"] for item in rho_results[label]])
        plt.plot(rho_axis, delay_values, marker="o", linewidth=1.6, label=label)
    plt.title(r"AverageDelay Comparison under Different $\rho$")
    plt.xlabel(r"Arrival probability $\rho$")
    plt.ylabel("AverageDelay (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Rho Sweep Complete ---")
    for _, label in MODES:
        for item in rho_results[label]:
            print(
                f"{label}, rho={item['rho']:.4f}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"avg_delay_ms={item['average_delay_ms']:.2f}, "
                f"final_N={item['final_n_estimate']:.2f}"
            )
    raise SystemExit

if epsilon_sweep:
    # Epsilon sweep for convex group satellite selection.
    EPSILON_VALUES = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    EPSILON_RHO = 0.01
    EPSILON_SECONDS = 100
    EPSILON_NUM_UE = 10000
    EPSILON_MODE = [1, 1]
    EPSILON_SEED = 42
    epsilon_results = []

    for eps in EPSILON_VALUES:
        print(f"\nRunning epsilon sweep: epsilon={eps}")
        avg_throughput, plr, n_history, actual_pi, observe_pi, load_imbalance_history, run_history = main.main(
            EPSILON_RHO,
            EPSILON_SECONDS,
            EPSILON_NUM_UE,
            EPSILON_MODE,
            EPSILON_SEED,
            IMBALANCE_EPSILON=eps,
        )
        epsilon_results.append({
            "epsilon": eps,
            "plr": plr,
            "throughput": avg_throughput,
            "average_delay_ms": run_history.get("average_delay_ms", np.nan),
            "load_variance": -np.mean(load_imbalance_history) if len(load_imbalance_history) > 0 else np.nan,
        })

    epsilon_labels = [str(item["epsilon"]) for item in epsilon_results]
    epsilon_plr = [item["plr"] for item in epsilon_results]
    epsilon_throughput = [item["throughput"] for item in epsilon_results]
    epsilon_delay = [item["average_delay_ms"] for item in epsilon_results]
    epsilon_load_variance = [item["load_variance"] for item in epsilon_results]

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_labels, epsilon_plr, marker="o", color="#3498db")
    plt.title("Packet Loss Rate under Imbalance Epsilon")
    plt.xlabel("Imbalance epsilon")
    plt.ylabel("PLR")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_labels, epsilon_throughput, marker="o", color="#27ae60")
    plt.title("Average Throughput under Imbalance Epsilon")
    plt.xlabel("Imbalance epsilon")
    plt.ylabel("Packets / Second")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_labels, epsilon_delay, marker="o", color="#8e44ad")
    plt.title("AverageDelay under Imbalance Epsilon")
    plt.xlabel("Imbalance epsilon")
    plt.ylabel("ms")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_labels, epsilon_load_variance, marker="o", color="#f39c12")
    plt.title("Load Variance Across Satellites under Imbalance Epsilon")
    plt.xlabel("Imbalance epsilon")
    plt.ylabel("Load Variance Across Satellites")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("\n--- Epsilon Sweep Complete ---")
    for item in epsilon_results:
        print(
            f"epsilon={item['epsilon']}: "
            f"PLR={item['plr']:.4f}, "
            f"throughput={item['throughput']:.2f}, "
            f"avg_delay_ms={item['average_delay_ms']:.2f}, "
            f"load_variance_across_satellites={item['load_variance']:.4f}"
        )
    raise SystemExit


# Current single-run experiment.
num = 10000
m = [1, 1]
USE_REAL_PS = False
result_key = "Proposed"
results = {}
# Proposed satellite selection and backoff control.
a, b, c, d, e, f, g = main.main(0.01, 2, num, m, 42, 0.01, USE_REAL_PS=USE_REAL_PS)
load_variance_history = -np.asarray(f, dtype=float)

results[result_key] = {
    "N_tilde": c,
    "Pi": d,
    "Loads": a,
    "SuccessRate": b,
    "LoadVarianceHistory": load_variance_history,
    "TruePi": e,
    "RunHistory": g,
}

single_throughput = results[result_key]["Loads"]
single_delay_ms = results[result_key]["RunHistory"].get("average_delay_ms", np.nan)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Throughput and AverageDelay - {result_key}", fontsize=14)

axes[0].bar([result_key], [single_throughput], color="#27ae60")
axes[0].set_title("Average Throughput")
axes[0].set_ylabel("Packets / Second")
axes[0].grid(True, axis="y", alpha=0.3)

axes[1].bar([result_key], [single_delay_ms], color="#8e44ad")
axes[1].set_title("AverageDelay")
axes[1].set_ylabel("ms")
axes[1].grid(True, axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.show()


plt.figure(figsize=(10, 6))
plt.axhline(y=num, color="black", linestyle="--", label=f"True N ({num})", alpha=0.6)
plt.plot(
    range(len(results[result_key]["N_tilde"])),
    results[result_key]["N_tilde"],
    label="Proposed",
    color="blue",
    linewidth=1.5,
)

plt.title("Population Estimation Convergence")
plt.xlabel("Time Slot (n)")
plt.ylabel("Estimated Population")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(
    range(len(results[result_key]["LoadVarianceHistory"])),
    results[result_key]["LoadVarianceHistory"],
    label="Load Variance Across Satellites",
    color="blue",
    linewidth=1.2,
)

plt.title("Load Variance Across Satellites over Time Slots")
plt.xlabel("Time Slot (n)")
plt.ylabel("Load Variance Across Satellites")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# State distribution over time for all pi_n states.
pi_history = np.asarray(results[result_key]["Pi"], dtype=float)
estimated_active_pi = np.asarray(results[result_key]["TruePi"], dtype=float)
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
ps_history = results[result_key]["RunHistory"].get("ps_history", [])
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
print(f"Packet Loss Rate: {results[result_key]['SuccessRate']:.4f}")
print(f"Average Throughput: {single_throughput:.2f}")
print(f"AverageDelay (ms): {single_delay_ms:.2f}")
