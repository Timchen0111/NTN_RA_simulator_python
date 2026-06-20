import matplotlib.pyplot as plt
import numpy as np

import main

RUN_RHO_SWEEP = False
RUN_SATELLITE_SELECTION_SWEEP = False
RUN_SATELLITE_SELECTION_PERFORMANCE = False
RUN_ESTIMATION_VALIDATION_RHO_SWEEP = True
epsilon_sweep = False

if RUN_RHO_SWEEP:
    NUM_UE = 10000
    SECONDS = 10
    SEED = 42
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    RHO_VALUES = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
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
            print(f"\nRunning PLR lambda sweep: {label}, lambda={rho}")
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
    plt.title(r"PLR Comparison under Different $\lambda$")
    plt.xlabel(r"Arrival rate $\lambda$ (packets/s)")
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
    plt.title(r"Throughput Comparison under Different $\lambda$")
    plt.xlabel(r"Arrival rate $\lambda$ (packets/s)")
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
    plt.title(r"AverageDelay Comparison under Different $\lambda$")
    plt.xlabel(r"Arrival rate $\lambda$ (packets/s)")
    plt.ylabel("AverageDelay (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Lambda Sweep Complete ---")
    for _, label in MODES:
        for item in rho_results[label]:
            print(
                f"{label}, lambda={item['rho']:.4f}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"avg_delay_ms={item['average_delay_ms']:.2f}, "
                f"final_N={item['final_n_estimate']:.2f}"
            )
    raise SystemExit

if RUN_SATELLITE_SELECTION_SWEEP:
    NUM_UE = 10000
    SECONDS = 10
    SEED = 42
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    RHO_VALUES = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    EXPERIMENTS = [
        #([1, 1], "Proposed, epsilon=0.01", 0.01),
        #([1, 1], "Proposed, epsilon=0.001", 0.001),
        #([3, 1], "Visible-Uniform", IMBALANCE_EPSILON),
        #([4, 1], "Highest-Elevation", IMBALANCE_EPSILON),
        ([5, 1], "Load-Link-Aware", IMBALANCE_EPSILON),
    ]

    # Satellite-selection baselines keep the proposed backoff controller fixed
    # so the PLR curves isolate the satellite selection policy.
    selection_results = {label: [] for _, label, _ in EXPERIMENTS}
    for mode, label, epsilon in EXPERIMENTS:
        for rho in RHO_VALUES:
            print(f"\nRunning satellite selection lambda sweep: {label}, lambda={rho}")
            avg_throughput, plr, n_history, actual_pi, observe_pi, load_imbalance_history, run_history = main.main(
                rho,
                SECONDS,
                NUM_UE,
                mode,
                SEED,
                epsilon,
                USE_REAL_PS=USE_REAL_PS,
            )
            final_n_estimate = n_history[-1] if len(n_history) > 0 else np.nan
            selection_results[label].append({
                "rho": rho,
                "epsilon": epsilon,
                "plr": plr,
                "throughput": avg_throughput,
                "average_delay_ms": run_history.get("average_delay_ms", np.nan),
                "final_n_estimate": final_n_estimate,
            })

    plt.figure(figsize=(10, 6))
    for _, label, _ in EXPERIMENTS:
        rho_axis = np.array([item["rho"] for item in selection_results[label]])
        plr_values = np.array([item["plr"] for item in selection_results[label]])
        plt.plot(rho_axis, plr_values, marker="o", linewidth=1.6, label=label)
    plt.title(r"Satellite Selection PLR Comparison under Different $\lambda$")
    plt.xlabel(r"Arrival rate $\lambda$ (packets/s)")
    plt.ylabel("Packet Loss Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for _, label, _ in EXPERIMENTS:
        rho_axis = np.array([item["rho"] for item in selection_results[label]])
        throughput_values = np.array([item["throughput"] for item in selection_results[label]])
        plt.plot(rho_axis, throughput_values, marker="o", linewidth=1.6, label=label)
    plt.title(r"Satellite Selection Throughput Comparison under Different $\lambda$")
    plt.xlabel(r"Arrival rate $\lambda$ (packets/s)")
    plt.ylabel("Average Throughput (packets/second)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for _, label, _ in EXPERIMENTS:
        rho_axis = np.array([item["rho"] for item in selection_results[label]])
        delay_values = np.array([item["average_delay_ms"] for item in selection_results[label]])
        plt.plot(rho_axis, delay_values, marker="o", linewidth=1.6, label=label)
    plt.title(r"Satellite Selection AverageDelay Comparison under Different $\lambda$")
    plt.xlabel(r"Arrival rate $\lambda$ (packets/s)")
    plt.ylabel("AverageDelay (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Satellite Selection Lambda Sweep Complete ---")
    for _, label, _ in EXPERIMENTS:
        for item in selection_results[label]:
            print(
                f"{label}, lambda={item['rho']:.4f}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"avg_delay_ms={item['average_delay_ms']:.2f}, "
                f"final_N={item['final_n_estimate']:.2f}"
            )
    raise SystemExit

if RUN_ESTIMATION_VALIDATION_RHO_SWEEP:
    NUM_UE = 10000
    SECONDS = 1
    SEED = 42
    MODE = [6, 1]
    IMBALANCE_EPSILON = 0.01
    USE_REAL_PS = False
    ADAPTIVE_EPSILON_ALPHA = 2.0
    RHO_VALUES = np.array([1.0, 1.5, 2.0, 2.5, 3.0])

    validation_results = []
    for rho in RHO_VALUES:
        print(f"\nRunning estimation validation lambda sweep: lambda={rho}")
        avg_throughput, plr, n_history, actual_pi, observe_pi, load_imbalance_history, run_history = main.main(
            rho,
            SECONDS,
            NUM_UE,
            MODE,
            SEED,
            IMBALANCE_EPSILON,
            USE_REAL_PS=USE_REAL_PS,
            ADAPTIVE_EPSILON_ALPHA=ADAPTIVE_EPSILON_ALPHA,
        )

        ps_history = run_history.get("ps_history", [])
        if len(ps_history) > 0:
            ps_error = np.array([item["error"] for item in ps_history], dtype=float)
            ps_mae = np.mean(np.abs(ps_error))
        else:
            ps_mae = np.nan

        final_n_estimate = n_history[-1] if len(n_history) > 0 else np.nan
        n_signed_error = final_n_estimate - NUM_UE
        n_abs_relative_error = abs(n_signed_error) / NUM_UE if np.isfinite(final_n_estimate) else np.nan

        pi_history = np.asarray(actual_pi, dtype=float)
        estimated_active_pi = np.asarray(observe_pi, dtype=float)
        estimated_pi = np.concatenate(([max(0.0, 1.0 - np.sum(estimated_active_pi))], estimated_active_pi))
        if pi_history.size > 0:
            if pi_history.ndim == 1:
                pi_history = pi_history.reshape(1, -1)
            state_count = min(pi_history.shape[1], estimated_pi.size)
            pi_error_by_state = np.mean(
                estimated_pi[:state_count] - pi_history[:, :state_count],
                axis=0,
            )
        else:
            state_count = estimated_pi.size
            pi_error_by_state = np.full(state_count, np.nan)

        validation_results.append({
            "rho": rho,
            "plr": plr,
            "throughput": avg_throughput,
            "ps_mae": ps_mae,
            "final_n_estimate": final_n_estimate,
            "n_signed_error": n_signed_error,
            "n_abs_relative_error": n_abs_relative_error,
            "pi_error_by_state": pi_error_by_state,
        })

    rho_axis = np.array([item["rho"] for item in validation_results])

    plt.figure(figsize=(10, 6))
    plt.plot(
        rho_axis,
        np.array([item["ps_mae"] for item in validation_results]),
        marker="o",
        linewidth=1.6,
        color="#3498db",
    )
    plt.title("Successful Transmission Probability Error under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
    plt.ylabel("Mean absolute error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    max_state_count = max(len(item["pi_error_by_state"]) for item in validation_results)
    pi_error_matrix = np.vstack([
        np.pad(
            item["pi_error_by_state"],
            (0, max_state_count - len(item["pi_error_by_state"])),
            constant_values=np.nan,
        )
        for item in validation_results
    ])
    x = np.arange(len(rho_axis))
    bar_width = 0.8 / max_state_count

    plt.figure(figsize=(11, 6))
    for state_idx in range(max_state_count):
        state_label = "Idle" if state_idx == 0 else f"State {state_idx}"
        offsets = x - 0.4 + bar_width * (state_idx + 0.5)
        plt.bar(
            offsets,
            pi_error_matrix[:, state_idx],
            width=bar_width,
            label=state_label,
        )
    plt.axhline(y=0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.title("State Probability Estimation Error under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
    plt.ylabel("Mean error (estimated minus true)")
    plt.xticks(x, [f"{rho:g}" for rho in rho_axis])
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(
        rho_axis,
        np.array([item["n_abs_relative_error"] for item in validation_results]) * 100.0,
        marker="o",
        linewidth=1.6,
        color="#8e44ad",
    )
    plt.title("Final UE Number Estimation Error under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
    plt.ylabel("Final absolute relative error (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n--- Estimation Validation Rho Sweep Complete ---")
    for item in validation_results:
        pi_summary = ", ".join(
            f"{'Idle' if idx == 0 else f'State {idx}'}={value:.6f}"
            for idx, value in enumerate(item["pi_error_by_state"])
        )
        print(
            f"lambda={item['rho']:.4f}: "
            f"p_s_MAE={item['ps_mae']:.6f}, "
            f"true_system_PLR={item['plr']:.4f}, "
            f"final_N={item['final_n_estimate']:.2f}, "
            f"N_signed_error={item['n_signed_error']:+.2f}, "
            f"N_abs_relative_error={item['n_abs_relative_error'] * 100:.2f}%, "
            f"state_probability_mean_error_by_state=[{pi_summary}]"
        )
    raise SystemExit

if RUN_SATELLITE_SELECTION_PERFORMANCE:
    NUM_UE = 10000
    SECONDS = 180
    SEED = 42
    USE_REAL_PS = False
    FIXED_EPSILON_MODE = [1, 1]
    ADAPTIVE_EPSILON_MODE = [6, 1]
    FIXED_EPSILON_RHO = 1.0
    FIXED_EPSILON_VALUES = [1e-4, 1e-3, 1e-2, 1e-1]
    ADAPTIVE_EPSILON_RHO_VALUES = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    ADAPTIVE_EPSILON_ALPHA = 2.0

    fixed_epsilon_results = []
    for eps in FIXED_EPSILON_VALUES:
        print(f"\nRunning fixed-epsilon satellite selection performance: epsilon={eps}")
        avg_throughput, plr, n_history, actual_pi, observe_pi, load_imbalance_history, run_history = main.main(
            FIXED_EPSILON_RHO,
            SECONDS,
            NUM_UE,
            FIXED_EPSILON_MODE,
            SEED,
            IMBALANCE_EPSILON=eps,
            USE_REAL_PS=USE_REAL_PS,
        )
        ps_history = run_history.get("ps_history", [])
        if len(ps_history) > 0:
            pbar_s = np.mean([item["precomputed"] for item in ps_history])
        else:
            pbar_s = np.nan
        fixed_epsilon_results.append({
            "epsilon": eps,
            "pbar_s": pbar_s,
            "plr": plr,
            "throughput": avg_throughput,
        })

    epsilon_values_for_plot = np.array([item["epsilon"] for item in fixed_epsilon_results])
    epsilon_labels = [f"{item['epsilon']:.0e}" for item in fixed_epsilon_results]
    pbar_values = np.array([item["pbar_s"] for item in fixed_epsilon_results])

    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(epsilon_values_for_plot, pbar_values, marker="o", linewidth=1.6, color="#3498db")
    plt.xscale("log")
    plt.xticks(epsilon_values_for_plot, epsilon_labels)
    plt.title(r"Average $\bar{p}_s$ under Fixed Imbalance Epsilon")
    plt.xlabel(r"Fixed imbalance threshold $\epsilon$ (log scale)")
    plt.ylabel(r"Average $\bar{p}_s$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    adaptive_epsilon_results = []
    for rho in ADAPTIVE_EPSILON_RHO_VALUES:
        print(f"\nRunning adaptive-epsilon trajectory: lambda={rho}")
        avg_throughput, plr, n_history, actual_pi, observe_pi, load_imbalance_history, run_history = main.main(
            rho,
            SECONDS,
            NUM_UE,
            ADAPTIVE_EPSILON_MODE,
            SEED,
            IMBALANCE_EPSILON=0.01,
            USE_REAL_PS=USE_REAL_PS,
            ADAPTIVE_EPSILON_ALPHA=ADAPTIVE_EPSILON_ALPHA,
        )
        epsilon_history = run_history.get("adaptive_epsilon_history", [])
        epsilon_values = np.array([item["epsilon"] for item in epsilon_history], dtype=float)
        adaptive_epsilon_results.append({
            "rho": rho,
            "epsilon_values": epsilon_values,
            "plr": plr,
            "throughput": avg_throughput,
        })

    plt.figure(figsize=(10, 6), dpi=120)
    for item in adaptive_epsilon_results:
        epsilon_values = item["epsilon_values"]
        if len(epsilon_values) == 0:
            continue
        plt.plot(
            np.arange(len(epsilon_values)),
            epsilon_values,
            linewidth=1.4,
            label=rf"$\lambda={item['rho']:g}$",
        )
    plt.title(r"Adaptive Imbalance Epsilon under Different $\lambda$")
    plt.xlabel("Time Slot (n)")
    plt.ylabel(r"Adaptive imbalance threshold $\epsilon^m$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Satellite Selection Performance Complete ---")
    for item in fixed_epsilon_results:
        print(
            f"fixed epsilon={item['epsilon']:.4g}: "
            f"avg_p_s={item['pbar_s']:.6f}, "
            f"PLR={item['plr']:.4f}, "
            f"throughput={item['throughput']:.2f}"
        )
    for item in adaptive_epsilon_results:
        final_epsilon = item["epsilon_values"][-1] if len(item["epsilon_values"]) > 0 else np.nan
        print(
            f"adaptive lambda={item['rho']:.4f}: "
            f"final_epsilon={final_epsilon:.6g}, "
            f"PLR={item['plr']:.4f}, "
            f"throughput={item['throughput']:.2f}"
        )
    raise SystemExit

if epsilon_sweep:
    # Epsilon sweep for convex group satellite selection.
    EPSILON_VALUES = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    EPSILON_RHO = 1.0
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
m = [6, 1] #Satellite selection mode and backoff control mode. 
USE_REAL_PS = False
result_key = "Proposed"
results = {}
# Proposed satellite selection and backoff control.
a, b, c, d, e, f, g = main.main(1.0, 1, num, m, 42, 0.01, USE_REAL_PS=USE_REAL_PS)
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


# Adaptive epsilon over time; only meaningful for satellite selection MODE 6.
if m[0] == 6:
    adaptive_epsilon_history = results[result_key]["RunHistory"].get("adaptive_epsilon_history", [])
    if len(adaptive_epsilon_history) > 0:
        epsilon_time = np.arange(len(adaptive_epsilon_history))
        epsilon_values = np.array([item["epsilon"] for item in adaptive_epsilon_history], dtype=float)

        plt.figure(figsize=(10, 6))
        plt.plot(epsilon_time, epsilon_values, label="Adaptive epsilon", color="#d35400", linewidth=1.3)
        plt.title("Adaptive Imbalance Epsilon over Time")
        plt.xlabel("Time Slot (n)")
        plt.ylabel("Imbalance epsilon")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No adaptive epsilon history was recorded; skip epsilon plot.")


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

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(r"Convergence of State Distributions ($\pi_n$)", fontsize=14)

    # Draw idle probability separately so its larger scale does not flatten the active states.
    idle_line = axes[0].plot(
        time_slots,
        pi_history[:, 0],
        label="Idle",
        linewidth=1.5,
    )[0]
    if estimated_pi.size > 0:
        axes[0].axhline(
            y=estimated_pi[0],
            color=idle_line.get_color(),
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label="Estimated Idle",
        )
    axes[0].set_title("Idle State")
    axes[0].set_ylabel("Probability")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for state_idx in range(1, state_count):
        state_label = f"State {state_idx}"
        line = axes[1].plot(
            time_slots,
            pi_history[:, state_idx],
            label=state_label,
            linewidth=1.5,
        )[0]

        if state_idx < estimated_pi.size:
            axes[1].axhline(
                y=estimated_pi[state_idx],
                color=line.get_color(),
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                label=f"Estimated {state_label}",
            )

    axes[1].set_title("Active States")
    axes[1].set_xlabel("Time Slot (n)")
    axes[1].set_ylabel("Probability")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.show()
else:
    print("No pi history was recorded; skip state distribution plot.")


# Precomputed p_s error over time.
ps_history = results[result_key]["RunHistory"].get("ps_history", [])
ps_mae = np.nan
if len(ps_history) > 0:
    ps_time = np.array([item["time_slot"] for item in ps_history])
    ps_error = np.array([item["error"] for item in ps_history])
    ps_mae = np.mean(np.abs(ps_error))
    real_ps = np.array([item["real"] for item in ps_history])
    precomputed_ps = np.array([item["precomputed"] for item in ps_history])

    plt.figure(figsize=(10, 6))
    plt.axhline(y=0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.plot(ps_time, ps_error, label=r"Error: real $p_s$ - estimated $p_s$", color="red", linewidth=1.3)
    plt.title(r"$p_s$ Error Over Time")
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
print(f"p_s MAE: {ps_mae:.6f}" if np.isfinite(ps_mae) else "p_s MAE: N/A")
