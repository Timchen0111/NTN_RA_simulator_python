import matplotlib.pyplot as plt
import numpy as np

import main


RUN_MODE5_ETA_SWEEP = False
RUN_EPSILON_N_ESTIMATION_SWEEP = False

NUM_UE = 10000
SECONDS = 20
SEED = 42
USE_REAL_PS = False
IMBALANCE_EPSILON = 0.01


if RUN_MODE5_ETA_SWEEP:
    RHO = 1.0
    MODE = [5, 1]
    ETA_VALUES = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])

    eta_results = []
    for eta in ETA_VALUES:
        print(f"\nRunning MODE5 eta sweep: eta={eta}")
        avg_throughput, plr, n_history, actual_pi, observe_pi, reward_history, run_history = main.main(
            RHO,
            SECONDS,
            NUM_UE,
            MODE,
            SEED,
            IMBALANCE_EPSILON=IMBALANCE_EPSILON,
            USE_REAL_PS=USE_REAL_PS,
            LOAD_AWARE_ETA=eta,
        )
        final_n_estimate = n_history[-1] if len(n_history) > 0 else np.nan
        eta_results.append({
            "eta": eta,
            "plr": plr,
            "throughput": avg_throughput,
            "average_delay_ms": run_history.get("average_delay_ms", np.nan),
            "final_n_estimate": final_n_estimate,
        })

    eta_labels = [f"{item['eta']:.2g}" for item in eta_results]
    eta_plr = np.array([item["plr"] for item in eta_results])

    plt.figure(figsize=(10, 6))
    plt.plot(eta_labels, eta_plr, marker="o", linewidth=1.6)
    plt.title(r"MODE5 PLR under Different Load-Aware $\eta$")
    plt.xlabel(r"Load-aware parameter $\eta$")
    plt.ylabel("Packet Loss Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n--- MODE5 Eta Sweep Complete ---")
    for item in eta_results:
        print(
            f"eta={item['eta']:.4g}: "
            f"PLR={item['plr']:.4f}, "
            f"throughput={item['throughput']:.2f}, "
            f"avg_delay_ms={item['average_delay_ms']:.2f}, "
            f"final_N={item['final_n_estimate']:.2f}"
        )


if RUN_EPSILON_N_ESTIMATION_SWEEP:
    # Sweep only the convex satellite-selection imbalance epsilon.  All other
    # parameters are fixed so the curve isolates epsilon's effect on N estimation.
    EPSILON_VALUES = np.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1])
    RHO = 0.01
    IMBALANCE_EPSILON_MODE = [1, 1]

    results = []
    for epsilon in EPSILON_VALUES:
        print(f"\nRunning epsilon sweep: epsilon={epsilon}")
        avg_throughput, plr, n_history, actual_pi, observe_pi, reward_history, run_history = main.main(
            RHO,
            SECONDS,
            NUM_UE,
            IMBALANCE_EPSILON_MODE,
            SEED,
            IMBALANCE_EPSILON=epsilon,
            USE_REAL_PS=USE_REAL_PS,
        )
        final_n_estimate = n_history[-1] if len(n_history) > 0 else np.nan
        signed_error = final_n_estimate - NUM_UE
        relative_error = signed_error / NUM_UE
        results.append({
            "epsilon": epsilon,
            "final_n_estimate": final_n_estimate,
            "signed_error": signed_error,
            "relative_error": relative_error,
            "absolute_relative_error": abs(relative_error),
            "plr": plr,
            "throughput": avg_throughput,
        })

    epsilon_labels = [f"{item['epsilon']:.0e}" if item["epsilon"] > 0 else "0" for item in results]
    absolute_relative_errors = np.array([item["absolute_relative_error"] for item in results]) * 100
    signed_relative_errors = np.array([item["relative_error"] for item in results]) * 100
    plr_values = np.array([item["plr"] for item in results])

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_labels, absolute_relative_errors, marker="o", linewidth=1.6, label=r"$|\hat{N}_{final}-N|/N$")
    for label, abs_err, signed_err in zip(epsilon_labels, absolute_relative_errors, signed_relative_errors):
        plt.annotate(
            f"{signed_err:+.1f}%",
            (label, abs_err),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    plt.title(r"Final UE Number Estimation Error under Different $\epsilon$")
    plt.xlabel(r"Imbalance epsilon $\epsilon$")
    plt.ylabel("Absolute relative N estimation error (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_labels, plr_values, marker="o", color="#3498db")
    plt.title("Packet Loss Rate under Imbalance Epsilon")
    plt.xlabel("Imbalance epsilon")
    plt.ylabel("PLR")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("\n--- Epsilon N Estimation Sweep Complete ---")
    for item in results:
        print(
            f"epsilon={item['epsilon']:.4g}: "
            f"final_N={item['final_n_estimate']:.2f}, "
            f"signed_error={item['signed_error']:+.2f}, "
            f"relative_error={item['relative_error'] * 100:+.2f}%, "
            f"PLR={item['plr']:.4f}, "
            f"throughput={item['throughput']:.2f}"
        )
