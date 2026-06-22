import matplotlib.pyplot as plt
import numpy as np

import main


RUN_MODE6_ALPHA_SWEEP = False
RUN_MODE5_ETA_SWEEP = False
RUN_SATELLITE_SELECTION_VALIDATION = True
RUN_EPSILON_N_ESTIMATION_SWEEP = False

NUM_UE = 10000
SECONDS = 5
SEED = 42
USE_REAL_PS = False
IMBALANCE_EPSILON = 0.01


if RUN_SATELLITE_SELECTION_VALIDATION:
    RHO_VALUES = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    MODE5_ETA_VALUES = np.array([0.2, 1.0, 5.0])
    EXPERIMENTS = [
        ([3, 1], "VU", {}),
        ([4, 1], "HE", {}),
        ([6, 1], "MODE6", {}),
    ]
    EXPERIMENTS.extend(
        ([5, 1], rf"MODE5, $\eta={eta:g}$", {"LOAD_AWARE_ETA": eta})
        for eta in MODE5_ETA_VALUES
    )

    validation_results = {label: [] for _, label, _ in EXPERIMENTS}
    for mode, label, extra_kwargs in EXPERIMENTS:
        for rho in RHO_VALUES:
            print(f"\nRunning satellite selection validation: {label}, rho_s={rho}")
            avg_throughput, plr, n_history, actual_pi, observe_pi, reward_history, run_history = main.main(
                rho,
                SECONDS,
                NUM_UE,
                mode,
                SEED,
                IMBALANCE_EPSILON=IMBALANCE_EPSILON,
                USE_REAL_PS=USE_REAL_PS,
                **extra_kwargs,
            )
            final_n_estimate = n_history[-1] if len(n_history) > 0 else np.nan
            validation_results[label].append({
                "rho": rho,
                "plr": plr,
                "throughput": avg_throughput,
                "average_delay_ms": run_history.get("average_delay_ms", np.nan),
                "final_n_estimate": final_n_estimate,
            })

    plt.figure(figsize=(10, 6))
    for _, label, _ in EXPERIMENTS:
        rho_axis = np.array([item["rho"] for item in validation_results[label]])
        plr_values = np.array([item["plr"] for item in validation_results[label]])
        plt.plot(rho_axis, plr_values, marker="o", linewidth=1.6, label=label)
    plt.title("Satellite Selection PLR Validation")
    plt.xlabel(r"Arrival rate $\rho_s$ (packets/s)")
    plt.ylabel("Packet Loss Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for _, label, _ in EXPERIMENTS:
        rho_axis = np.array([item["rho"] for item in validation_results[label]])
        throughput_values = np.array([item["throughput"] for item in validation_results[label]])
        plt.plot(rho_axis, throughput_values, marker="o", linewidth=1.6, label=label)
    plt.title("Satellite Selection Throughput Validation")
    plt.xlabel(r"Arrival rate $\rho_s$ (packets/s)")
    plt.ylabel("Average Throughput (packets/second)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for _, label, _ in EXPERIMENTS:
        rho_axis = np.array([item["rho"] for item in validation_results[label]])
        delay_values = np.array([item["average_delay_ms"] for item in validation_results[label]])
        plt.plot(rho_axis, delay_values, marker="o", linewidth=1.6, label=label)
    plt.title("Satellite Selection Average Delay Validation")
    plt.xlabel(r"Arrival rate $\rho_s$ (packets/s)")
    plt.ylabel("Average delay (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Satellite Selection Validation Complete ---")
    for _, label, _ in EXPERIMENTS:
        for item in validation_results[label]:
            print(
                f"{label}, rho_s={item['rho']:.4g}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"avg_delay_ms={item['average_delay_ms']:.2f}, "
                f"final_N={item['final_n_estimate']:.2f}"
            )


if RUN_MODE6_ALPHA_SWEEP:
    MODE = [6, 1]
    LAMBDA_VALUES = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    ALPHA_VALUES = np.array([0.25, 0.5, 1.0, 2.0])
    ADAPTIVE_EPSILON_MIN = 1e-4
    ADAPTIVE_EPSILON_MAX = 1e-2
    ADAPTIVE_EPSILON_BETA = 0.2

    alpha_results = {alpha: [] for alpha in ALPHA_VALUES}
    for alpha in ALPHA_VALUES:
        for lam in LAMBDA_VALUES:
            print(f"\nRunning MODE6 alpha sweep: alpha={alpha}, lambda={lam}")
            avg_throughput, plr, n_history, actual_pi, observe_pi, reward_history, run_history = main.main(
                lam,
                SECONDS,
                NUM_UE,
                MODE,
                SEED,
                IMBALANCE_EPSILON=IMBALANCE_EPSILON,
                USE_REAL_PS=USE_REAL_PS,
                ADAPTIVE_EPSILON_MIN=ADAPTIVE_EPSILON_MIN,
                ADAPTIVE_EPSILON_MAX=ADAPTIVE_EPSILON_MAX,
                ADAPTIVE_EPSILON_ALPHA=alpha,
                ADAPTIVE_EPSILON_BETA=ADAPTIVE_EPSILON_BETA,
            )
            epsilon_history = run_history.get("adaptive_epsilon_history", [])
            final_epsilon = epsilon_history[-1]["epsilon"] if len(epsilon_history) > 0 else np.nan
            alpha_results[alpha].append({
                "lambda": lam,
                "plr": plr,
                "throughput": avg_throughput,
                "average_delay_ms": run_history.get("average_delay_ms", np.nan),
                "final_epsilon": final_epsilon,
            })

    plt.figure(figsize=(10, 6))
    for alpha in ALPHA_VALUES:
        lambda_axis = np.array([item["lambda"] for item in alpha_results[alpha]])
        plr_values = np.array([item["plr"] for item in alpha_results[alpha]])
        plt.plot(lambda_axis, plr_values, marker="o", linewidth=1.6, label=rf"$\alpha={alpha:g}$")
    plt.title(r"MODE6 PLR under Different $\alpha$")
    plt.xlabel(r"Arrival rate $\lambda$ (packets/s)")
    plt.ylabel("Packet Loss Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- MODE6 Alpha Sweep Complete ---")
    for alpha in ALPHA_VALUES:
        for item in alpha_results[alpha]:
            print(
                f"alpha={alpha:.4g}, lambda={item['lambda']:.4f}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"avg_delay_ms={item['average_delay_ms']:.2f}, "
                f"final_epsilon={item['final_epsilon']:.6g}"
            )


if RUN_MODE5_ETA_SWEEP:
    MODE5 = [5, 1]
    MODE3 = [3, 1]
    RHO_VALUES = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    ETA_VALUES = np.array([0.1, 0.5, 1, 5, 10])

    eta_results = {rho: [] for rho in RHO_VALUES}
    mode3_results = {}
    for rho in RHO_VALUES:
        print(f"\nRunning VU baseline for eta sweep: rho_s={rho}")
        avg_throughput, plr, n_history, actual_pi, observe_pi, reward_history, run_history = main.main(
            rho,
            SECONDS,
            NUM_UE,
            MODE3,
            SEED,
            IMBALANCE_EPSILON=IMBALANCE_EPSILON,
            USE_REAL_PS=USE_REAL_PS,
        )
        final_n_estimate = n_history[-1] if len(n_history) > 0 else np.nan
        mode3_results[rho] = {
            "plr": plr,
            "throughput": avg_throughput,
            "average_delay_ms": run_history.get("average_delay_ms", np.nan),
            "final_n_estimate": final_n_estimate,
        }

        for eta in ETA_VALUES:
            print(f"\nRunning MODE5 eta sweep: rho_s={rho}, eta={eta}")
            avg_throughput, plr, n_history, actual_pi, observe_pi, reward_history, run_history = main.main(
                rho,
                SECONDS,
                NUM_UE,
                MODE5,
                SEED,
                IMBALANCE_EPSILON=IMBALANCE_EPSILON,
                USE_REAL_PS=USE_REAL_PS,
                LOAD_AWARE_ETA=eta,
            )
            final_n_estimate = n_history[-1] if len(n_history) > 0 else np.nan
            eta_results[rho].append({
                "eta": eta,
                "plr": plr,
                "throughput": avg_throughput,
                "average_delay_ms": run_history.get("average_delay_ms", np.nan),
                "final_n_estimate": final_n_estimate,
            })

    plt.figure(figsize=(10, 6))
    for eta in ETA_VALUES:
        plr_values = []
        for rho in RHO_VALUES:
            eta_item = next(item for item in eta_results[rho] if item["eta"] == eta)
            plr_values.append(eta_item["plr"])
        plt.plot(RHO_VALUES, np.array(plr_values), marker="o", linewidth=1.6, label=rf"MODE5, $\eta={eta:g}$")
    mode3_plr_values = np.array([mode3_results[rho]["plr"] for rho in RHO_VALUES])
    plt.plot(RHO_VALUES, mode3_plr_values, marker="s", linestyle="--", linewidth=1.8, color="black", label="VU")
    plt.title("MODE5 Eta Sweep with VU Baseline")
    plt.xlabel(r"Arrival rate $\rho_s$ (packets/s)")
    plt.ylabel("Packet Loss Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- MODE5 Eta Sweep Complete ---")
    for rho in RHO_VALUES:
        best_item = min(eta_results[rho], key=lambda item: item["plr"])
        mode3_item = mode3_results[rho]
        best_delta = best_item["plr"] - mode3_item["plr"]
        print(
            f"Best eta for rho_s={rho:g}: eta={best_item['eta']:.4g}, "
            f"MODE5 PLR={best_item['plr']:.4f}, "
            f"VU PLR={mode3_item['plr']:.4f}, "
            f"delta={best_delta:+.4f}"
        )
        print(
            f"VU, rho_s={rho:.4g}: "
            f"PLR={mode3_item['plr']:.4f}, "
            f"throughput={mode3_item['throughput']:.2f}, "
            f"avg_delay_ms={mode3_item['average_delay_ms']:.2f}, "
            f"final_N={mode3_item['final_n_estimate']:.2f}"
        )
        for item in eta_results[rho]:
            print(
                f"rho_s={rho:.4g}, eta={item['eta']:.4g}: "
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
