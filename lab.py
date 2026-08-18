from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import Load_estimator
import backoff_control
import main

# =============================================================================
# 第一層實驗模式索引
# =============================================================================
# EXPERIMENT_CODE 只負責選擇「要執行哪一組實驗／產生哪一組圖」。
#
# 與 chap5.tex 的圖名對照如下。英文圖名、PDF 檔名與 LaTeX label 均依論文：
#
# 0  SINGLE_RUN
#    用途：單次模擬診斷，觀察 throughput、delay、UE 數量估測收斂、
#          satellite load variance、adaptive epsilon、pi 與 p_s。
#    論文圖：UE number estimator performance over time
#            ue_estimate.pdf, \label{fig: n_over_time}
#    備註：其餘輸出為除錯／診斷圖，未直接對應目前第五章定稿圖。
#
# 1  RUN_ALL
#    用途：完整 DCLARA 與 ALLA 整合式 baseline 的系統效能比較。
#    論文圖：Packet loss rate comparison of different integrated schemes
#            all_plr.pdf, \label{fig:allplr}
#            Delay budget utilization comparison of different schemes
#            db_consumption.pdf, \label{fig:db_consumption}
#    備註：程式另畫 throughput，該圖未直接收入目前第五章。
#
# 2  RHO_SWEEP_PB
#    用途：比較不同 traffic load 下，各剩餘 delay state 的平均 backoff
#          probability，檢查 backoff controller 的輸出行為。
#    論文圖：Backoff controller performance 3D bar chart
#            backoff_heatmap.pdf, \label{fig: pb}
#
# 3  RUN_QOS_DISTRIBUTION_COMPARISON
#    用途：比較 Balanced、Non-urgent、Urgent、Bimodal 四種初始
#          delay-budget distribution 下的 PLR。
#    論文圖：Packet loss rate comparison over different delay-budget distributions
#            qos_plr.pdf, \label{fig:qos_plr}
#
# 4  RUN_RHO_SWEEP
#    用途：固定 DCLARA satellite selection，比較 DCLARA-BC、DACB、
#          SAACB 三種 backoff control scheme。
#    論文圖：Packet loss rate comparison of different backoff control schemes
#            bs_new.pdf, \label{fig:bs}
#    備註：程式另畫 throughput 與 delay，未直接收入目前第五章。
#
# 5  RUN_FIXED_LOAD_IMBALANCE_SWEEP
#    用途：在不同 traffic load 下，比較 fixed epsilon 與 adaptive epsilon
#          對 PLR 的影響。
#    論文圖：Comparison of the proposed adaptive load-imbalance threshold
#            with fixed $\epsilon$ values
#            eps_tuning.pdf, \label{fig:fixedplr}
#
# 6  RUN_SATELLITE_SELECTION_SWEEP
#    用途：固定 SAACB backoff，比較不同 satellite selection scheme 的 PLR。
#    論文圖：Packet loss rate comparison of different satellite selection schemes
#            ssplr.pdf, \label{fig:ssplr}
#    備註：程式另畫 throughput 與 delay，未直接收入目前第五章。
#
# 7  RUN_ESTIMATION_VALIDATION_RHO_SWEEP
#    用途：在不同 traffic load 下驗證 predicted p_s、steady-state pi，
#          並附帶檢查最終 UE number estimate。
#    論文圖：Predicted transmission success probability error over different
#            load conditions
#            newpserror.pdf, \label{fig:ps_error}
#            Steady state probability error over different load
#            newpierror.pdf, \label{fig: pi_estimate}
#    備註：最終 UE number relative error 圖為診斷圖；論文中的 UE number
#          convergence over time 由 mode 0 觀察。
#
# 8  RUN_SATELLITE_SELECTION_PERFORMANCE
#    用途：分析 fixed epsilon 對平均 p_s 的影響，以及 adaptive epsilon
#          在不同 traffic load 下隨時間的變化。
#    論文圖：Expected successful preamble transmission probability
#            $\bar{p}^{\mathrm{s}}$
#            across different load-imbalance thresholds
#            fixed_eps.pdf, \label{fig: ps_across_eps}
#            Adaptive load-imbalance threshold $\epsilon_m$ under different
#            traffic load conditions over time
#            rho_s_over_t.pdf, \label{fig: different_load_eps}
#
# 9  RUN_GROUP_LEVEL_STATISTICS
#    用途：檢查每個 RAO 的 group 數量，以及各時間區段 dominant group 的
#          group weight 分布。
#    論文圖：Number of existing groups during the simulation period
#            Group_num.pdf, \label{fig: group_num}
#            Weight distribution of the dominant groups
#            Group_heat.pdf, \label{fig:group_heatmap}
#
# 10 RUN_ALLA_ETA_SWEEP
#    用途：在不同 traffic load 下調整 ALLA 的 weighting factor eta。
#    論文圖：Tuning of the weighting factor $\eta$ for adapted LLA schemes
#            eta_sweep.pdf, \label{fig: eta_sweep}
#
# 11 RUN_OFFERED_LOAD_RHO_SWEEP
#    用途：將 per-UE arrival rate 對應到 normalized offered load，決定後續
#          baseline comparison 採用的 traffic-load 範圍。
#    論文圖：Average normalized offered load among different traffic load conditions
#            rhos_range.pdf, \label{fig:rhos_range}
#
# 12 RUN_SATELLITE_SELECTION_CONCENTRATION
#    用途：比較各 satellite selection scheme 隨時間的最大單星選擇占比，
#          用來觀察 UE selection concentration。
#    論文圖：Maximum satellite selection share comparison of different
#            satellite selection schemes over time
#            max_ss.pdf, \label{fig:maxss}
#
# 13 RUN_LOAD_ESTIMATOR_PERFORMANCE
#    用途：比較 MoM load estimator 的 estimated load、true load 與
#          absolute estimation error。
#    論文圖：Load estimator performance
#            load_estimator_performance.pdf,
#            \label{fig: load_estimator_performance}
#
# 14 RUN_COLLISION_RATE_COMPARISON
#    用途：比較衛星端量測的 real average collision rate 與 DCLARA-SS
#          根據 predicted effective-load fractions 得到的 collision rate。
#
# 15 RUN_SERVICE_RADIUS_COMPARISON
#    Compare the same two integrated schemes as mode 1 under 100, 200, and
#    300 km service radii at a fixed arrival rate of 1.5 packets/s.
#    Each radius uses its matching precomputed group probability table while
#    keeping the fixed satellite pool, arrival rate, UE count, and seed equal.
#
# 16 RUN_BACKOFF_INITIAL_GUESS_SENSITIVITY
#    Keep several sampled RAO states fixed and rerun only the backoff optimizer
#    from different initial vectors to test initial-guess sensitivity.
#
# 17 RUN_ORBIT_PLANE_COMPARISON
#    Compare the same two integrated schemes under independently generated
#    1-, 2-, 3-, and 4-orbit-plane scenarios at 200 km and 1.5 packets/s.
#    The 3-plane scenario reuses the original satellite pool and group table.
#
# 18 RUN_TOP_K_GROUPING_ANALYSIS
#    Reproduce Fig. 11 by comparing Global, Top-1, Top-2, and Top-3 grouping
#    policies using the precomputed ordered Top-3 group table.
#
# 19 RUN_TOP_K_ORBIT_PLANE_COMPARISON
#    Compare the grouping performance-complexity tradeoff under independently
#    generated 1-, 2-, 3-, and 4-orbit-plane scenarios.
#
# 20 RUN_SATELLITE_SELECTION_TOP5_OVER_TIME
#    Aggregate all UE satellite selections in six non-overlapping 300-RAO
#    windows and plot the five largest satellite-selection shares per window.
#
# 21 RUN_SATELLITE_POOL_REALIZATION_COMPARISON
#    Run the same simulation for the baseline and three alternative satellite
#    pools. Compare DCLARA with ALLA with SAACB using a grouped PLR bar chart
#    and print the complete metric summary.
#
# 22 RUN_UE_SPATIAL_DISTRIBUTION_COMPARISON
#    Keep the virtual-UE reference uniform and sweep the actual UE normalized
#    enclosed-area distribution u ~ Beta(1, b),
#    b in {1, 1.25, 1.5, 1.75, 2},
#    through the existing
#    offline per-RAO evaluation of transmission success probability theta and
#    its prediction error.
#
# 23 RUN_BETA_SPATIAL_MISMATCH_COMPARISON
#    Use the same Beta sweep in the full simulator and compare the PLR of
#    DCLARA with the integrated ALLA with SAACB baseline.
#
# =============================================================================
EXPERIMENT_CODE = 2
SIM_SECONDS = 5
SIM_RHO_VALUES = np.array([1.0,1.5,2.0,2.5,3.0])
# Kept separate because this diagnostic intentionally spans a much wider load
# range than the rho values used by the comparison experiments.
OFFERED_LOAD_RHO_VALUES = np.array(
    [0.1,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
)
EXPERIMENT_SWITCHES = {
    0: "Single-run diagnostics",
    1: "Integrated scheme comparison",
    2: "Backoff probability 3D bar chart",
    3: "Delay-budget distribution comparison",
    4: "Backoff control scheme comparison",
    5: "Fixed versus adaptive epsilon comparison",
    6: "Satellite selection scheme comparison",
    7: "Controller-side estimation validation",
    8: "Satellite selection component analysis",
    9: "Group-level statistics validation",
    10: "ALLA eta tuning",
    11: "Normalized offered-load range",
    12: "Satellite selection concentration",
    13: "Load estimator performance",
    14: "Real and SS-predicted collision rate comparison",
    15: "Service radius comparison",
    16: "Backoff optimizer initial-guess sensitivity",
    17: "Orbit plane count comparison",
    18: "Top-k grouping policy analysis",
    19: "Top-k grouping comparison across orbit plane counts",
    20: "Top-5 satellite selection shares over time",
    21: "Satellite-pool realization comparison",
    22: "Beta spatial-distribution theta-prediction comparison",
    23: "Beta spatial-distribution PLR comparison",
}
if EXPERIMENT_CODE not in EXPERIMENT_SWITCHES:
    raise ValueError(f"Unknown EXPERIMENT_CODE: {EXPERIMENT_CODE}")

RUN_ALL = EXPERIMENT_CODE == 1
RHO_SWEEP_PB = EXPERIMENT_CODE == 2
RUN_QOS_DISTRIBUTION_COMPARISON = EXPERIMENT_CODE == 3
RUN_RHO_SWEEP = EXPERIMENT_CODE == 4
RUN_FIXED_LOAD_IMBALANCE_SWEEP = EXPERIMENT_CODE == 5
RUN_SATELLITE_SELECTION_SWEEP = EXPERIMENT_CODE == 6
RUN_ESTIMATION_VALIDATION_RHO_SWEEP = EXPERIMENT_CODE == 7
RUN_SATELLITE_SELECTION_PERFORMANCE = EXPERIMENT_CODE == 8 #Different epsilon values
RUN_GROUP_LEVEL_STATISTICS = EXPERIMENT_CODE == 9
RUN_ALLA_ETA_SWEEP = EXPERIMENT_CODE == 10
RUN_OFFERED_LOAD_RHO_SWEEP = EXPERIMENT_CODE == 11
RUN_SATELLITE_SELECTION_CONCENTRATION = EXPERIMENT_CODE == 12
RUN_LOAD_ESTIMATOR_PERFORMANCE = EXPERIMENT_CODE == 13
RUN_COLLISION_RATE_COMPARISON = EXPERIMENT_CODE == 14
RUN_SERVICE_RADIUS_COMPARISON = EXPERIMENT_CODE == 15
RUN_BACKOFF_INITIAL_GUESS_SENSITIVITY = EXPERIMENT_CODE == 16
RUN_ORBIT_PLANE_COMPARISON = EXPERIMENT_CODE == 17
RUN_TOP_K_GROUPING_ANALYSIS = EXPERIMENT_CODE == 18
RUN_TOP_K_ORBIT_PLANE_COMPARISON = EXPERIMENT_CODE == 19
RUN_SATELLITE_SELECTION_TOP5_OVER_TIME = EXPERIMENT_CODE == 20
RUN_SATELLITE_POOL_REALIZATION_COMPARISON = EXPERIMENT_CODE == 21
RUN_UE_SPATIAL_DISTRIBUTION_COMPARISON = EXPERIMENT_CODE == 22
RUN_BETA_SPATIAL_MISMATCH_COMPARISON = EXPERIMENT_CODE == 23


def validate_beta_spatial_sampler(
    center,
    radius_km,
    beta_values,
    seed,
    sample_count=100000,
):
    """Validate the enclosed-area Beta sampler before a spatial experiment."""
    legacy_uniform = main.generate_ue_locations(
        sample_count,
        center=center,
        radius_km=radius_km,
        distribution="uniform",
        random_generator=np.random.RandomState(seed),
    )
    beta_one = main.generate_ue_locations(
        sample_count,
        center=center,
        radius_km=radius_km,
        distribution="beta_enclosed_area",
        random_generator=np.random.RandomState(seed),
        beta_b=1.0,
    )
    if not np.array_equal(legacy_uniform, beta_one):
        raise AssertionError(
            "b=1 locations differ from the legacy uniform-disk sampler for "
            "the same seed."
        )

    print("\n--- Beta Spatial Sampler Sanity Checks ---")
    print(
        f"{'b':>5} | {'Sample E[u]':>12} | {'Theory E[u]':>12} | "
        f"{'Max u':>10} | {'Angular mean':>12}"
    )
    print("-" * 62)
    for beta_b in beta_values:
        locations = main.generate_ue_locations(
            sample_count,
            center=center,
            radius_km=radius_km,
            distribution="beta_enclosed_area",
            random_generator=np.random.RandomState(seed),
            beta_b=float(beta_b),
        )
        north_km = (locations[:, 0] - center[0]) * 111.0
        east_km = (locations[:, 1] - center[1]) * 100.0
        normalized_radius_squared = (
            north_km ** 2 + east_km ** 2
        ) / float(radius_km) ** 2
        angles = np.arctan2(north_km, east_km)
        sample_mean = float(np.mean(normalized_radius_squared))
        theoretical_mean = 1.0 / (1.0 + float(beta_b))
        maximum_u = float(np.max(normalized_radius_squared))
        angular_mean = float(np.abs(np.mean(np.exp(1j * angles))))
        print(
            f"{float(beta_b):5g} | {sample_mean:12.6f} | "
            f"{theoretical_mean:12.6f} | {maximum_u:10.6f} | "
            f"{angular_mean:12.6f}"
        )
        if abs(sample_mean - theoretical_mean) > 0.01:
            raise AssertionError(
                f"Beta sampler mean check failed for b={beta_b:g}."
            )
        if maximum_u > 1.0 + 1e-10:
            raise AssertionError(
                f"Beta sampler boundary check failed for b={beta_b:g}."
            )
        if angular_mean > 0.01:
            raise AssertionError(
                f"Beta sampler angular-uniformity check failed for b={beta_b:g}."
            )
    print("b=1 exact legacy-uniform regression: passed")


if RUN_ALL:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    RHO_VALUES = SIM_RHO_VALUES
    # Proposed satellite selection uses MODE6 adaptive epsilon in combined comparisons.
    MODES = [
        ([6, 1], "DCLARA"),
        ([5, 3], "ALLA with SAACB"),
    ]

    # Backoff settings 2 and 3 are ACB baselines; all other experiment parameters are
    # kept identical to the proposed setting so the PLR curves isolate the backoff controller.
    rho_results = {label: [] for _, label in MODES}
    for mode, label in MODES:
        for rho in RHO_VALUES:
            print(f"\nRunning PLR arrival-rate sweep: {label}, arrival rate={rho}")
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
                "average_deadline_budget_utilization": run_history.get(
                    "average_deadline_budget_utilization",
                    np.nan,
                ),
                "final_n_estimate": final_n_estimate,
            })

    plt.figure(figsize=(10, 6))
    for _, label in MODES:
        rho_axis = np.array([item["rho"] for item in rho_results[label]])
        plr_values = np.array([item["plr"] for item in rho_results[label]])
        plt.plot(rho_axis, plr_values, marker="o", linewidth=1.6, label=label)
    plt.title("PLR Comparison under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
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
    plt.title("Throughput Comparison under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
    plt.ylabel("Average Throughput (packets/second)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for _, label in MODES:
        rho_axis = np.array([item["rho"] for item in rho_results[label]])
        utilization_values = np.array([
            item["average_deadline_budget_utilization"] * 100.0
            for item in rho_results[label]
        ])
        plt.plot(rho_axis, utilization_values, marker="o", linewidth=1.6, label=label)
    plt.title("Deadline Budget Utilization under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
    plt.ylabel("Average deadline budget utilized (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Lambda Sweep Complete ---")
    for _, label in MODES:
        for item in rho_results[label]:
            print(
                f"{label}, arrival rate={item['rho']:.4f}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"deadline_budget_utilization="
                f"{item['average_deadline_budget_utilization'] * 100:.2f}%"
                + (f", final_N={item['final_n_estimate']:.2f}" if np.isfinite(item["final_n_estimate"]) else "")
            )
    raise SystemExit

if RHO_SWEEP_PB:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    MODE = [1, 1]
    RHO_VALUES = SIM_RHO_VALUES

    pb_results = []
    for rho in RHO_VALUES:
        print(f"\nRunning p_b arrival-rate sweep: rho_s={rho}")
        avg_throughput, plr, n_history, actual_pi, observe_pi, load_imbalance_history, run_history = main.main(
            rho,
            SECONDS,
            NUM_UE,
            MODE,
            SEED,
            IMBALANCE_EPSILON,
            USE_REAL_PS=USE_REAL_PS,
        )

        p_b_history = np.asarray(run_history.get("p_b_history", []), dtype=float)
        average_p_b = np.full(20, np.nan)
        if p_b_history.size > 0:
            if p_b_history.ndim == 1:
                p_b_history = p_b_history.reshape(1, -1)
            state_count = min(20, p_b_history.shape[1])
            average_p_b[:state_count] = np.mean(p_b_history[:, :state_count], axis=0)

        pb_results.append({
            "rho": rho,
            "average_p_b": average_p_b,
            "plr": plr,
            "throughput": avg_throughput,
            "final_n_estimate": n_history[-1] if len(n_history) > 0 else np.nan,
        })

    rho_axis = np.array([item["rho"] for item in pb_results])
    average_p_b_matrix = np.vstack([item["average_p_b"] for item in pb_results])

    remaining_rao_axis = np.arange(1, 21, dtype=float)
    if not np.all(np.isfinite(average_p_b_matrix)):
        raise ValueError(
            "The 3D bar chart requires 20 finite backoff probabilities "
            "for every arrival rate."
        )
    remaining_rao_grid, rho_grid = np.meshgrid(
        remaining_rao_axis,
        rho_axis,
    )
    bar_width = 0.62
    if len(rho_axis) > 1:
        bar_depth = 0.42 * float(np.min(np.diff(np.sort(rho_axis))))
    else:
        bar_depth = 0.2
    x_positions = remaining_rao_grid.ravel() - bar_width / 2.0
    y_positions = rho_grid.ravel() - bar_depth / 2.0
    z_positions = np.zeros(average_p_b_matrix.size, dtype=float)
    bar_heights = average_p_b_matrix.ravel()

    figure = plt.figure(figsize=(12, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.bar3d(
        x_positions,
        y_positions,
        z_positions,
        bar_width,
        bar_depth,
        bar_heights,
        color="#6FA8DC",
        edgecolor="#4F5D75",
        linewidth=0.25,
        shade=True,
    )
    axis.set(
        title="Average Backoff Probability under Different Arrival Rates",
        xlabel="",
        ylabel="",
        zlabel="",
        xlim=(20.6, 0.4),
        ylim=(
            float(np.min(rho_axis)) - bar_depth,
            float(np.max(rho_axis)) + bar_depth,
        ),
        zlim=(0.0, 1.0),
        xticks=[1, 5, 10, 15, 20],
        yticks=rho_axis,
        zticks=np.linspace(0.0, 1.0, 6),
    )
    axis.set_yticklabels([f"{rho:g}" for rho in rho_axis])
    axis.view_init(elev=25, azim=55)
    axis.set_box_aspect((1.8, 1.2, 0.8))
    # mplot3d chooses native y/z-axis edges differently across Matplotlib
    # versions. Hide those version-dependent axes and draw fixed front-left
    # axes in data coordinates so local and remote renders remain identical.
    axis.yaxis.line.set_visible(False)
    axis.zaxis.line.set_visible(False)
    axis.tick_params(axis="y", which="both", length=0)
    axis.tick_params(axis="z", which="both", length=0)
    for tick_label in axis.get_yticklabels():
        tick_label.set_alpha(0.0)
    for tick_label in axis.get_zticklabels():
        tick_label.set_alpha(0.0)
    custom_z_axis_x = remaining_rao_axis[0] - 0.45
    custom_z_axis_y = float(np.min(rho_axis)) - bar_depth
    custom_y_axis_end = float(np.max(rho_axis)) + bar_depth
    axis.plot(
        [custom_z_axis_x, custom_z_axis_x],
        [custom_z_axis_y, custom_y_axis_end],
        [0.0, 0.0],
        color="black",
        linewidth=1.0,
    )
    for rho_tick in rho_axis:
        axis.plot(
            [custom_z_axis_x, custom_z_axis_x + 0.28],
            [rho_tick, rho_tick],
            [0.0, 0.0],
            color="black",
            linewidth=0.8,
        )
        axis.text(
            custom_z_axis_x - 0.20,
            rho_tick,
            0.0,
            f"{rho_tick:g}",
            ha="right",
            va="top",
        )
    z_tick_values = np.linspace(0.0, 1.0, 6)
    axis.plot(
        [custom_z_axis_x, custom_z_axis_x],
        [custom_z_axis_y, custom_z_axis_y],
        [0.0, 1.0],
        color="black",
        linewidth=1.0,
    )
    for z_tick in z_tick_values:
        axis.plot(
            [custom_z_axis_x, custom_z_axis_x + 0.28],
            [custom_z_axis_y, custom_z_axis_y],
            [z_tick, z_tick],
            color="black",
            linewidth=0.8,
        )
        axis.text(
            custom_z_axis_x - 0.20,
            custom_z_axis_y,
            z_tick,
            f"{z_tick:.1f}",
            ha="right",
            va="center",
        )
    axis.text2D(
        0.64,
        0.10,
        "Remaining RAO",
        transform=axis.transAxes,
        rotation=0,
        ha="center",
        va="center",
    )
    axis.text2D(
        0.29,
        0.08,
        "Arrival rate (packets/s)",
        transform=axis.transAxes,
        rotation=0,
        ha="center",
        va="center",
    )
    axis.text2D(
        0.08,
        0.72,
        "Average backoff probability",
        transform=axis.transAxes,
        rotation=0,
        ha="left",
        va="center",
    )
    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.06,
        top=0.90,
    )
    plt.show()

    print("\n--- Rho Sweep p_b Complete ---")
    for item in pb_results:
        pb_summary = ", ".join(
            f"State {idx + 1}={value:.6f}"
            for idx, value in enumerate(item["average_p_b"])
        )
        print(
            f"rho_s={item['rho']:.4f}: "
            f"average_p_b=[{pb_summary}], "
            f"PLR={item['plr']:.4f}, "
            f"throughput={item['throughput']:.2f}, "
            f"final_N={item['final_n_estimate']:.2f}"
        )
    raise SystemExit

if RUN_QOS_DISTRIBUTION_COMPARISON:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    RHO = 1.5
    # Proposed satellite selection uses MODE6 adaptive epsilon in combined comparisons.
    MODES = [
        ([6, 1], "DCLARA"),
        ([5, 3], "ALLA with SAACB"),
    ]

    def build_qos_distribution(probability_by_rao_budget):
        qos_distribution = np.zeros(20)
        for rao_budget, probability in probability_by_rao_budget.items():
            qos_distribution[rao_budget - 1] = probability
        return qos_distribution

    # The QoS figure uses four physically meaningful delay budgets:
    # 5, 10, 15, and 20 RAOs, corresponding to 0.5, 1.0, 1.5, and 2.0 seconds.
    QOS_DISTRIBUTIONS = [
        (
            "Balanced",
            build_qos_distribution({5: 0.25, 10: 0.25, 15: 0.25, 20: 0.25}),
        ),
        (
            "Non-urgent",
            build_qos_distribution({20: 1.0}),
        ),
        (
            "Urgent",
            build_qos_distribution({5: 1.0}),
        ),
        (
            "Bimodal",
            build_qos_distribution({5: 0.5, 20: 0.5}),
        ),
    ]

    qos_results = {label: [] for _, label in MODES}
    for qos_label, qos_distribution in QOS_DISTRIBUTIONS:
        for mode, label in MODES:
            print(f"\nRunning QoS distribution comparison: {label}, {qos_label}, rho_s={RHO}")
            avg_throughput, plr, n_history, actual_pi, observe_pi, load_imbalance_history, run_history = main.main(
                RHO,
                SECONDS,
                NUM_UE,
                mode,
                SEED,
                IMBALANCE_EPSILON,
                USE_REAL_PS=USE_REAL_PS,
                QOS_DISTRIBUTION=qos_distribution,
            )
            final_n_estimate = n_history[-1] if len(n_history) > 0 else np.nan
            qos_results[label].append({
                "qos_label": qos_label,
                "qos_distribution": qos_distribution,
                "plr": plr,
                "throughput": avg_throughput,
                "average_delay_ms": run_history.get("average_delay_ms", np.nan),
                "final_n_estimate": final_n_estimate,
            })

    x = np.arange(len(QOS_DISTRIBUTIONS))
    bar_width = 0.8 / len(MODES)
    plt.figure(figsize=(12, 6))
    for mode_idx, (_, label) in enumerate(MODES):
        offsets = x - 0.4 + bar_width * (mode_idx + 0.5)
        plr_values = np.array([item["plr"] for item in qos_results[label]])
        plt.bar(offsets, plr_values, width=bar_width, label=label)
    plt.title("PLR Comparison under Different Delay Budget Distributions")
    plt.xlabel("Delay budget distribution")
    plt.ylabel("Packet Loss Rate")
    plt.xticks(x, [label for label, _ in QOS_DISTRIBUTIONS])
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.show()

    print("\n--- QoS Distribution Comparison Complete ---")
    for _, label in MODES:
        for item in qos_results[label]:
            print(
                f"{label}, {item['qos_label']}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"avg_delay_ms={item['average_delay_ms']:.2f}"
                + (f", final_N={item['final_n_estimate']:.2f}" if np.isfinite(item["final_n_estimate"]) else "")
            )
    raise SystemExit

if RUN_RHO_SWEEP:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    RHO_VALUES = SIM_RHO_VALUES
    MODES = [
        ([6, 1], "DCLARA-BC"),
        ([6, 2], "DACB"),
        ([6, 3], "SAACB"),
    ]

    # Backoff settings 2 and 3 are ACB baselines; all other experiment parameters are
    # kept identical to the proposed setting so the PLR curves isolate the backoff controller.
    rho_results = {label: [] for _, label in MODES}
    for mode, label in MODES:
        for rho in RHO_VALUES:
            print(f"\nRunning PLR arrival-rate sweep: {label}, arrival rate={rho}")
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
    plt.title("PLR Comparison under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
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
    plt.title("Throughput Comparison under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
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
    plt.title("Average Delay Comparison under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
    plt.ylabel("Average Delay (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Lambda Sweep Complete ---")
    for _, label in MODES:
        for item in rho_results[label]:
            print(
                f"{label}, arrival rate={item['rho']:.4f}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"avg_delay_ms={item['average_delay_ms']:.2f}"
                + (f", final_N={item['final_n_estimate']:.2f}" if np.isfinite(item["final_n_estimate"]) else "")
            )
    raise SystemExit

if RUN_FIXED_LOAD_IMBALANCE_SWEEP:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    USE_REAL_PS = False
    RHO_VALUES = SIM_RHO_VALUES

    def epsilon_plot_label(epsilon):
        exponent = np.log10(epsilon) if epsilon > 0 else np.nan
        rounded_exponent = int(np.round(exponent)) if np.isfinite(exponent) else None
        if rounded_exponent is not None and np.isclose(epsilon, 10.0 ** rounded_exponent):
            return rf"$\epsilon=10^{{{rounded_exponent}}}$"
        return rf"$\epsilon={epsilon:g}$"

    def epsilon_text_label(epsilon):
        exponent = np.log10(epsilon) if epsilon > 0 else np.nan
        rounded_exponent = int(np.round(exponent)) if np.isfinite(exponent) else None
        if rounded_exponent is not None and np.isclose(epsilon, 10.0 ** rounded_exponent):
            return f"\u03b5=10^{rounded_exponent}"
        return f"\u03b5={epsilon:g}"

    EXPERIMENTS = [
        ([1, 1], epsilon_plot_label(1e-4), epsilon_text_label(1e-4), 1e-4),
        ([1, 1], epsilon_plot_label(1e-3), epsilon_text_label(1e-3), 1e-3),
        ([1, 1], epsilon_plot_label(1e-2), epsilon_text_label(1e-2), 1e-2),
        ([1, 1], epsilon_plot_label(1e-1), epsilon_text_label(1e-1), 1e-1),
        ([6, 1], r"Adaptive $\epsilon$", "Adaptive \u03b5", 0.1),
    ]

    constraint_results = {plot_label: [] for _, plot_label, _, _ in EXPERIMENTS}
    for mode, plot_label, text_label, epsilon in EXPERIMENTS:
        for rho in RHO_VALUES:
            print(
                f"\nRunning load-imbalance constraint sweep: "
                f"{text_label}, arrival rate={rho}"
            )
            avg_throughput, plr, n_history, actual_pi, observe_pi, load_imbalance_history, run_history = main.main(
                rho,
                SECONDS,
                NUM_UE,
                mode,
                SEED,
                epsilon,
                USE_REAL_PS=USE_REAL_PS,
            )
            constraint_results[plot_label].append({
                "rho": rho,
                "text_label": text_label,
                "epsilon": epsilon,
                "plr": plr,
                "throughput": avg_throughput,
                "average_delay_ms": run_history.get("average_delay_ms", np.nan),
                "final_n_estimate": n_history[-1] if len(n_history) > 0 else np.nan,
            })

    plt.figure(figsize=(10, 6))
    for _, plot_label, _, _ in EXPERIMENTS:
        rho_axis = np.array([item["rho"] for item in constraint_results[plot_label]])
        plr_values = np.array([item["plr"] for item in constraint_results[plot_label]])
        plt.plot(rho_axis, plr_values, marker="o", linewidth=1.6, label=plot_label)
    plt.title("PLR under Fixed and Adaptive Load-Imbalance Constraints")
    plt.xlabel("Arrival rate (packets/s)")
    plt.ylabel("Packet Loss Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Fixed Load-Imbalance Constraint Sweep Complete ---")
    for _, plot_label, _, _ in EXPERIMENTS:
        for item in constraint_results[plot_label]:
            print(
                f"{item['text_label']}, arrival rate={item['rho']:.4f}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"avg_delay_ms={item['average_delay_ms']:.2f}, "
                f"final_N={item['final_n_estimate']:.2f}"
            )
    raise SystemExit

if RUN_SATELLITE_SELECTION_SWEEP:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    RHO_VALUES = SIM_RHO_VALUES

    EXPERIMENTS = [
        ([3, 3], "VU", {}),
        ([5, 3], "ALLA", {}),
        ([6, 3], "DCLARA-SS", {}),
        ([7, 3], "LLA", {}),
    ]

    # Satellite-selection baselines keep the proposed backoff controller fixed
    # so the PLR curves isolate the satellite selection policy.
    selection_results = {label: [] for _, label, _ in EXPERIMENTS}
    for mode, label, extra_kwargs in EXPERIMENTS:
        for rho in RHO_VALUES:
            print(
                f"\nRunning satellite selection arrival-rate sweep: "
                f"{label}, arrival rate={rho}"
            )
            avg_throughput, plr, n_history, actual_pi, observe_pi, load_imbalance_history, run_history = main.main(
                rho,
                SECONDS,
                NUM_UE,
                mode,
                SEED,
                IMBALANCE_EPSILON,
                USE_REAL_PS=USE_REAL_PS,
                **extra_kwargs,
            )
            final_n_estimate = n_history[-1] if len(n_history) > 0 else np.nan
            selection_results[label].append({
                "rho": rho,
                "plr": plr,
                "throughput": avg_throughput,
                "average_delay_ms": run_history.get("average_delay_ms", np.nan),
                "final_n_estimate": final_n_estimate,
            })

    plr_by_label = {
        label: np.array([item["plr"] for item in selection_results[label]])
        for _, label, _ in EXPERIMENTS
    }
    rho_axis = np.array([item["rho"] for item in selection_results[EXPERIMENTS[0][1]]])
    normal_plr = np.concatenate([
        values for label, values in plr_by_label.items() if label != "LLA"
    ])
    lla_plr = plr_by_label["LLA"]

    if np.min(lla_plr) > np.max(normal_plr):
        fig, (ax_high, ax_low) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6),
            gridspec_kw={"height_ratios": [1, 2], "hspace": 0.05},
        )
        for _, label, _ in EXPERIMENTS:
            for axis in (ax_high, ax_low):
                axis.plot(rho_axis, plr_by_label[label], marker="o", linewidth=1.6, label=label)

        gap = np.min(lla_plr) - np.max(normal_plr)
        normal_pad = max(0.01, 0.1 * np.ptp(normal_plr))
        lla_pad = max(0.01, 0.1 * np.ptp(lla_plr))
        ax_low.set_ylim(max(0.0, np.min(normal_plr) - normal_pad), np.max(normal_plr) + 0.2 * gap)
        ax_high.set_ylim(np.min(lla_plr) - 0.2 * gap, min(1.0, np.max(lla_plr) + lla_pad))
        ax_high.spines["bottom"].set_visible(False)
        ax_low.spines["top"].set_visible(False)
        ax_high.tick_params(labeltop=False, bottom=False)

        diagonal = 0.008
        for axis, y in ((ax_high, 0), (ax_low, 1)):
            axis.plot((-diagonal, diagonal), (y - diagonal, y + diagonal),
                      transform=axis.transAxes, color="black", clip_on=False)
            axis.plot((1 - diagonal, 1 + diagonal), (y - diagonal, y + diagonal),
                      transform=axis.transAxes, color="black", clip_on=False)

        ax_high.grid(True, alpha=0.3)
        ax_low.grid(True, alpha=0.3)
        ax_high.legend()
        fig.suptitle("Satellite Selection PLR Comparison under Different Arrival Rates")
        fig.supxlabel("Arrival rate (packets/s)")
        fig.supylabel("Packet Loss Rate")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        for _, label, _ in EXPERIMENTS:
            plt.plot(rho_axis, plr_by_label[label], marker="o", linewidth=1.6, label=label)
        plt.title("Satellite Selection PLR Comparison under Different Arrival Rates")
        plt.xlabel("Arrival rate (packets/s)")
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
    plt.title("Satellite Selection Throughput Comparison under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
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
    plt.title("Satellite Selection Average Delay Comparison under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
    plt.ylabel("Average Delay (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Satellite Selection Lambda Sweep Complete ---")
    for _, label, _ in EXPERIMENTS:
        for item in selection_results[label]:
            print(
                f"{label}, arrival rate={item['rho']:.4f}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"avg_delay_ms={item['average_delay_ms']:.2f}, "
                f"final_N={item['final_n_estimate']:.2f}"
            )
    raise SystemExit

if RUN_ESTIMATION_VALIDATION_RHO_SWEEP:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    MODE = [6, 1]
    IMBALANCE_EPSILON = 0.01
    USE_REAL_PS = False
    ADAPTIVE_EPSILON_ALPHA = 2.0
    RHO_VALUES = SIM_RHO_VALUES

    validation_results = []
    for rho in RHO_VALUES:
        print(f"\nRunning estimation validation arrival-rate sweep: arrival rate={rho}")
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
            "ps_sample_count": len(ps_history),
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
    idle_probability_error = pi_error_matrix[:, 0]
    non_idle_state_average_error = np.nanmean(pi_error_matrix[:, 1:], axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        rho_axis,
        idle_probability_error,
        marker="o",
        linewidth=1.6,
        label="Idle probability error",
    )
    plt.plot(
        rho_axis,
        non_idle_state_average_error,
        marker="s",
        linewidth=1.6,
        label="Average non-idle state error",
    )
    plt.axhline(y=0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.title("State Probability Estimation Error under Different Arrival Rates")
    plt.xlabel("Arrival rate (packets/s)")
    plt.ylabel("Mean error (estimated minus true)")
    plt.grid(True, alpha=0.3)
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

    print("\n--- Predicted Transmission Success Probability Error ---")
    print(f"{'Arrival rate':>12} | {'p_s MAE':>10} | {'RAO samples':>11}")
    print("-" * 41)
    for item in validation_results:
        print(
            f"{item['rho']:12.4f} | "
            f"{item['ps_mae']:10.6f} | "
            f"{item['ps_sample_count']:11d}"
        )

    print("\n--- Estimation Validation Rho Sweep Complete ---")
    for item in validation_results:
        pi_summary = ", ".join(
            f"{'Idle' if idx == 0 else f'State {idx}'}={value:.6f}"
            for idx, value in enumerate(item["pi_error_by_state"])
        )
        print(
            f"arrival rate={item['rho']:.4f}: "
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
    SECONDS = SIM_SECONDS
    SEED = 42
    USE_REAL_PS = False
    FIXED_EPSILON_MODE = [1, 1]
    ADAPTIVE_EPSILON_MODE = [6, 1]
    FIXED_EPSILON_RHO = 1.0
    FIXED_EPSILON_VALUES = [1e-4, 1e-3, 1e-2, 1e-1]
    ADAPTIVE_EPSILON_RHO_VALUES = SIM_RHO_VALUES
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
        print(f"\nRunning adaptive-epsilon trajectory: arrival rate={rho}")
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
            label=rf"$\rho_s={item['rho']:g}$",
        )
    plt.title("Adaptive Imbalance Epsilon under Different Arrival Rates")
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
            f"adaptive arrival rate={item['rho']:.4f}: "
            f"final_epsilon={final_epsilon:.6g}, "
            f"PLR={item['plr']:.4f}, "
            f"throughput={item['throughput']:.2f}"
        )
    raise SystemExit

if RUN_GROUP_LEVEL_STATISTICS:
    GROUP_TABLE_FILENAME = "group_ps_table.npz"
    ANALYSIS_SECONDS = SIM_SECONDS
    SEGMENT_SIZE_RAOS = 100
    SEGMENT_TOP_K = 3

    def format_group_label(group):
        return str(tuple(int(satellite_id) + 1 for satellite_id in group))

    group_weight_table, group_ps_table, _ = main.load_ps_tables(
        GROUP_TABLE_FILENAME
    )
    with np.load(GROUP_TABLE_FILENAME, allow_pickle=True) as group_table_data:
        TRAO_MS = (
            int(group_table_data["trao_ms"])
            if "trao_ms" in group_table_data.files
            else 100
        )

    max_rao_count = ANALYSIS_SECONDS * 1000 // TRAO_MS
    used_rao_count = min(max_rao_count, len(group_weight_table))
    if used_rao_count <= 0:
        raise ValueError("No group-level statistics are available for plotting.")

    group_weight_table = group_weight_table[:used_rao_count]
    group_ps_table = group_ps_table[:used_rao_count]
    group_counts = np.array([
        len(groups)
        for groups in group_weight_table
    ])

    first_ps_table = next(
        (table for table in group_ps_table if len(table) > 0),
        None,
    )
    satellite_count = (
        len(next(iter(first_ps_table.values())))
        if first_ps_table is not None
        else 0
    )

    print("\n--- Group-Level Statistics Summary ---")
    print(f"Satellite count: {satellite_count}")
    print(f"Analysis window: first {ANALYSIS_SECONDS} seconds")
    print(f"RAO duration: {TRAO_MS} ms")
    print(f"RAO count: {len(group_counts)}")
    print(f"Average groups per RAO: {np.mean(group_counts):.4f}")
    print(f"Median groups per RAO: {np.median(group_counts):.4f}")
    print(f"Min groups per RAO: {np.min(group_counts)}")
    print(f"Max groups per RAO: {np.max(group_counts)}")

    plt.figure(figsize=(12, 5))
    plt.plot(range(len(group_counts)), group_counts, linewidth=1.4)
    plt.title("Number of Preselection Groups per RAO")
    plt.xlabel("RAO Index")
    plt.ylabel("Group Count")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    integrated_group_weights = Counter()
    for weights in group_weight_table:
        integrated_group_weights.update(weights)
    average_group_weights = {
        group: total_weight / len(group_weight_table)
        for group, total_weight in integrated_group_weights.items()
    }

    segment_records = []
    segment_top_groups = set()
    for start in range(0, len(group_weight_table), SEGMENT_SIZE_RAOS):
        end = min(start + SEGMENT_SIZE_RAOS, len(group_weight_table))
        segment_weights = Counter()
        for weights in group_weight_table[start:end]:
            segment_weights.update(weights)

        segment_length = end - start
        averaged_weights = {
            group: total_weight / segment_length
            for group, total_weight in segment_weights.items()
        }
        top_groups = sorted(
            averaged_weights.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:SEGMENT_TOP_K]
        segment_records.append((start, end - 1, averaged_weights, top_groups))
        segment_top_groups.update(group for group, _ in top_groups)

    heatmap_groups = sorted(
        segment_top_groups,
        key=lambda group: average_group_weights.get(group, 0.0),
        reverse=True,
    )
    if heatmap_groups:
        heatmap = np.array([
            [
                averaged_weights.get(group, np.nan)
                if group in {top_group for top_group, _ in top_groups}
                else np.nan
                for _, _, averaged_weights, top_groups in segment_records
            ]
            for group in heatmap_groups
        ])
        x_labels = [
            f"{start}-{end}"
            for start, end, _, _ in segment_records
        ]
        y_labels = [
            format_group_label(group)
            for group in heatmap_groups
        ]

        fig_width = max(12, len(x_labels) * 0.7)
        fig_height = max(6, len(y_labels) * 0.35)
        plt.figure(figsize=(fig_width, fig_height))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")
        plt.imshow(
            np.ma.masked_invalid(heatmap),
            aspect="auto",
            cmap=cmap,
        )
        plt.colorbar(label="Average group weight in each segment")
        plt.title("Weight distribution of the dominant groups")
        plt.xlabel("RAO Segment")
        plt.ylabel("Group")
        plt.xticks(
            range(len(x_labels)),
            x_labels,
            rotation=45,
            ha="right",
        )
        plt.yticks(range(len(y_labels)), y_labels)
        plt.tight_layout()
        plt.show()

    raise SystemExit


if RUN_ALLA_ETA_SWEEP:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    RHO_VALUES = SIM_RHO_VALUES
    MODE = [5, 3]
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    ETA_VALUES = np.array([0.5,1,2,4,8,16])

    eta_results = {}
    for rho in RHO_VALUES:
        eta_results[rho] = []
        for eta in ETA_VALUES:
            print(f"\nRunning ALLA eta sweep: rho={rho:g}, eta={eta:g}")
            _, plr, _, _, _, _, _ = main.main(
                rho,
                SECONDS,
                NUM_UE,
                MODE,
                SEED,
                IMBALANCE_EPSILON,
                USE_REAL_PS=USE_REAL_PS,
                LOAD_AWARE_ETA=eta,
            )
            eta_results[rho].append(plr)

    plt.figure(figsize=(10, 6))
    for rho in RHO_VALUES:
        plt.plot(ETA_VALUES, eta_results[rho], marker="o", label=rf"$\rho_s={rho:g}$")
    plt.title(r"ALLA PLR under Different $\eta$ and Loads")
    plt.xlabel(r"ALLA $\eta$")
    plt.ylabel("Packet Loss Rate")
    plt.xscale("log")
    plt.xticks(ETA_VALUES, [f"{eta:g}" for eta in ETA_VALUES])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- ALLA Eta Sweep Complete ---")
    for rho in RHO_VALUES:
        for eta, plr in zip(ETA_VALUES, eta_results[rho]):
            print(f"rho={rho:g}, eta={eta:g}: PLR={plr:.4f}")
    raise SystemExit


if RUN_OFFERED_LOAD_RHO_SWEEP:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    MODE = [6, 1]
    IMBALANCE_EPSILON = 0.01
    USE_REAL_PS = False

    offered_load_results = []
    for rho in OFFERED_LOAD_RHO_VALUES:
        print(f"\nRunning normalized offered-load sweep: rho_s={rho:g}")
        _, _, _, _, _, _, run_history = main.main(
            rho,
            SECONDS,
            NUM_UE,
            MODE,
            SEED,
            IMBALANCE_EPSILON,
            USE_REAL_PS=USE_REAL_PS,
        )
        offered_arrival_history = np.asarray(
            run_history.get("offered_arrival_history", []),
            dtype=float,
        )
        total_resources_kz = float(run_history.get("total_resources_kz", np.nan))
        if len(offered_arrival_history) > 0 and total_resources_kz > 0:
            average_normalized_offered_load = float(
                np.mean(offered_arrival_history / total_resources_kz)
            )
        else:
            average_normalized_offered_load = np.nan
        offered_load_results.append({
            "rho": rho,
            "average_normalized_offered_load": average_normalized_offered_load,
            "average_offered_packets_per_rao": (
                float(np.mean(offered_arrival_history))
                if len(offered_arrival_history) > 0
                else np.nan
            ),
            "total_resources_kz": total_resources_kz,
        })

    rho_axis = np.array([item["rho"] for item in offered_load_results])
    normalized_offered_load_values = np.array([
        item["average_normalized_offered_load"]
        for item in offered_load_results
    ])

    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(
        rho_axis,
        normalized_offered_load_values,
        marker="o",
        linewidth=1.8,
        color="#2980b9",
        label="Normalized offered load",
    )
    plt.axhline(
        1.0,
        color="#c0392b",
        linestyle="--",
        linewidth=1.2,
        label="Full resource utilization",
    )
    if len(SIM_RHO_VALUES) > 0:
        comparison_min = float(np.min(SIM_RHO_VALUES))
        comparison_max = float(np.max(SIM_RHO_VALUES))
        plt.axvspan(
            comparison_min,
            comparison_max,
            color="#f1c40f",
            alpha=0.16,
            label=(
                f"Selected comparison range: "
                f"{comparison_min:g} to {comparison_max:g} packets/s"
            ),
        )
    plt.title("Average Normalized Offered Load")
    plt.xlabel("Per-UE arrival rate (packets/s)")
    plt.ylabel("Normalized offered load")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Normalized Offered-Load Sweep Complete ---")
    for item in offered_load_results:
        print(
            f"rho_s={item['rho']:g}: "
            f"average_offered_packets_per_RAO="
            f"{item['average_offered_packets_per_rao']:.4f}, "
            f"KZ={item['total_resources_kz']:.0f}, "
            f"average_offered_load_over_KZ="
            f"{item['average_normalized_offered_load']:.6f}"
        )
    raise SystemExit


if RUN_SATELLITE_SELECTION_CONCENTRATION:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    RHO = 1.5
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    BACKOFF_MODE = 3
    CONCENTRATION_AVERAGE_WINDOW_RAOS = 100
    EXPERIMENTS = [
        ([6, BACKOFF_MODE], "DCLARA-SS"),
        ([3, BACKOFF_MODE], "VU"),
        #([4, BACKOFF_MODE], "HE"),
        ([5, BACKOFF_MODE], "ALLA"),
        ([7, BACKOFF_MODE], "LLA"),
    ]

    concentration_results = {}
    for mode, label in EXPERIMENTS:
        print(
            f"\nRunning UE satellite-selection concentration analysis: "
            f"{label}, arrival rate={RHO:g}"
        )
        _, _, _, _, _, _, run_history = main.main(
            RHO,
            SECONDS,
            NUM_UE,
            mode,
            SEED,
            IMBALANCE_EPSILON,
            USE_REAL_PS=USE_REAL_PS,
        )
        selection_history = run_history.get(
            "ue_satellite_selection_history",
            [],
        )
        concentration_results[label] = {
            "rao_index": np.array(
                [item["time_slot"] for item in selection_history],
                dtype=int,
            ),
            "highest_satellite_share": np.array(
                [item["highest_satellite_share"] for item in selection_history],
                dtype=float,
            ),
            "total_selections": np.array(
                [item["total_selections"] for item in selection_history],
                dtype=int,
            ),
            "most_selected_satellite": np.array(
                [
                    item["most_selected_satellite"]
                    if item["most_selected_satellite"] is not None
                    else -1
                    for item in selection_history
                ],
                dtype=int,
            ),
        }

    def average_by_rao_window(rao_index, values, window_size):
        if window_size <= 1:
            return rao_index, values

        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return np.array([], dtype=float), np.array([], dtype=float)

        valid_rao_index = rao_index[finite_mask]
        valid_values = values[finite_mask]
        window_ids = valid_rao_index // window_size
        averaged_rao_index = []
        averaged_values = []

        for window_id in np.unique(window_ids):
            in_window = window_ids == window_id
            averaged_rao_index.append(np.mean(valid_rao_index[in_window]))
            averaged_values.append(np.mean(valid_values[in_window]))

        return (
            np.array(averaged_rao_index, dtype=float),
            np.array(averaged_values, dtype=float),
        )

    plt.figure(figsize=(11, 6), dpi=120)
    for _, label in EXPERIMENTS:
        item = concentration_results[label]
        averaged_rao_index, averaged_share = average_by_rao_window(
            item["rao_index"],
            item["highest_satellite_share"],
            CONCENTRATION_AVERAGE_WINDOW_RAOS,
        )
        plt.plot(
            averaged_rao_index,
            averaged_share,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=label,
        )
    plt.title(
        "Maximum Satellite Selection Share over Time "
        f"({CONCENTRATION_AVERAGE_WINDOW_RAOS}-RAO Average)"
    )
    plt.xlabel(f"RAO index ({CONCENTRATION_AVERAGE_WINDOW_RAOS}-RAO average)")
    plt.ylabel("Highest satellite selection share")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Satellite Selection Concentration Analysis Complete ---")
    for _, label in EXPERIMENTS:
        item = concentration_results[label]
        valid_mask = np.isfinite(item["highest_satellite_share"])
        valid_shares = item["highest_satellite_share"][valid_mask]
        valid_dominant_ids = item["most_selected_satellite"][valid_mask]
        if len(valid_shares) == 0:
            print(f"{label}: no UE satellite selections were recorded")
            continue
        dominant_id_counts = np.bincount(valid_dominant_ids)
        most_frequent_dominant_id = int(np.argmax(dominant_id_counts))
        dominant_id_frequency = (
            dominant_id_counts[most_frequent_dominant_id]
            / len(valid_dominant_ids)
        )
        print(
            f"{label}: "
            f"average_highest_share={np.mean(valid_shares):.4f}, "
            f"maximum_highest_share={np.max(valid_shares):.4f}, "
            f"average_UE_selections_per_RAO="
            f"{np.mean(item['total_selections']):.2f}, "
            f"most_frequent_dominant_satellite="
            f"{most_frequent_dominant_id}, "
            f"dominant_in_valid_RAOs={dominant_id_frequency:.4f}"
        )
    raise SystemExit


if RUN_LOAD_ESTIMATOR_PERFORMANCE:
    PREAMBLE_COUNT = 54
    NMAX = 1000
    TRUE_LOAD_VALUES = np.arange(1, 302, 10)
    NUM_TRIALS = 50
    SEED = 42

    np.random.seed(SEED)
    expected_tables = Load_estimator.precompute_expected_tables(
        PREAMBLE_COUNT,
        NMAX,
    )

    ground_truth = []
    estimations = []
    for true_load in TRUE_LOAD_VALUES:
        trial_estimates = []
        for _ in range(NUM_TRIALS):
            selections = np.random.randint(
                0,
                PREAMBLE_COUNT,
                size=true_load,
            )
            counts = np.bincount(
                selections,
                minlength=PREAMBLE_COUNT,
            )
            idle_preambles = np.sum(counts == 0)
            successful_preambles = np.sum(counts == 1)
            collided_preambles = np.sum(counts > 1)
            estimated_load = Load_estimator.load_estimator(
                np.array([idle_preambles]),
                np.array([successful_preambles]),
                np.array([collided_preambles]),
                expected_tables,
            )
            trial_estimates.append(estimated_load[0])

        ground_truth.append(true_load)
        estimations.append(np.mean(trial_estimates))

    ground_truth = np.array(ground_truth)
    estimations = np.array(estimations)
    absolute_errors = np.abs(estimations - ground_truth)
    desired_load_ratios = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    corresponding_loads = desired_load_ratios * PREAMBLE_COUNT

    fig, (ax_estimate, ax_error) = plt.subplots(
        2,
        1,
        figsize=(11, 10),
    )
    fig.suptitle(
        "MoM Estimator Performance Analysis",
        fontsize=12,
        y=0.98,
    )

    ax_estimate.plot(
        ground_truth,
        ground_truth,
        "r--",
        label="Ideal (Ground Truth)",
    )
    ax_estimate.scatter(
        ground_truth,
        estimations,
        color="blue",
        label="MoM Estimation",
    )
    ax_estimate.set_ylabel("Estimated Load")
    ax_estimate.legend(loc="upper left")
    ax_estimate.grid(True, alpha=0.3)
    ax_estimate.set_xlim(left=-10, right=315)

    estimate_ratio_axis = ax_estimate.twiny()
    estimate_ratio_axis.set_xlim(ax_estimate.get_xlim())
    estimate_ratio_axis.set_xticks(corresponding_loads)
    estimate_ratio_axis.set_xticklabels([
        f"{ratio:.1f}"
        for ratio in desired_load_ratios
    ])
    estimate_ratio_axis.set_xlabel("Load Ratio")
    for load_position in corresponding_loads:
        ax_estimate.axvline(
            x=load_position,
            color="gray",
            linestyle=":",
            alpha=0.4,
        )

    ax_error.plot(
        ground_truth,
        absolute_errors,
        color="purple",
        marker="o",
        linewidth=1.5,
        label="Absolute Error",
    )
    ax_error.axhline(
        y=0,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Zero Error",
    )
    ax_error.set_xlabel("True Load")
    ax_error.set_ylabel("Absolute Error")
    ax_error.legend(loc="upper left")
    ax_error.grid(True, alpha=0.3)
    ax_error.set_xlim(ax_estimate.get_xlim())

    error_ratio_axis = ax_error.twiny()
    error_ratio_axis.set_xlim(ax_error.get_xlim())
    error_ratio_axis.set_xticks(corresponding_loads)
    error_ratio_axis.set_xticklabels([
        f"{ratio:.1f}"
        for ratio in desired_load_ratios
    ])
    for load_position in corresponding_loads:
        ax_error.axvline(
            x=load_position,
            color="gray",
            linestyle=":",
            alpha=0.4,
        )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    plt.show()

    print("\n--- Load Estimator Performance Complete ---")
    print(f"Seed: {SEED}")
    print(f"Mean absolute error: {np.mean(absolute_errors):.4f}")
    raise SystemExit


if RUN_COLLISION_RATE_COMPARISON:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    MODE = [6, 1]
    IMBALANCE_EPSILON = 0.01
    USE_REAL_PS = False
    RHO_VALUES = SIM_RHO_VALUES

    collision_results = []
    for rho in RHO_VALUES:
        print(
            f"\nRunning collision-rate comparison: arrival rate={rho:g}"
        )
        _, _, _, _, _, _, run_history = main.main(
            rho,
            SECONDS,
            NUM_UE,
            MODE,
            SEED,
            IMBALANCE_EPSILON,
            USE_REAL_PS=USE_REAL_PS,
            COLLECT_COLLISION_DIAGNOSTICS=True,
        )
        collision_history = run_history["collision_history"]
        valid_history = [
            item
            for item in collision_history
            if item["total_received_load"] > 0
            and np.isfinite(item["real_collision_rate"])
            and np.isfinite(item["ss_predicted_collision_rate"])
        ]
        if len(valid_history) == 0:
            raise ValueError(
                f"No valid collision samples for arrival rate {rho:g}."
            )

        total_received_load = float(np.sum([
            item["total_received_load"] for item in valid_history
        ]))
        real_collision_rate = float(np.sum([
            item["collision_transmissions"] for item in valid_history
        ]) / total_received_load)
        ss_predicted_collision_rate = float(np.sum([
            item["total_received_load"]
            * item["ss_predicted_collision_rate"]
            for item in valid_history
        ]) / total_received_load)
        collision_results.append({
            "rho": float(rho),
            "real_collision_rate": real_collision_rate,
            "ss_predicted_collision_rate": ss_predicted_collision_rate,
            "samples": len(valid_history),
        })

    rho_axis = np.array([
        item["rho"] for item in collision_results
    ])
    real_values = np.array([
        item["real_collision_rate"] for item in collision_results
    ])
    predicted_values = np.array([
        item["ss_predicted_collision_rate"] for item in collision_results
    ])
    prediction_errors = (predicted_values - real_values) * 100.0
    mean_absolute_error = float(np.mean(np.abs(prediction_errors)))

    figure, axis = plt.subplots(figsize=(10, 6), dpi=120)
    bars = axis.bar(
        rho_axis,
        prediction_errors,
        width=0.3,
        color="#4C78A8",
    )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.bar_label(bars, fmt="%+.2f", padding=3)
    axis.set(
        title="SS Collision Rate Prediction Error",
        xlabel="Arrival rate (packets/s)",
        ylabel="Prediction error (percentage points)",
        xticks=rho_axis,
    )
    axis.text(
        0.98,
        0.95,
        f"MAE = {mean_absolute_error:.2f} percentage points",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    figure.tight_layout()
    plt.show()

    print("\n--- Collision Rate Comparison Complete ---")
    for item in collision_results:
        print(
            f"arrival rate={item['rho']:g}: "
            f"real_collision_rate={item['real_collision_rate']:.6f}, "
            f"ss_predicted_collision_rate="
            f"{item['ss_predicted_collision_rate']:.6f}, "
            f"prediction_error="
            f"{item['ss_predicted_collision_rate'] - item['real_collision_rate']:+.6f}, "
            f"samples={item['samples']}"
        )
    raise SystemExit


if RUN_SERVICE_RADIUS_COMPARISON:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    RHO = 1.5
    MODES = [
        ([6, 1], "DCLARA"),
        ([5, 3], "ALLA with SAACB"),
    ]
    RADIUS_SCENARIOS = [
        (100.0, "group_ps_table_radius_100km.npz"),
        (200.0, "group_ps_table.npz"),
        (300.0, "group_ps_table_radius_300km.npz"),
    ]

    radius_results = {label: [] for _, label in MODES}
    for mode, label in MODES:
        for radius_km, group_table_filename in RADIUS_SCENARIOS:
            print(
                f"\nRunning service-radius comparison: "
                f"radius={radius_km:g} km, method={label}, "
                f"arrival rate={RHO:g}"
            )
            (
                avg_throughput,
                plr,
                n_history,
                _,
                _,
                _,
                run_history,
            ) = main.main(
                RHO,
                SECONDS,
                NUM_UE,
                mode,
                SEED,
                IMBALANCE_EPSILON,
                USE_REAL_PS=USE_REAL_PS,
                SERVICE_RADIUS_KM=radius_km,
                GROUP_TABLE_FILENAME=group_table_filename,
            )
            final_n_estimate = (
                n_history[-1] if len(n_history) > 0 else np.nan
            )
            radius_results[label].append({
                "radius_km": float(radius_km),
                "plr": float(plr),
                "throughput": float(avg_throughput),
                "average_deadline_budget_utilization": float(
                    run_history.get(
                        "average_deadline_budget_utilization",
                        np.nan,
                    )
                ),
                "final_n_estimate": float(final_n_estimate),
            })

    def plot_metric(metric, ylabel, title, scale=1.0):
        figure, axis = plt.subplots(figsize=(10, 6), dpi=120)
        for _, label in MODES:
            results = radius_results[label]
            radius_axis = np.array([item["radius_km"] for item in results])
            values = np.array([item[metric] for item in results]) * scale
            axis.plot(
                radius_axis,
                values,
                marker="o",
                linewidth=1.6,
                label=label,
            )
        axis.set(
            title=title,
            xlabel="Service radius (km)",
            ylabel=ylabel,
            xticks=[radius for radius, _ in RADIUS_SCENARIOS],
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        plt.show()

    plot_metric(
        "plr",
        "Packet loss rate",
        "PLR Comparison under Different Service Radii",
    )
    plot_metric(
        "throughput",
        "Average throughput (packets/second)",
        "Throughput Comparison under Different Service Radii",
    )
    plot_metric(
        "average_deadline_budget_utilization",
        "Average deadline budget utilized (%)",
        "Deadline Budget Utilization under Different Service Radii",
        scale=100.0,
    )

    print("\n--- Service Radius Comparison Complete ---")
    for _, label in MODES:
        for item in radius_results[label]:
            print(
                f"radius={item['radius_km']:g} km, {label}, "
                f"arrival rate={RHO:.4f}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"deadline_budget_utilization="
                f"{item['average_deadline_budget_utilization'] * 100:.2f}%"
                + (
                    f", final_N={item['final_n_estimate']:.2f}"
                    if np.isfinite(item["final_n_estimate"])
                    else ""
                )
            )
    raise SystemExit


if RUN_BACKOFF_INITIAL_GUESS_SENSITIVITY:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    MODE = [6, 1]
    RHO = 1.5
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    NUM_SAMPLED_RAOS = 10
    NUM_RANDOM_INITIAL_GUESSES = 100

    if NUM_RANDOM_INITIAL_GUESSES < 2:
        raise ValueError(
            "NUM_RANDOM_INITIAL_GUESSES must be at least 2."
        )

    print("\nRunning one baseline simulation to collect optimizer inputs.")
    (
        _avg_throughput,
        _plr,
        n_history,
        _actual_pi_history,
        _final_pi,
        _reward_history,
        run_history,
    ) = main.main(
        RHO,
        SECONDS,
        NUM_UE,
        MODE,
        SEED,
        IMBALANCE_EPSILON,
        USE_REAL_PS=USE_REAL_PS,
    )

    ps_records = [
        item for item in run_history["ps_history"]
        if int(item["time_slot"]) >= 2
    ]
    first_eligible_index = int(np.floor(0.2 * len(ps_records)))
    eligible_records = ps_records[first_eligible_index:]
    if len(eligible_records) < NUM_SAMPLED_RAOS:
        raise ValueError(
            "Not enough RAOs with measured channel attempts for the "
            "backoff initial-guess sensitivity test."
        )
    sample_indices = np.linspace(
        0,
        len(eligible_records) - 1,
        NUM_SAMPLED_RAOS,
        dtype=int,
    )
    sampled_records = [eligible_records[index] for index in sample_indices]
    sampled_raos = [int(item["time_slot"]) for item in sampled_records]

    random_generator = np.random.default_rng(SEED)
    initial_guesses = [
        (
            f"Random {guess_index + 1}",
            random_generator.uniform(0.0, 1.0, 20),
        )
        for guess_index in range(NUM_RANDOM_INITIAL_GUESSES)
    ]
    print(
        f"\nGenerated {NUM_RANDOM_INITIAL_GUESSES} random "
        "backoff-vector initial guesses."
    )

    qos_distribution = np.zeros(20, dtype=float)
    qos_distribution[[4, 9, 14, 19]] = 0.25
    rho_rao = 1.0 - np.exp(-RHO * 0.1)
    total_resources = float(run_history["total_resources_kz"])
    sensitivity_by_rao = []

    for ps_record in sampled_records:
        rao = int(ps_record["time_slot"])
        n_tilde = float(n_history[rao])
        p_s = float(ps_record["control"])

        rao_results = {}
        print(
            f"\nRAO {rao}: running "
            f"{NUM_RANDOM_INITIAL_GUESSES} random initial guesses"
        )
        for guess_number, (label, initial_guess) in enumerate(
            initial_guesses,
            start=1,
        ):
            optimized_pb, _ = backoff_control.proposed_backoff_control(
                n_tilde,
                np.asarray(initial_guess, dtype=float),
                rho_rao,
                20,
                qos_distribution,
                p_s,
                1,
                total_resources,
            )
            env = backoff_control.SatelliteEnv(n_tilde, rho_rao)
            p_c, _, _ = env.solve_p_c(
                optimized_pb,
                20,
                qos_distribution,
                p_s,
                1,
                total_resources,
            )
            objective = backoff_control.get_loss(
                optimized_pb,
                p_c,
                p_s,
                qos_distribution,
                20,
            )
            rao_results[label] = {
                "objective": float(objective),
                "p_b": np.asarray(optimized_pb, dtype=float),
            }
            if (
                guess_number % 10 == 0
                or guess_number == NUM_RANDOM_INITIAL_GUESSES
            ):
                print(
                    f"RAO {rao}: completed {guess_number}/"
                    f"{NUM_RANDOM_INITIAL_GUESSES}"
                )

        objective_values = np.asarray(
            [result["objective"] for result in rao_results.values()],
            dtype=float,
        )
        optimized_vectors = [
            result["p_b"] for result in rao_results.values()
        ]
        max_pairwise_p_b_difference = 0.0
        for first_index in range(len(optimized_vectors)):
            for second_index in range(
                first_index + 1,
                len(optimized_vectors),
            ):
                pairwise_difference = float(np.max(np.abs(
                    optimized_vectors[first_index]
                    - optimized_vectors[second_index]
                )))
                max_pairwise_p_b_difference = max(
                    max_pairwise_p_b_difference,
                    pairwise_difference,
                )

        rao_sensitivity = {
            "rao": rao,
            "minimum_objective": float(np.min(objective_values)),
            "maximum_objective": float(np.max(objective_values)),
            "objective_range": float(np.ptp(objective_values)),
            "max_pairwise_p_b_difference": max_pairwise_p_b_difference,
        }
        sensitivity_by_rao.append(rao_sensitivity)
        print(
            f"RAO {rao} sensitivity: "
            f"loss range={rao_sensitivity['objective_range']:.3e}, "
            f"max pairwise p_b difference="
            f"{max_pairwise_p_b_difference:.6f}"
        )

    print("\n--- Backoff Initial-Guess Sensitivity Complete ---")
    print(f"Arrival rate: {RHO:g} packets/s")
    print(f"Sampled RAOs: {sampled_raos}")
    print(f"Random initial guesses: {NUM_RANDOM_INITIAL_GUESSES}")
    print(
        f"{'RAO':>6} | {'Min loss':>12} | {'Max loss':>12} | "
        f"{'Loss range':>11} | {'Max Pb diff':>11}"
    )
    print("-" * 73)
    for item in sensitivity_by_rao:
        print(
            f"{item['rao']:6d} | "
            f"{item['minimum_objective']:12.10f} | "
            f"{item['maximum_objective']:12.10f} | "
            f"{item['objective_range']:11.3e} | "
            f"{item['max_pairwise_p_b_difference']:11.6f}"
        )
    objective_ranges = np.asarray(
        [item["objective_range"] for item in sensitivity_by_rao],
        dtype=float,
    )
    p_b_differences = np.asarray(
        [
            item["max_pairwise_p_b_difference"]
            for item in sensitivity_by_rao
        ],
        dtype=float,
    )
    print(
        f"Overall loss range: mean={np.mean(objective_ranges):.3e}, "
        f"max={np.max(objective_ranges):.3e}"
    )
    print(
        f"Overall max pairwise p_b difference: "
        f"mean={np.mean(p_b_differences):.6f}, "
        f"max={np.max(p_b_differences):.6f}"
    )
    raise SystemExit


if RUN_ORBIT_PLANE_COMPARISON:
    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    RHO = 1.5
    SERVICE_RADIUS_KM = 200.0
    MODES = [
        ([6, 1], "DCLARA"),
        ([5, 3], "ALLA with SAACB"),
    ]
    ORBIT_PLANE_SCENARIOS = [
        (
            1,
            "fixed_satellite_pool_planes_1.json",
            "group_ps_table_planes_1.npz",
        ),
        (
            2,
            "fixed_satellite_pool_planes_2.json",
            "group_ps_table_planes_2.npz",
        ),
        (3, "fixed_satellite_pool.json", "group_ps_table.npz"),
        (
            4,
            "fixed_satellite_pool_planes_4.json",
            "group_ps_table_planes_4.npz",
        ),
    ]

    orbit_plane_results = {label: [] for _, label in MODES}
    for mode, label in MODES:
        for (
            orbit_plane_count,
            satellite_pool_filename,
            group_table_filename,
        ) in ORBIT_PLANE_SCENARIOS:
            print(
                f"\nRunning orbit-plane comparison: "
                f"planes={orbit_plane_count}, method={label}, "
                f"arrival rate={RHO:g}"
            )
            (
                avg_throughput,
                plr,
                n_history,
                _,
                _,
                _,
                run_history,
            ) = main.main(
                RHO,
                SECONDS,
                NUM_UE,
                mode,
                SEED,
                IMBALANCE_EPSILON,
                USE_REAL_PS=USE_REAL_PS,
                SERVICE_RADIUS_KM=SERVICE_RADIUS_KM,
                GROUP_TABLE_FILENAME=group_table_filename,
                SATELLITE_POOL_FILENAME=satellite_pool_filename,
            )
            final_n_estimate = (
                n_history[-1] if len(n_history) > 0 else np.nan
            )
            orbit_plane_results[label].append({
                "orbit_plane_count": int(orbit_plane_count),
                "plr": float(plr),
                "throughput": float(avg_throughput),
                "average_deadline_budget_utilization": float(
                    run_history.get(
                        "average_deadline_budget_utilization",
                        np.nan,
                    )
                ),
                "final_n_estimate": float(final_n_estimate),
            })

    def plot_orbit_plane_metric(metric, ylabel, title, scale=1.0):
        figure, axis = plt.subplots(figsize=(10, 6), dpi=120)
        for _, label in MODES:
            results = orbit_plane_results[label]
            plane_axis = np.array([
                item["orbit_plane_count"] for item in results
            ])
            values = np.array([item[metric] for item in results]) * scale
            axis.plot(
                plane_axis,
                values,
                marker="o",
                linewidth=1.6,
                label=label,
            )
        axis.set(
            title=title,
            xlabel="Number of orbital planes",
            ylabel=ylabel,
            xticks=[
                plane_count
                for plane_count, _, _ in ORBIT_PLANE_SCENARIOS
            ],
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        plt.show()

    plot_orbit_plane_metric(
        "plr",
        "Packet loss rate",
        "PLR Comparison under Different Numbers of Orbital Planes",
    )
    plot_orbit_plane_metric(
        "throughput",
        "Average throughput (packets/second)",
        "Throughput Comparison under Different Numbers of Orbital Planes",
    )
    plot_orbit_plane_metric(
        "average_deadline_budget_utilization",
        "Average deadline budget utilized (%)",
        "Deadline Budget Utilization under Different Numbers of Orbital Planes",
        scale=100.0,
    )

    print("\n--- Orbit Plane Count Comparison Complete ---")
    for _, label in MODES:
        for item in orbit_plane_results[label]:
            print(
                f"planes={item['orbit_plane_count']}, {label}, "
                f"arrival rate={RHO:.4f}: "
                f"PLR={item['plr']:.4f}, "
                f"throughput={item['throughput']:.2f}, "
                f"deadline_budget_utilization="
                f"{item['average_deadline_budget_utilization'] * 100:.2f}%"
                + (
                    f", final_N={item['final_n_estimate']:.2f}"
                    if np.isfinite(item["final_n_estimate"])
                    else ""
                )
            )
    raise SystemExit


if RUN_TOP_K_GROUPING_ANALYSIS:
    import csv
    import warnings
    from datetime import timedelta
    from pathlib import Path

    from skyfield.api import load
    from skyfield.framelib import itrs

    from satellite_preselection import generate_uniform_locations
    from satellite_preselection_top3 import prepare_ue_geometry
    from scenario_time import get_tle_scenario_metadata
    from selection import solve_group_selection_policy

    TABLE_FILE = "group_ps_table_top3.npz"
    OUTPUT_CSV = "topk_ss_channel_success_all_rao.csv"
    OUTPUT_FIGURE = "topk_ss_channel_success_all_rao.png"
    OUTPUT_PDF = Path("output/pdf/topk_ss_channel_success_all_rao.pdf")
    TOP_K_SEED = 42
    TOP_K_EPSILON = 0.001
    TOP_K_POLICIES = (
        ("Global", 0),
        ("Top-1", 1),
        ("Top-2", 2),
        ("Top-3", 3),
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in reduce",
        category=RuntimeWarning,
    )

    def get_top_k_ue_channel_data(real_sats, current_time, ue_geometry):
        ecef, east, north, up = ue_geometry
        sat_ecef = np.stack([
            sat.at(current_time).frame_xyz(itrs).km
            for sat in real_sats
        ])
        delta = sat_ecef[None, :, :] - ecef[:, None, :]
        up_component = np.einsum("nkd,nd->nk", delta, up)
        east_component = np.einsum("nkd,nd->nk", delta, east)
        north_component = np.einsum("nkd,nd->nk", delta, north)
        angles = np.degrees(np.arctan2(
            up_component,
            np.hypot(east_component, north_component),
        ))
        distances = np.linalg.norm(delta, axis=2)
        channel_ps = main.estimate_channel_success_probability(
            angles,
            distances,
        )
        ranking = np.argsort(angles, axis=1)[:, ::-1]
        return channel_ps, ranking

    def merge_top_k_groups(weights_max_k, ps_max_k, prefix_length):
        weights = {}
        weighted_ps = {}
        for max_k_group, weight in weights_max_k.items():
            group = (
                tuple(max_k_group[:prefix_length])
                if prefix_length
                else ()
            )
            weights[group] = weights.get(group, 0.0) + weight
            weighted_ps.setdefault(
                group,
                np.zeros_like(ps_max_k[max_k_group]),
            )
            weighted_ps[group] += weight * ps_max_k[max_k_group]
        ps_by_group = {
            group: weighted_ps[group] / weights[group]
            for group in weights
        }
        return weights, ps_by_group

    def evaluate_top_k_policy(
        policy,
        prefix_length,
        weights,
        group_ps,
        channel_ps,
        ranking,
        selection_uniforms,
        channel_uniforms,
    ):
        num_ues, num_sats = channel_ps.shape
        selection_ps = np.zeros((num_ues, num_sats))
        fallback_count = 0

        for ue_index in range(num_ues):
            group = (
                tuple(ranking[ue_index, :prefix_length])
                if prefix_length
                else ()
            )
            if group in policy:
                selection_ps[ue_index] = policy[group]
            else:
                selection_ps[ue_index, ranking[ue_index, 0]] = 1.0
                fallback_count += 1

        effective_load = sum(
            weights[group] * policy[group] * group_ps[group]
            for group in weights
        )
        predicted_ps = float(np.sum(effective_load))
        imbalance = float(np.sum(
            (effective_load - predicted_ps / num_sats) ** 2
        ))
        per_ue_expected_ps = float(np.mean(np.sum(
            selection_ps * channel_ps,
            axis=1,
        )))

        selected_sats = np.sum(
            selection_uniforms[:, None] > np.cumsum(selection_ps, axis=1),
            axis=1,
        )
        selected_sats = np.minimum(selected_sats, num_sats - 1)
        ue_indices = np.arange(num_ues)
        channel_passed = (
            channel_uniforms[ue_indices, selected_sats]
            < channel_ps[ue_indices, selected_sats]
        )
        return {
            "expected_ps": predicted_ps,
            "per_ue_expected_ps": per_ue_expected_ps,
            "simulated_channel_success_rate": float(np.mean(channel_passed)),
            "imbalance": imbalance,
            "fallbacks": fallback_count,
        }

    def save_top_k_tradeoff_figure(results):
        names = [name for name, _ in TOP_K_POLICIES]
        mean_ps = np.array([
            np.mean([
                float(row["expected_ps"])
                for row in results
                if row["policy"] == name
            ])
            for name in names
        ])
        mean_groups = np.array([
            np.mean([
                float(row["groups"])
                for row in results
                if row["policy"] == name
            ])
            for name in names
        ])

        x = np.arange(len(names))
        width = 0.36
        figure, success_axis = plt.subplots(figsize=(9.5, 5.5), dpi=140)
        group_axis = success_axis.twinx()

        success_bars = success_axis.bar(
            x - width / 2,
            mean_ps,
            width,
            color="#4C78A8",
            alpha=0.9,
            label="Preamble transmission success probability",
        )
        success_axis.bar_label(success_bars, fmt="%.4f", padding=3)
        success_axis.set_ylabel(
            "Preamble transmission success probability"
        )
        success_axis.set_ylim(0, 0.75)

        group_bars = group_axis.bar(
            x + width / 2,
            mean_groups,
            width,
            color="#6B7280",
            alpha=0.9,
            label="Average number of groups",
        )
        group_axis.bar_label(group_bars, fmt="%.1f", padding=3)
        group_axis.set_ylabel("Average number of groups")
        group_axis.set_ylim(0, 40)

        success_axis.set(
            title=(
                "Preamble Transmission Success Probability under "
                "Different Grouping Policies"
            ),
            xlabel="Grouping policy",
            xticks=x,
            xticklabels=names,
        )
        success_axis.grid(axis="y", alpha=0.25)
        success_axis.set_axisbelow(True)
        success_axis.legend(
            [success_bars, group_bars],
            [
                "Preamble transmission success probability",
                "Average number of groups",
            ],
            loc="upper left",
        )
        figure.tight_layout()
        figure.savefig(OUTPUT_FIGURE, bbox_inches="tight")
        OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(OUTPUT_PDF, bbox_inches="tight")
        plt.close(figure)

    with np.load(TABLE_FILE, allow_pickle=True) as top_k_data:
        weight_table = top_k_data["group_weight_table"]
        ps_table = top_k_data["group_ps_table"]
        rao_indices = np.asarray(top_k_data["rao_indices"], dtype=int)
        table_seconds = int(top_k_data["seconds"])
        trao_ms = int(top_k_data["trao_ms"])
        num_ues = int(top_k_data["num_points"])
        table_seed = int(top_k_data["random_seed"])
        center = (
            float(top_k_data["center_lat"]),
            float(top_k_data["center_lon"]),
        )
        radius_km = float(top_k_data["radius_km"])
        table_satellite_ids = np.asarray(
            top_k_data["sat_norad_ids"],
            dtype=int,
        )
        table_start_dt_iso = str(top_k_data["scenario_start_dt_iso"])
        table_tle_hash = str(top_k_data["tle_file_sha256"])

    real_sats = main.load_fixed_satellites()
    actual_satellite_ids = np.array([
        int(sat.model.satnum) for sat in real_sats
    ])
    if not np.array_equal(actual_satellite_ids, table_satellite_ids):
        raise ValueError("Satellite pool does not match the Top-3 table.")

    np.random.seed(table_seed)
    locations = generate_uniform_locations(num_ues, center, radius_km)
    ue_geometry = prepare_ue_geometry(locations)
    scenario = get_tle_scenario_metadata()
    if table_start_dt_iso != scenario["start_dt_iso"]:
        raise ValueError(
            "Top-3 table and current scenario start time do not match."
        )
    if table_tle_hash != scenario["tle_file_sha256"]:
        raise ValueError("Top-3 table and current TLE file do not match.")
    if SIM_SECONDS > table_seconds:
        raise ValueError(
            f"Mode 18 requests {SIM_SECONDS} seconds, but {TABLE_FILE} "
            f"only covers {table_seconds} seconds."
        )
    requested_end_rao = SIM_SECONDS * 1000 // trao_ms
    rows = np.flatnonzero(rao_indices < requested_end_rao)
    if len(rows) == 0:
        raise ValueError(
            "Mode 18 requires SIM_SECONDS to include at least one "
            "sampled RAO."
        )
    timescale = load.timescale()
    random_generator = np.random.default_rng(TOP_K_SEED)
    top_k_results = []

    for row in rows:
        actual_rao = int(rao_indices[row])
        current_dt = scenario["start_dt"] + timedelta(
            milliseconds=actual_rao * trao_ms
        )
        current_time = timescale.from_datetime(current_dt)
        channel_ps, ranking = get_top_k_ue_channel_data(
            real_sats,
            current_time,
            ue_geometry,
        )
        selection_uniforms = random_generator.random(num_ues)
        channel_uniforms = random_generator.random(channel_ps.shape)

        for name, prefix_length in TOP_K_POLICIES:
            weights, group_ps = merge_top_k_groups(
                weight_table[row],
                ps_table[row],
                prefix_length,
            )
            policy = solve_group_selection_policy(
                weights,
                group_ps,
                sat_num=len(real_sats),
                imbalance_epsilon=TOP_K_EPSILON,
                initial_policy=None,
            )
            metrics = evaluate_top_k_policy(
                policy,
                prefix_length,
                weights,
                group_ps,
                channel_ps,
                ranking,
                selection_uniforms,
                channel_uniforms,
            )
            top_k_results.append({
                "rao": actual_rao,
                "policy": name,
                "groups": len(weights),
                **metrics,
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=top_k_results[0])
        writer.writeheader()
        writer.writerows(top_k_results)

    policy_names = [name for name, _ in TOP_K_POLICIES]
    predicted_values_by_policy = [
        np.array([
            result["expected_ps"]
            for result in top_k_results
            if result["policy"] == name
        ])
        for name in policy_names
    ]
    simulated_values_by_policy = [
        np.array([
            result["simulated_channel_success_rate"]
            for result in top_k_results
            if result["policy"] == name
        ])
        for name in policy_names
    ]
    save_top_k_tradeoff_figure(top_k_results)

    print(
        f"Analyzed {len(rows)} sampled RAOs over the first "
        f"{SIM_SECONDS} seconds: {int(rao_indices[rows[0]])} to "
        f"{int(rao_indices[rows[-1]])}"
    )
    for name, predicted_values, simulated_values in zip(
        policy_names,
        predicted_values_by_policy,
        simulated_values_by_policy,
    ):
        policy_rows = [
            result
            for result in top_k_results
            if result["policy"] == name
        ]
        print(
            f"{name}: expected p_s={predicted_values.mean():.6f}, "
            f"per-UE simulation={simulated_values.mean():.6f}, "
            f"mean groups="
            f"{np.mean([result['groups'] for result in policy_rows]):.1f}, "
            f"max imbalance="
            f"{max(result['imbalance'] for result in policy_rows):.6g}, "
            f"fallbacks="
            f"{sum(result['fallbacks'] for result in policy_rows)}"
        )
    raise SystemExit


if RUN_TOP_K_ORBIT_PLANE_COMPARISON:
    import csv
    import warnings
    from pathlib import Path

    from matplotlib.lines import Line2D

    from scenario_time import get_tle_scenario_metadata
    from selection import solve_group_selection_policy

    TOP_K_ORBIT_SCENARIOS = (
        (
            1,
            Path("group_ps_table_planes_1_top3.npz"),
            Path("group_ps_table_planes_1.npz"),
        ),
        (
            2,
            Path("group_ps_table_planes_2_top3.npz"),
            Path("group_ps_table_planes_2.npz"),
        ),
        (
            3,
            Path("group_ps_table_planes_3_nested_top3.npz"),
            None,
        ),
        (
            4,
            Path("group_ps_table_planes_4_nested_top3.npz"),
            None,
        ),
    )
    TOP_K_ORBIT_POLICIES = (
        ("Global", 0, "o"),
        ("Top-1", 1, "s"),
        ("Top-2", 2, "^"),
        ("Top-3", 3, "D"),
    )
    TOP_K_ORBIT_EPSILON = 0.001
    TOP_K_ORBIT_OUTPUT_CSV = Path("topk_orbit_plane_comparison.csv")
    TOP_K_ORBIT_SUMMARY_CSV = Path(
        "topk_orbit_plane_comparison_summary.csv"
    )
    TOP_K_ORBIT_OUTPUT_PNG = Path("topk_orbit_plane_tradeoff.png")
    TOP_K_ORBIT_OUTPUT_PDF = Path(
        "output/pdf/topk_orbit_plane_tradeoff.pdf"
    )
    TOP_K_ORBIT_PANEL_PNG = Path("topk_orbit_plane_comparison.png")
    TOP_K_ORBIT_PANEL_PDF = Path(
        "output/pdf/topk_orbit_plane_comparison.pdf"
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in reduce",
        category=RuntimeWarning,
    )

    def merge_mode19_groups(weights_top3, ps_top3, prefix_length):
        weights = {}
        weighted_ps = {}
        for top3_group, weight in weights_top3.items():
            top3_group = tuple(top3_group)
            if len(top3_group) != 3:
                raise ValueError(
                    "Mode 19 requires ordered Top-3 group keys; "
                    f"received {top3_group}."
                )
            group = top3_group[:prefix_length] if prefix_length else ()
            weights[group] = weights.get(group, 0.0) + float(weight)
            weighted_ps.setdefault(
                group,
                np.zeros_like(ps_top3[top3_group], dtype=float),
            )
            weighted_ps[group] += (
                float(weight)
                * np.asarray(ps_top3[top3_group], dtype=float)
            )
        ps_by_group = {
            group: weighted_ps[group] / weights[group]
            for group in weights
        }
        return weights, ps_by_group

    def load_mode19_scenario(
        orbit_plane_count,
        top3_table_path,
        reference_table_path,
        current_scenario_metadata,
    ):
        if not top3_table_path.exists():
            raise FileNotFoundError(
                f"Mode 19 Top-3 table not found: {top3_table_path}"
            )
        if (
            reference_table_path is not None
            and not reference_table_path.exists()
        ):
            raise FileNotFoundError(
                f"Mode 19 reference table not found: {reference_table_path}"
            )

        required_top3_keys = (
            "group_weight_table",
            "group_ps_table",
            "sat_norad_ids",
            "scenario_start_dt_iso",
            "tle_file_sha256",
            "seconds",
            "trao_ms",
            "num_points",
            "rao_indices",
            "rao_step",
            "group_size",
            "random_seed",
            "center_lat",
            "center_lon",
            "radius_km",
        )
        with np.load(top3_table_path, allow_pickle=True) as top3_data:
            missing_keys = [
                key for key in required_top3_keys
                if key not in top3_data.files
            ]
            if missing_keys:
                raise ValueError(
                    f"{top3_table_path} is missing required fields: "
                    f"{missing_keys}"
                )
            scenario_data = {
                "orbit_plane_count": int(orbit_plane_count),
                "top3_table_path": top3_table_path,
                "weight_table": top3_data["group_weight_table"],
                "ps_table": top3_data["group_ps_table"],
                "satellite_ids": np.asarray(
                    top3_data["sat_norad_ids"],
                    dtype=int,
                ),
                "scenario_start_dt_iso": str(
                    top3_data["scenario_start_dt_iso"]
                ),
                "tle_file_sha256": str(top3_data["tle_file_sha256"]),
                "seconds": int(top3_data["seconds"]),
                "trao_ms": int(top3_data["trao_ms"]),
                "num_points": int(top3_data["num_points"]),
                "rao_indices": np.asarray(
                    top3_data["rao_indices"],
                    dtype=int,
                ),
                "rao_step": int(top3_data["rao_step"]),
                "group_size": int(top3_data["group_size"]),
                "random_seed": int(top3_data["random_seed"]),
                "center_lat": float(top3_data["center_lat"]),
                "center_lon": float(top3_data["center_lon"]),
                "radius_km": float(top3_data["radius_km"]),
            }
            if "orbit_plane_count" in top3_data.files:
                stored_plane_count = int(top3_data["orbit_plane_count"])
                if stored_plane_count != orbit_plane_count:
                    raise ValueError(
                        f"{top3_table_path} declares {stored_plane_count} "
                        f"orbital planes; expected {orbit_plane_count}."
                    )

        if scenario_data["group_size"] != 3:
            raise ValueError(
                f"{top3_table_path} group_size is "
                f"{scenario_data['group_size']}; expected 3."
            )
        if scenario_data["scenario_start_dt_iso"] != (
            current_scenario_metadata["start_dt_iso"]
        ):
            raise ValueError(
                f"{top3_table_path} and current scenario start time "
                "do not match."
            )
        if scenario_data["tle_file_sha256"] != (
            current_scenario_metadata["tle_file_sha256"]
        ):
            raise ValueError(
                f"{top3_table_path} and current TLE file do not match."
            )

        if reference_table_path is not None:
            with np.load(reference_table_path, allow_pickle=True) as reference:
                required_reference_keys = (
                    "sat_norad_ids",
                    "scenario_start_dt_iso",
                    "tle_file_sha256",
                    "seconds",
                    "trao_ms",
                    "num_points",
                )
                missing_reference_keys = [
                    key for key in required_reference_keys
                    if key not in reference.files
                ]
                if missing_reference_keys:
                    raise ValueError(
                        f"{reference_table_path} is missing required fields: "
                        f"{missing_reference_keys}"
                    )
                reference_satellite_ids = np.asarray(
                    reference["sat_norad_ids"],
                    dtype=int,
                )
                if not np.array_equal(
                    scenario_data["satellite_ids"],
                    reference_satellite_ids,
                ):
                    raise ValueError(
                        f"{top3_table_path} satellite IDs do not match "
                        f"{reference_table_path}."
                    )
                if str(reference["scenario_start_dt_iso"]) != (
                    scenario_data["scenario_start_dt_iso"]
                ):
                    raise ValueError(
                        f"{top3_table_path} and {reference_table_path} "
                        "start times do not match."
                    )
                if str(reference["tle_file_sha256"]) != (
                    scenario_data["tle_file_sha256"]
                ):
                    raise ValueError(
                        f"{top3_table_path} and {reference_table_path} "
                        "TLE hashes do not match."
                    )
                for metadata_key in (
                    "seconds",
                    "trao_ms",
                    "num_points",
                ):
                    if int(reference[metadata_key]) != (
                        scenario_data[metadata_key]
                    ):
                        raise ValueError(
                            f"{top3_table_path} and "
                            f"{reference_table_path} {metadata_key} "
                            "values do not match."
                        )
                reference_radius = (
                    float(reference["radius_km"])
                    if "radius_km" in reference.files
                    else 200.0
                )
                if not np.isclose(
                    reference_radius,
                    scenario_data["radius_km"],
                ):
                    raise ValueError(
                        f"{top3_table_path} and {reference_table_path} "
                        "service radii do not match."
                    )

        if len(scenario_data["weight_table"]) != len(
            scenario_data["rao_indices"]
        ) or len(scenario_data["ps_table"]) != len(
            scenario_data["rao_indices"]
        ):
            raise ValueError(
                f"{top3_table_path} table rows and RAO indices do not match."
            )
        return scenario_data

    current_scenario_metadata = get_tle_scenario_metadata()
    mode19_scenarios = [
        load_mode19_scenario(
            orbit_plane_count,
            top3_table_path,
            reference_table_path,
            current_scenario_metadata,
        )
        for (
            orbit_plane_count,
            top3_table_path,
            reference_table_path,
        ) in TOP_K_ORBIT_SCENARIOS
    ]

    baseline_scenario = mode19_scenarios[0]
    for scenario_data in mode19_scenarios[1:]:
        for metadata_key in (
            "scenario_start_dt_iso",
            "tle_file_sha256",
            "seconds",
            "trao_ms",
            "num_points",
            "rao_step",
            "random_seed",
        ):
            if scenario_data[metadata_key] != baseline_scenario[metadata_key]:
                raise ValueError(
                    "Mode 19 Top-3 tables use different "
                    f"{metadata_key} values."
                )
        for metadata_key in ("center_lat", "center_lon", "radius_km"):
            if not np.isclose(
                scenario_data[metadata_key],
                baseline_scenario[metadata_key],
            ):
                raise ValueError(
                    "Mode 19 Top-3 tables use different "
                    f"{metadata_key} values."
                )
        if not np.array_equal(
            scenario_data["rao_indices"],
            baseline_scenario["rao_indices"],
        ):
            raise ValueError(
                "Mode 19 Top-3 tables do not use identical sampled RAOs."
            )

    mode19_results = []
    for scenario_data in mode19_scenarios:
        orbit_plane_count = scenario_data["orbit_plane_count"]
        weight_table = scenario_data["weight_table"]
        ps_table = scenario_data["ps_table"]
        rao_indices = scenario_data["rao_indices"]
        satellite_count = len(scenario_data["satellite_ids"])

        print(
            f"\nRunning Mode 19 grouping analysis: "
            f"{orbit_plane_count} orbital plane(s), "
            f"{len(rao_indices)} sampled RAOs"
        )
        for row_index, actual_rao in enumerate(rao_indices):
            for policy_name, prefix_length, _ in TOP_K_ORBIT_POLICIES:
                weights, group_ps = merge_mode19_groups(
                    weight_table[row_index],
                    ps_table[row_index],
                    prefix_length,
                )
                policy = solve_group_selection_policy(
                    weights,
                    group_ps,
                    sat_num=satellite_count,
                    imbalance_epsilon=TOP_K_ORBIT_EPSILON,
                    initial_policy=None,
                )
                effective_load = sum(
                    weights[group]
                    * np.asarray(policy[group], dtype=float)
                    * np.asarray(group_ps[group], dtype=float)
                    for group in weights
                )
                expected_ps = float(np.sum(effective_load))
                imbalance = float(np.sum(
                    (
                        effective_load
                        - expected_ps / satellite_count
                    ) ** 2
                ))
                mode19_results.append({
                    "orbit_plane_count": orbit_plane_count,
                    "rao": int(actual_rao),
                    "policy": policy_name,
                    "expected_ps": expected_ps,
                    "group_count": len(weights),
                    "imbalance": imbalance,
                })

            completed_rows = row_index + 1
            if completed_rows % 20 == 0 or completed_rows == len(rao_indices):
                print(
                    f"{orbit_plane_count}-plane: completed "
                    f"{completed_rows}/{len(rao_indices)} sampled RAOs"
                )

    with TOP_K_ORBIT_OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=mode19_results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(mode19_results)

    mode19_summary = []
    for scenario_data in mode19_scenarios:
        orbit_plane_count = scenario_data["orbit_plane_count"]
        for policy_name, _, _ in TOP_K_ORBIT_POLICIES:
            matching_results = [
                result
                for result in mode19_results
                if result["orbit_plane_count"] == orbit_plane_count
                and result["policy"] == policy_name
            ]
            mode19_summary.append({
                "orbit_plane_count": orbit_plane_count,
                "policy": policy_name,
                "mean_expected_ps": float(np.mean([
                    result["expected_ps"] for result in matching_results
                ])),
                "average_group_count": float(np.mean([
                    result["group_count"] for result in matching_results
                ])),
                "maximum_imbalance": float(np.max([
                    result["imbalance"] for result in matching_results
                ])),
            })

    with TOP_K_ORBIT_SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=mode19_summary[0].keys(),
        )
        writer.writeheader()
        writer.writerows(mode19_summary)

    plane_styles = {
        1: ("#4C78A8", "-"),
        2: ("#F58518", "--"),
        3: ("#54A24B", "-."),
        4: ("#E45756", ":"),
    }
    figure, axis = plt.subplots(figsize=(10, 6), dpi=140)
    plane_legend_handles = []

    for scenario_data in mode19_scenarios:
        orbit_plane_count = scenario_data["orbit_plane_count"]
        color, line_style = plane_styles[orbit_plane_count]
        plane_summary = [
            next(
                item
                for item in mode19_summary
                if item["orbit_plane_count"] == orbit_plane_count
                and item["policy"] == policy_name
            )
            for policy_name, _, _ in TOP_K_ORBIT_POLICIES
        ]
        average_group_counts = np.array([
            item["average_group_count"] for item in plane_summary
        ])
        mean_expected_ps = np.array([
            item["mean_expected_ps"] for item in plane_summary
        ])
        axis.plot(
            average_group_counts,
            mean_expected_ps,
            color=color,
            linestyle=line_style,
            linewidth=1.7,
            zorder=2,
        )
        for point_index, (_, _, marker) in enumerate(TOP_K_ORBIT_POLICIES):
            axis.scatter(
                average_group_counts[point_index],
                mean_expected_ps[point_index],
                color=color,
                marker=marker,
                s=58,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
        plane_label = (
            "1 orbital plane"
            if orbit_plane_count == 1
            else f"{orbit_plane_count} orbital planes"
        )
        plane_legend_handles.append(Line2D(
            [0],
            [0],
            color=color,
            linestyle=line_style,
            linewidth=1.7,
            label=plane_label,
        ))

    policy_legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=marker,
            linestyle="None",
            markersize=6.5,
            label=policy_name,
        )
        for policy_name, _, marker in TOP_K_ORBIT_POLICIES
    ]
    plane_legend = axis.legend(
        handles=plane_legend_handles,
        title="Number of orbital planes",
        loc="lower right",
    )
    axis.add_artist(plane_legend)
    axis.legend(
        handles=policy_legend_handles,
        title="Grouping policy",
        loc="upper left",
    )
    axis.set(
        title=(
            "Grouping Performance-Complexity Tradeoff under Different "
            "Numbers of Orbital Planes"
        ),
        xlabel="Average number of groups",
        ylabel="Preamble transmission success probability",
    )
    axis.grid(True, alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(TOP_K_ORBIT_OUTPUT_PNG, bbox_inches="tight")
    TOP_K_ORBIT_OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(TOP_K_ORBIT_OUTPUT_PDF, bbox_inches="tight")

    policy_axis = np.arange(len(TOP_K_ORBIT_POLICIES))
    policy_labels = [
        policy_name
        for policy_name, _, _ in TOP_K_ORBIT_POLICIES
    ]
    panel_figure, (success_axis, group_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        dpi=140,
        sharex=True,
    )
    panel_legend_handles = []

    for scenario_data in mode19_scenarios:
        orbit_plane_count = scenario_data["orbit_plane_count"]
        color, _ = plane_styles[orbit_plane_count]
        plane_summary = [
            next(
                item
                for item in mode19_summary
                if item["orbit_plane_count"] == orbit_plane_count
                and item["policy"] == policy_name
            )
            for policy_name, _, _ in TOP_K_ORBIT_POLICIES
        ]
        mean_expected_ps = np.array([
            item["mean_expected_ps"] for item in plane_summary
        ])
        average_group_counts = np.array([
            item["average_group_count"] for item in plane_summary
        ])
        plane_label = (
            "1 orbital plane"
            if orbit_plane_count == 1
            else f"{orbit_plane_count} orbital planes"
        )

        success_axis.plot(
            policy_axis,
            mean_expected_ps,
            color=color,
            linestyle="-",
            marker="o",
            markersize=6,
            linewidth=1.6,
            label=plane_label,
        )
        group_axis.plot(
            policy_axis,
            average_group_counts,
            color=color,
            linestyle="-",
            marker="o",
            markersize=6,
            linewidth=1.6,
            label=plane_label,
        )
        panel_legend_handles.append(Line2D(
            [0],
            [0],
            color=color,
            linestyle="-",
            marker="o",
            markersize=6,
            linewidth=1.6,
            label=plane_label,
        ))

    success_axis.set(
        ylabel="Preamble transmission success probability",
    )
    group_axis.set(
        xlabel="Grouping policy",
        ylabel="Average number of groups",
        xticks=policy_axis,
        xticklabels=policy_labels,
    )
    for panel_axis in (success_axis, group_axis):
        panel_axis.grid(True, alpha=0.25)
        panel_axis.set_axisbelow(True)

    panel_figure.suptitle(
        "Effect of Grouping Granularity under Different Numbers of "
        "Orbital Planes",
        y=0.995,
    )
    panel_figure.legend(
        handles=panel_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=True,
    )
    panel_figure.tight_layout(rect=(0, 0, 1, 0.91))
    panel_figure.savefig(TOP_K_ORBIT_PANEL_PNG, bbox_inches="tight")
    TOP_K_ORBIT_PANEL_PDF.parent.mkdir(parents=True, exist_ok=True)
    panel_figure.savefig(TOP_K_ORBIT_PANEL_PDF, bbox_inches="tight")

    print("\n--- Top-k Orbit-Plane Comparison Complete ---")
    print(
        f"{'Planes':>6} | {'Policy':>7} | {'Mean p_s':>10} | "
        f"{'Avg groups':>10} | {'Max imbalance':>13}"
    )
    print("-" * 61)
    for item in mode19_summary:
        print(
            f"{item['orbit_plane_count']:6d} | "
            f"{item['policy']:>7} | "
            f"{item['mean_expected_ps']:10.6f} | "
            f"{item['average_group_count']:10.2f} | "
            f"{item['maximum_imbalance']:13.6g}"
        )

    print("\nMarginal changes:")
    for scenario_data in mode19_scenarios:
        orbit_plane_count = scenario_data["orbit_plane_count"]
        summary_by_policy = {
            item["policy"]: item
            for item in mode19_summary
            if item["orbit_plane_count"] == orbit_plane_count
        }
        top1 = summary_by_policy["Top-1"]
        top2 = summary_by_policy["Top-2"]
        top3 = summary_by_policy["Top-3"]
        print(
            f"{orbit_plane_count}-plane: "
            f"Top-1->Top-2 delta_p_s="
            f"{top2['mean_expected_ps'] - top1['mean_expected_ps']:.6f}, "
            f"delta_groups="
            f"{top2['average_group_count'] - top1['average_group_count']:.2f}; "
            f"Top-2->Top-3 delta_p_s="
            f"{top3['mean_expected_ps'] - top2['mean_expected_ps']:.6f}, "
            f"delta_groups="
            f"{top3['average_group_count'] - top2['average_group_count']:.2f}"
        )

    print(f"Saved RAO results to {TOP_K_ORBIT_OUTPUT_CSV}")
    print(f"Saved summary to {TOP_K_ORBIT_SUMMARY_CSV}")
    print(f"Saved tradeoff figure to {TOP_K_ORBIT_OUTPUT_PDF}")
    print(f"Saved two-panel comparison figure to {TOP_K_ORBIT_PANEL_PDF}")
    plt.show()
    raise SystemExit


if RUN_SATELLITE_SELECTION_TOP5_OVER_TIME:
    NUM_UE = 10000
    SECONDS = 180
    SEED = 42
    RHO = 1.5
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    MODE = [6, 3]
    WINDOW_RAOS = 300
    WINDOW_COUNT = 6
    TOP_SATELLITE_COUNT = 5
    EXPECTED_RAOS = WINDOW_RAOS * WINDOW_COUNT

    print(
        "\nRunning Mode 20 satellite-selection analysis: "
        f"{SECONDS} seconds, {EXPECTED_RAOS} RAOs, "
        f"{WINDOW_COUNT} windows of {WINDOW_RAOS} RAOs"
    )
    _, _, _, _, _, _, run_history = main.main(
        RHO,
        SECONDS,
        NUM_UE,
        MODE,
        SEED,
        IMBALANCE_EPSILON,
        USE_REAL_PS=USE_REAL_PS,
    )
    selection_history = run_history.get(
        "ue_satellite_selection_history",
        [],
    )
    if len(selection_history) != EXPECTED_RAOS:
        raise ValueError(
            "Mode 20 requires exactly "
            f"{EXPECTED_RAOS} satellite-selection records; received "
            f"{len(selection_history)}."
        )

    rao_indices = np.array(
        [item["time_slot"] for item in selection_history],
        dtype=int,
    )
    if not np.array_equal(rao_indices, np.arange(EXPECTED_RAOS)):
        raise ValueError(
            "Mode 20 requires consecutive RAO indices from 0 to "
            f"{EXPECTED_RAOS - 1}."
        )

    selection_counts_by_rao = np.vstack([
        np.asarray(item["selection_counts"], dtype=np.int64)
        for item in selection_history
    ])
    if selection_counts_by_rao.ndim != 2:
        raise ValueError("Mode 20 selection counts must form a 2-D array.")
    if selection_counts_by_rao.shape[1] < TOP_SATELLITE_COUNT:
        raise ValueError(
            "Mode 20 requires at least five satellites in the scenario."
        )

    top_satellite_ids = np.empty(
        (WINDOW_COUNT, TOP_SATELLITE_COUNT),
        dtype=int,
    )
    top_satellite_shares = np.empty(
        (WINDOW_COUNT, TOP_SATELLITE_COUNT),
        dtype=float,
    )
    top_satellite_counts = np.empty(
        (WINDOW_COUNT, TOP_SATELLITE_COUNT),
        dtype=np.int64,
    )
    total_selections_by_window = np.empty(WINDOW_COUNT, dtype=np.int64)

    for window_index in range(WINDOW_COUNT):
        start_rao = window_index * WINDOW_RAOS
        stop_rao = start_rao + WINDOW_RAOS
        window_counts = np.sum(
            selection_counts_by_rao[start_rao:stop_rao],
            axis=0,
        )
        total_selections = int(np.sum(window_counts))
        if total_selections <= 0:
            raise ValueError(
                f"Mode 20 window {window_index + 1} has no UE satellite "
                "selections."
            )
        ranked_satellite_ids = np.argsort(
            -window_counts,
            kind="stable",
        )[:TOP_SATELLITE_COUNT]
        top_satellite_ids[window_index] = ranked_satellite_ids
        top_satellite_counts[window_index] = window_counts[
            ranked_satellite_ids
        ]
        top_satellite_shares[window_index] = (
            window_counts[ranked_satellite_ids] / total_selections
        )
        total_selections_by_window[window_index] = total_selections

    window_positions = np.arange(WINDOW_COUNT, dtype=float)
    bar_width = 0.15
    unique_top_satellite_ids = np.unique(top_satellite_ids)
    satellite_palette = plt.get_cmap(
        "tab20",
        len(unique_top_satellite_ids),
    )
    satellite_colors = {
        int(satellite_id): satellite_palette(color_index)
        for color_index, satellite_id in enumerate(unique_top_satellite_ids)
    }
    figure, axis = plt.subplots(figsize=(12, 6.5), dpi=140)
    maximum_share_percent = float(np.max(top_satellite_shares) * 100)

    for rank_index in range(TOP_SATELLITE_COUNT):
        bar_positions = (
            window_positions
            + (rank_index - (TOP_SATELLITE_COUNT - 1) / 2) * bar_width
        )
        share_percent = top_satellite_shares[:, rank_index] * 100
        bars = axis.bar(
            bar_positions,
            share_percent,
            width=bar_width,
            color=[
                satellite_colors[int(satellite_id)]
                for satellite_id in top_satellite_ids[:, rank_index]
            ],
            edgecolor="white",
            linewidth=0.5,
        )
        for window_index, bar in enumerate(bars):
            satellite_id = top_satellite_ids[
                window_index,
                rank_index,
            ] + 1
            axis.annotate(
                f"S{satellite_id}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    time_labels = [
        f"{window_index * WINDOW_RAOS / 10:.0f}–"
        f"{(window_index + 1) * WINDOW_RAOS / 10:.0f} s"
        for window_index in range(WINDOW_COUNT)
    ]
    axis.set(
        title="Top-5 Satellite Selection Shares over Time",
        xlabel=(
            "Simulation time interval "
            f"({WINDOW_RAOS} RAOs per interval)"
        ),
        ylabel="UE satellite-selection share (%)",
        xticks=window_positions,
        xticklabels=time_labels,
    )
    axis.set_ylim(0, max(10.0, maximum_share_percent * 1.24))
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    plt.show()

    print("\n--- Mode 20 Satellite Selection Top-5 Summary ---")
    for window_index in range(WINDOW_COUNT):
        top5_text = ", ".join(
            f"S{top_satellite_ids[window_index, rank_index] + 1}: "
            f"{top_satellite_shares[window_index, rank_index]:.2%}"
            for rank_index in range(TOP_SATELLITE_COUNT)
        )
        print(
            f"Window {window_index + 1} "
            f"({window_index * WINDOW_RAOS / 10:g}-"
            f"{(window_index + 1) * WINDOW_RAOS / 10:g} s): "
            f"{top5_text}"
        )
    raise SystemExit


if RUN_SATELLITE_POOL_REALIZATION_COMPARISON:
    import json
    from pathlib import Path

    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    RHO = 1.5
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    MODES = (
        ([6, 1], "DCLARA"),
        ([5, 3], "ALLA with SAACB"),
    )
    SERVICE_RADIUS_KM = 200.0
    POOL_SET_SCENARIOS = (
        (
            "original pool (baseline)",
            "Baseline",
            Path("fixed_satellite_pool.json"),
            Path("group_ps_table.npz"),
        ),
        (
            "alternative pool A",
            "Alternative A",
            Path("fixed_satellite_pool_planes_3_set_1.json"),
            Path("group_ps_table_planes_3_set_1.npz"),
        ),
        (
            "alternative pool B",
            "Alternative B",
            Path("fixed_satellite_pool_planes_3_set_2.json"),
            Path("group_ps_table_planes_3_set_2.npz"),
        ),
        (
            "alternative pool C",
            "Alternative C",
            Path("fixed_satellite_pool_planes_3_set_3.json"),
            Path("group_ps_table_planes_3_set_3.npz"),
        ),
    )

    missing_inputs = [
        str(path)
        for _, _, pool_path, table_path in POOL_SET_SCENARIOS
        for path in (pool_path, table_path)
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "Mode 21 requires the original pool plus all three new satellite "
            "pools and their group tables. "
            "Generate them with generate_nonoverlapping_three_plane_pools.py. "
            "Missing: "
            + ", ".join(missing_inputs)
        )

    scenario_metadata = []
    for label, chart_label, pool_path, table_path in POOL_SET_SCENARIOS:
        with pool_path.open("r", encoding="utf-8") as pool_file:
            satellite_count = len(json.load(pool_file))
        with np.load(table_path, allow_pickle=True) as table_data:
            table_seconds = int(table_data["seconds"])
            table_rao_ms = int(table_data["trao_ms"])
        if SECONDS > table_seconds:
            raise ValueError(
                f"Mode 21 requests {SECONDS} seconds, but {table_path} only "
                f"contains {table_seconds} seconds."
            )
        scenario_metadata.append({
            "label": label,
            "chart_label": chart_label,
            "pool_path": pool_path,
            "table_path": table_path,
            "satellite_count": satellite_count,
            "table_seconds": table_seconds,
            "table_rao_ms": table_rao_ms,
        })

    print("\n=== Mode 21: Satellite-Pool Realization Comparison ===")
    print(f"Simulation time: {SECONDS} seconds")
    print(f"UE count: {NUM_UE}")
    print(f"Arrival rate: {RHO:g} packets/s")
    print(
        "Methods: "
        + ", ".join(f"{label}={mode}" for mode, label in MODES)
    )
    print(f"Random seed: {SEED}")

    pool_set_results = []
    for scenario in scenario_metadata:
        for mode, method_label in MODES:
            print(
                f"\nRunning {scenario['label']}, method={method_label}: "
                f"satellites={scenario['satellite_count']}, "
                f"pool={scenario['pool_path']}, "
                f"table={scenario['table_path']}"
            )
            (
                average_throughput,
                packet_loss_rate,
                n_history,
                _,
                _,
                _,
                run_history,
            ) = main.main(
                RHO,
                SECONDS,
                NUM_UE,
                mode,
                SEED,
                IMBALANCE_EPSILON,
                USE_REAL_PS=USE_REAL_PS,
                SERVICE_RADIUS_KM=SERVICE_RADIUS_KM,
                GROUP_TABLE_FILENAME=str(scenario["table_path"]),
                SATELLITE_POOL_FILENAME=str(scenario["pool_path"]),
            )
            final_n_estimate = (
                float(n_history[-1]) if len(n_history) > 0 else np.nan
            )
            pool_set_results.append({
                "label": scenario["label"],
                "chart_label": scenario["chart_label"],
                "method": method_label,
                "satellite_count": scenario["satellite_count"],
                "throughput": float(average_throughput),
                "plr": float(packet_loss_rate),
                "average_delay_ms": float(
                    run_history.get("average_delay_ms", np.nan)
                ),
                "deadline_budget_utilization": float(
                    run_history.get(
                        "average_deadline_budget_utilization",
                        np.nan,
                    )
                ),
                "final_n_estimate": final_n_estimate,
            })

    print("\n--- Mode 21 Results ---")
    print(
        f"{'Pool set':<29} | {'Method':<12} | {'Sats':>4} | {'Throughput':>10} | "
        f"{'PLR':>8} | {'Delay ms':>9} | {'DB util.':>8} | {'Final N':>10}"
    )
    print("-" * 118)
    for result in pool_set_results:
        deadline_utilization = result["deadline_budget_utilization"]
        deadline_text = (
            f"{deadline_utilization * 100:.2f}%"
            if np.isfinite(deadline_utilization)
            else "N/A"
        )
        delay_text = (
            f"{result['average_delay_ms']:.2f}"
            if np.isfinite(result["average_delay_ms"])
            else "N/A"
        )
        final_n_text = (
            f"{result['final_n_estimate']:.2f}"
            if np.isfinite(result["final_n_estimate"])
            else "N/A"
        )
        print(
            f"{result['label']:<29} | "
            f"{result['method']:<12} | "
            f"{result['satellite_count']:4d} | "
            f"{result['throughput']:10.2f} | "
            f"{result['plr']:8.4f} | "
            f"{delay_text:>9} | "
            f"{deadline_text:>8} | "
            f"{final_n_text:>10}"
        )

    pool_axis = np.arange(len(scenario_metadata), dtype=float)
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(11, 6.5), dpi=140)
    method_colors = ("#4C78A8", "#F58518")
    for method_index, ((_, method_label), color) in enumerate(
        zip(MODES, method_colors)
    ):
        plr_percent = np.array([
            next(
                result["plr"]
                for result in pool_set_results
                if result["label"] == scenario["label"]
                and result["method"] == method_label
            ) * 100.0
            for scenario in scenario_metadata
        ])
        positions = pool_axis + (method_index - 0.5) * bar_width
        bars = axis.bar(
            positions,
            plr_percent,
            width=bar_width,
            color=color,
            label=method_label,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.2f}%" for value in plr_percent],
            padding=3,
            fontsize=9,
        )

    axis.set(
        title="PLR Comparison across Satellite-Pool Realizations",
        xlabel="Satellite-pool realization",
        ylabel="Packet Loss Rate (%)",
        xticks=pool_axis,
        xticklabels=[
            scenario["chart_label"]
            for scenario in scenario_metadata
        ],
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend()
    figure.tight_layout()
    plt.show()
    raise SystemExit


if RUN_UE_SPATIAL_DISTRIBUTION_COMPARISON:
    import warnings
    from datetime import timedelta
    from pathlib import Path

    from skyfield.api import load, wgs84
    from skyfield.framelib import itrs

    from scenario_time import get_tle_scenario_metadata
    from selection import solve_group_selection_policy

    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    IMBALANCE_EPSILON = 0.001
    SAMPLED_RAO_STEP = 10
    CENTER = (25.03, 121.56)
    SERVICE_RADIUS_KM = 200.0
    SATELLITE_POOL_FILENAME = Path("fixed_satellite_pool.json")
    GROUP_TABLE_FILENAME = Path("group_ps_table.npz")
    OUTPUT_FIGURE = Path("ue_spatial_distribution_theta_accuracy.png")
    OUTPUT_PDF = Path(
        "output/pdf/ue_spatial_distribution_theta_accuracy.pdf"
    )
    BETA_B_VALUES = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
    UE_LOCATION_SEED = SEED
    validate_beta_spatial_sampler(
        CENTER,
        SERVICE_RADIUS_KM,
        BETA_B_VALUES,
        UE_LOCATION_SEED,
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in reduce",
        category=RuntimeWarning,
    )

    def prepare_mode22_ue_geometry(beta_b):
        coordinates = main.generate_ue_locations(
            NUM_UE,
            CENTER,
            SERVICE_RADIUS_KM,
            distribution="beta_enclosed_area",
            random_generator=np.random.RandomState(UE_LOCATION_SEED),
            beta_b=float(beta_b),
        )
        lat_rad = np.deg2rad(coordinates[:, 0])
        lon_rad = np.deg2rad(coordinates[:, 1])
        geo = wgs84.latlon(coordinates[:, 0], coordinates[:, 1])
        ecef = np.asarray(geo.itrs_xyz.km, dtype=float).T
        east = np.column_stack((
            -np.sin(lon_rad),
            np.cos(lon_rad),
            np.zeros(NUM_UE),
        ))
        north = np.column_stack((
            -np.sin(lat_rad) * np.cos(lon_rad),
            -np.sin(lat_rad) * np.sin(lon_rad),
            np.cos(lat_rad),
        ))
        up = np.column_stack((
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ))
        return ecef, east, north, up

    def get_mode22_channel_data(satellite_ecef, ue_geometry):
        ecef, east, north, up = ue_geometry
        delta = satellite_ecef[None, :, :] - ecef[:, None, :]
        up_component = np.einsum("nkd,nd->nk", delta, up)
        east_component = np.einsum("nkd,nd->nk", delta, east)
        north_component = np.einsum("nkd,nd->nk", delta, north)
        angles = np.degrees(np.arctan2(
            up_component,
            np.hypot(east_component, north_component),
        ))
        distances = np.linalg.norm(delta, axis=2)
        channel_ps = main.estimate_channel_success_probability(
            angles,
            distances,
        )
        ranking = np.argsort(angles, axis=1)[:, ::-1]
        return channel_ps, ranking

    def evaluate_mode22_policy(policy, channel_ps, ranking):
        num_sats = channel_ps.shape[1]
        group_codes = ranking[:, 0] * num_sats + ranking[:, 1]
        per_ue_expected_ps = np.empty(channel_ps.shape[0], dtype=float)
        fallback_count = 0

        for group_code in np.unique(group_codes):
            ue_indices = np.flatnonzero(group_codes == group_code)
            group = (
                int(group_code // num_sats),
                int(group_code % num_sats),
            )
            selection_probabilities = policy.get(group)
            if selection_probabilities is None:
                best_satellites = ranking[ue_indices, 0]
                per_ue_expected_ps[ue_indices] = channel_ps[
                    ue_indices,
                    best_satellites,
                ]
                fallback_count += len(ue_indices)
            else:
                per_ue_expected_ps[ue_indices] = (
                    channel_ps[ue_indices]
                    @ np.asarray(selection_probabilities, dtype=float)
                )

        return float(np.mean(per_ue_expected_ps)), fallback_count

    missing_inputs = [
        str(path)
        for path in (SATELLITE_POOL_FILENAME, GROUP_TABLE_FILENAME)
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "Mode 22 requires the baseline satellite pool and uniform "
            "virtual-UE group table. Missing: "
            + ", ".join(missing_inputs)
        )

    with np.load(GROUP_TABLE_FILENAME, allow_pickle=True) as table_data:
        weight_table = table_data["group_weight_table"]
        ps_table = table_data["group_ps_table"]
        table_seconds = int(table_data["seconds"])
        trao_ms = int(table_data["trao_ms"])
        virtual_ue_count = int(table_data["num_points"])
        table_satellite_ids = np.asarray(
            table_data["sat_norad_ids"],
            dtype=int,
        )
        table_start_dt_iso = str(table_data["scenario_start_dt_iso"])
        table_tle_hash = str(table_data["tle_file_sha256"])
    if SECONDS > table_seconds:
        raise ValueError(
            f"Mode 22 requests {SECONDS} seconds, but "
            f"{GROUP_TABLE_FILENAME} only contains {table_seconds} seconds."
        )

    requested_end_rao = SECONDS * 1000 // trao_ms
    if requested_end_rao <= 0:
        raise ValueError(
            "Mode 22 requires SIM_SECONDS to cover at least one RAO."
        )
    if requested_end_rao > len(weight_table):
        raise ValueError(
            f"Mode 22 requests data through RAO {requested_end_rao - 1}, but "
            f"{GROUP_TABLE_FILENAME} only contains {len(weight_table)} RAOs."
        )
    evaluation_rao_indices = np.arange(
        0,
        requested_end_rao,
        SAMPLED_RAO_STEP,
        dtype=int,
    )
    evaluation_count = len(evaluation_rao_indices)

    real_sats = main.load_fixed_satellites(
        filename=str(SATELLITE_POOL_FILENAME)
    )
    actual_satellite_ids = np.asarray([
        int(satellite.model.satnum)
        for satellite in real_sats
    ])
    if not np.array_equal(actual_satellite_ids, table_satellite_ids):
        raise ValueError(
            "Baseline satellite pool does not match the uniform virtual-UE "
            "group table."
        )

    scenario = get_tle_scenario_metadata()
    if table_start_dt_iso != scenario["start_dt_iso"]:
        raise ValueError(
            "Uniform virtual-UE table and current scenario start time do "
            "not match."
        )
    if table_tle_hash != scenario["tle_file_sha256"]:
        raise ValueError(
            "Uniform virtual-UE table and current TLE file do not match."
        )

    ue_geometries = {
        float(beta_b): prepare_mode22_ue_geometry(beta_b)
        for beta_b in BETA_B_VALUES
    }
    timescale = load.timescale()

    print("\n=== Mode 22: Beta Spatial-Distribution Theta Accuracy ===")
    print("Evaluation: offline per-RAO geometry analysis")
    print(f"Reference time: {scenario['start_dt_iso']}")
    print(
        f"Evaluation interval: first {SECONDS} seconds "
        f"({evaluation_count} sampled RAOs, "
        f"{SAMPLED_RAO_STEP * trao_ms} ms sampling interval)"
    )
    print(f"Actual UE count: {NUM_UE}")
    print(f"Virtual UE reference: Uniform ({virtual_ue_count} points)")
    print("Actual UE model: normalized enclosed area u ~ Beta(1, b)")
    print(f"Beta b values: {[float(value) for value in BETA_B_VALUES]}")
    print(f"Service radius: {SERVICE_RADIUS_KM:g} km")
    print("Method: DCLARA")
    print(f"Random seed: {SEED}")

    predicted_ps_by_rao = np.empty(evaluation_count, dtype=float)
    actual_ps_by_beta = {
        float(beta_b): np.empty(evaluation_count, dtype=float)
        for beta_b in BETA_B_VALUES
    }
    fallback_count_by_beta = {
        float(beta_b): 0
        for beta_b in BETA_B_VALUES
    }

    for sample_index, rao in enumerate(evaluation_rao_indices):
        rao_index = int(rao)
        weights = weight_table[rao_index]
        group_ps = ps_table[rao_index]
        policy = solve_group_selection_policy(
            weights,
            group_ps,
            sat_num=len(real_sats),
            imbalance_epsilon=IMBALANCE_EPSILON,
            initial_policy=None,
        )
        effective_load = sum(
            float(weights[group])
            * np.asarray(policy[tuple(group)], dtype=float)
            * np.asarray(group_ps[tuple(group)], dtype=float)
            for group in weights
        )
        predicted_ps_by_rao[sample_index] = float(np.sum(effective_load))

        current_dt = scenario["start_dt"] + timedelta(
            milliseconds=rao_index * trao_ms
        )
        current_time = timescale.from_datetime(current_dt)
        satellite_ecef = np.stack([
            satellite.at(current_time).frame_xyz(itrs).km
            for satellite in real_sats
        ])

        for beta_b in BETA_B_VALUES:
            beta_key = float(beta_b)
            channel_ps, ranking = get_mode22_channel_data(
                satellite_ecef,
                ue_geometries[beta_key],
            )
            actual_ps, fallback_count = evaluate_mode22_policy(
                policy,
                channel_ps,
                ranking,
            )
            actual_ps_by_beta[beta_key][sample_index] = (
                actual_ps
            )
            fallback_count_by_beta[beta_key] += (
                fallback_count
            )

        print(
            f"Evaluated sampled RAO {sample_index + 1}/"
            f"{evaluation_count} (RAO index {rao_index})"
        )

    beta_results = []
    for beta_b in BETA_B_VALUES:
        beta_key = float(beta_b)
        actual_values = actual_ps_by_beta[beta_key]
        prediction_error = actual_values - predicted_ps_by_rao
        fallback_count = fallback_count_by_beta[beta_key]
        beta_results.append({
            "beta_b": beta_key,
            "mean_actual_ps": float(np.mean(actual_values)),
            "mean_predicted_ps": float(np.mean(predicted_ps_by_rao)),
            "ps_mae": float(np.mean(np.abs(prediction_error))),
            "ps_rmse": float(np.sqrt(np.mean(prediction_error ** 2))),
            "ps_bias": float(np.mean(prediction_error)),
            "fallback_rate": float(
                fallback_count / (NUM_UE * evaluation_count)
            ),
        })

    print("\n--- Mode 22 Results ---")
    print(
        f"{'b':>5} | {'Actual theta':>12} | "
        f"{'Pred. theta':>11} | {'MAE':>9} | {'RMSE':>9} | "
        f"{'Bias':>9} | {'Fallback':>9}"
    )
    print("-" * 85)
    for result in beta_results:
        print(
            f"{result['beta_b']:5g} | "
            f"{result['mean_actual_ps']:12.6f} | "
            f"{result['mean_predicted_ps']:11.6f} | "
            f"{result['ps_mae']:9.6f} | "
            f"{result['ps_rmse']:9.6f} | "
            f"{result['ps_bias']:+9.6f} | "
            f"{result['fallback_rate'] * 100:8.3f}%"
        )

    beta_axis = np.array([
        result["beta_b"]
        for result in beta_results
    ], dtype=float)
    ps_mae_values = np.array([
        result["ps_mae"]
        for result in beta_results
    ])

    figure, accuracy_axis = plt.subplots(figsize=(7.5, 5.5), dpi=140)
    accuracy_axis.plot(
        beta_axis,
        ps_mae_values,
        marker="o",
        linewidth=1.6,
    )
    accuracy_axis.set(
        title=r"Impact of Spatial-Distribution Mismatch on $\theta$ Prediction",
        ylabel=r"MAE of $\theta$",
        xlabel=r"Beta concentration parameter $b$",
        xticks=beta_axis,
    )
    accuracy_axis.grid(axis="y", alpha=0.25)
    accuracy_axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"\nSaved figure: {OUTPUT_FIGURE}")
    print(f"Saved PDF: {OUTPUT_PDF}")
    plt.show()
    raise SystemExit


if RUN_BETA_SPATIAL_MISMATCH_COMPARISON:
    import csv
    from pathlib import Path

    NUM_UE = 10000
    SECONDS = SIM_SECONDS
    SEED = 42
    UE_LOCATION_SEED = SEED
    RHO = 1.5
    IMBALANCE_EPSILON = 0.001
    USE_REAL_PS = False
    CENTER = (25.03, 121.56)
    SERVICE_RADIUS_KM = 200.0
    BETA_B_VALUES = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
    MODES = (
        ([6, 1], "DCLARA"),
        ([5, 3], "ALLA with SAACB"),
    )
    OUTPUT_CSV = Path("ue_spatial_beta_plr_results.csv")
    OUTPUT_FIGURE = Path("ue_spatial_beta_plr_comparison.png")
    OUTPUT_PDF = Path("output/pdf/ue_spatial_beta_plr_comparison.pdf")

    validate_beta_spatial_sampler(
        CENTER,
        SERVICE_RADIUS_KM,
        BETA_B_VALUES,
        UE_LOCATION_SEED,
    )

    print("\n=== Mode 23: Beta Spatial-Distribution PLR Comparison ===")
    print("Actual UEs: normalized enclosed area u ~ Beta(1, b)")
    print("Virtual UEs: Uniform disk (unchanged precomputed group table)")
    print(f"Beta b values: {[float(value) for value in BETA_B_VALUES]}")
    print(f"Arrival rate: {RHO:g} packets/s")
    print(f"Simulation duration: {SECONDS} seconds")
    print(f"Simulation seed: {SEED}")
    print(f"UE-location seed: {UE_LOCATION_SEED}")

    results = []
    plr_by_scheme = {label: [] for _, label in MODES}
    for beta_b in BETA_B_VALUES:
        print(f"\nb={beta_b:g}")
        for mode, label in MODES:
            (
                average_throughput,
                plr,
                _,
                _,
                _,
                _,
                run_history,
            ) = main.main(
                RHO,
                SECONDS,
                NUM_UE,
                mode,
                SEED,
                IMBALANCE_EPSILON,
                USE_REAL_PS=USE_REAL_PS,
                UE_SPATIAL_DISTRIBUTION="beta_enclosed_area",
                UE_LOCATION_SEED=UE_LOCATION_SEED,
                UE_SPATIAL_BETA_B=float(beta_b),
            )
            plr = float(plr)
            average_throughput = float(average_throughput)
            average_delay_ms = float(run_history["average_delay_ms"])
            plr_by_scheme[label].append(plr)
            results.append({
                "beta_b": float(beta_b),
                "scheme": label,
                "selection_mode": int(mode[0]),
                "backoff_mode": int(mode[1]),
                "plr": plr,
                "average_throughput": average_throughput,
                "average_delay_ms": average_delay_ms,
                "arrival_rate_packets_per_s": RHO,
                "simulation_seconds": SECONDS,
                "num_ue": NUM_UE,
                "simulation_seed": SEED,
                "ue_location_seed": UE_LOCATION_SEED,
                "service_radius_km": SERVICE_RADIUS_KM,
            })
            print(f"{label} PLR = {plr:.8f}")

            # Preserve completed runs even if a later long simulation stops.
            with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=list(results[0]),
                )
                writer.writeheader()
                writer.writerows(results)

    figure, axis = plt.subplots(figsize=(7.5, 5.5), dpi=140)
    for _, label in MODES:
        axis.plot(
            BETA_B_VALUES,
            plr_by_scheme[label],
            marker="o",
            linewidth=1.6,
            label=label,
        )
    axis.set(
        title="PLR Comparison under Spatial-Distribution Mismatch",
        xlabel=r"Beta concentration parameter $b$",
        ylabel="Packet Loss Rate",
        xticks=BETA_B_VALUES,
    )
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    print(f"\nSaved raw results: {OUTPUT_CSV}")
    print(f"Saved figure: {OUTPUT_FIGURE}")
    print(f"Saved PDF: {OUTPUT_PDF}")
    plt.show()
    raise SystemExit


# Current single-run experiment.
num = 10000
m = [6,1] #Satellite selection mode and backoff control mode.
USE_REAL_PS = False
# Proposed satellite selection and backoff control.
a, b, c, d, e, f, g = main.main(
    1.2,
    SIM_SECONDS,
    num,
    m,
    42,
    0.01,
    USE_REAL_PS=USE_REAL_PS,
    COLLECT_BACKOFF_OPTIMIZER_DIAGNOSTICS=(EXPERIMENT_CODE == 0),
)

backoff_optimizer_history = g.get("backoff_optimizer_history", [])
if len(backoff_optimizer_history) > 0:
    print("\n--- Backoff Optimizer Update History ---")
    for item in backoff_optimizer_history:
        initial_p_b_text = ", ".join(
            f"{value:.6f}" for value in item["initial_p_b"]
        )
        final_p_b_text = ", ".join(
            f"{value:.6f}" for value in item["final_p_b"]
        )
        print(
            f"RAO {item['time_slot']}: "
            f"iterations={item['iterations']}, "
            f"function_evaluations={item['function_evaluations']}, "
            f"success={item['success']}, "
            f"max_update={item['max_abs_update']:.3e}, "
            f"loss={item['initial_loss']:.10f} -> "
            f"{item['final_loss']:.10f}"
        )
        print(f"  Initial p_b: [{initial_p_b_text}]")
        print(f"  Updated p_b: [{final_p_b_text}]")
        if not item["success"]:
            print(
                f"  Optimizer status {item['status']}: "
                f"{item['message']}"
            )

    iteration_counts = np.asarray(
        [item["iterations"] for item in backoff_optimizer_history],
        dtype=int,
    )
    update_sizes = np.asarray(
        [item["max_abs_update"] for item in backoff_optimizer_history],
        dtype=float,
    )
    success_count = sum(
        bool(item["success"]) for item in backoff_optimizer_history
    )
    print("\n--- Backoff Optimizer Iteration Summary ---")
    print(f"Recorded RAOs: {len(backoff_optimizer_history)}")
    print(
        f"Iterations: mean={np.mean(iteration_counts):.3f}, "
        f"median={np.median(iteration_counts):.3f}, "
        f"max={np.max(iteration_counts)}"
    )
    print(
        f"Zero-iteration RAOs: {np.sum(iteration_counts == 0)}/"
        f"{len(iteration_counts)} "
        f"({np.mean(iteration_counts == 0) * 100:.2f}%)"
    )
    for iteration_value in (1, 2):
        matching_count = int(np.sum(iteration_counts == iteration_value))
        print(
            f"{iteration_value}-iteration RAOs: {matching_count}/"
            f"{len(iteration_counts)} "
            f"({matching_count / len(iteration_counts) * 100:.2f}%)"
        )
    print(
        f"Numerically unchanged RAOs (max update <= 1e-8): "
        f"{np.sum(update_sizes <= 1e-8)}/{len(update_sizes)} "
        f"({np.mean(update_sizes <= 1e-8) * 100:.2f}%)"
    )
    print(
        f"Successful optimizations: {success_count}/"
        f"{len(backoff_optimizer_history)}"
    )

    unique_iterations, iteration_frequencies = np.unique(
        iteration_counts,
        return_counts=True,
    )
    iteration_percentages = (
        iteration_frequencies / len(iteration_counts) * 100.0
    )
    bar_positions = np.arange(len(unique_iterations))
    figure, axis = plt.subplots(figsize=(10, 6), dpi=120)
    bars = axis.bar(
        bar_positions,
        iteration_frequencies,
        width=0.7,
    )
    for bar, frequency, percentage in zip(
        bars,
        iteration_frequencies,
        iteration_percentages,
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{frequency}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
        )
    axis.set(
        title="Distribution of Backoff Optimization Iterations",
        xlabel="Number of L-BFGS-B iterations",
        ylabel="Number of RAOs",
        xticks=bar_positions,
        xticklabels=[str(value) for value in unique_iterations],
    )
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_ylim(0, max(iteration_frequencies) * 1.18)
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)
    figure.tight_layout()
    iteration_pdf_filename = "backoff_optimizer_iteration_distribution.pdf"
    figure.savefig(
        iteration_pdf_filename,
        format="pdf",
        bbox_inches="tight",
    )
    print(f"Saved iteration distribution figure to {iteration_pdf_filename}")
    plt.show()

# Keep the scalar diagnostics without generating the previous Mode 0 figures.
ps_history = g.get("ps_history", [])
ps_mae = np.nan
if len(ps_history) > 0:
    ps_error = np.asarray(
        [item["error"] for item in ps_history],
        dtype=float,
    )
    ps_mae = np.mean(np.abs(ps_error))

print("--- Test Complete ---")
print(f"Packet Loss Rate: {b:.4f}")
print(f"Average Throughput: {a:.2f}")
single_delay_ms = g.get("average_delay_ms", np.nan)
print(f"Average Delay (ms): {single_delay_ms:.2f}")
if m[1] == 1:
    print(f"p_s MAE: {ps_mae:.6f}" if np.isfinite(ps_mae) else "p_s MAE: N/A")
