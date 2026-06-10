import numpy as np
import Load_estimator as l
import old as main
import matplotlib.pyplot as plt
'''
# 1. 閮剖?撖阡??
rho = [0.2,0.4,0.6] 
SEEDS = [13] 
MODES = [1, 2] 

# ?脣??蝯???摮
# 蝯???{mode: [(avg_tp, avg_sr), ...]}
final_results = {mode: [] for mode in MODES}

for count in rho:
    # ?脣??嗅? UE ?賊?銝???Mode ??蝮賣??
    mode_sums = {mode: [0.0, 0.0] for mode in MODES} # [tp_sum, sr_sum]
    
    for s in SEEDS:
        print(f"Running simulations for UE count: {count} with seed: {s}...")
        
        for mode in MODES:
            # 瘥活?瑁???蝵桃車摮?蝣箔??詨? UE ??????
            np.random.seed(s)
            
            # ?澆 main 銝血?敺??喳?(throughput, success_rate)
            tp, sr = main.main(count, 1, 100, mode, s)
            
            mode_sums[mode][0] += tp
            mode_sums[mode][1] += sr
            
    # 閮?撟喳??潔蒂摮?蝯???
    num_seeds = len(SEEDS)
    for mode in MODES:
        avg_tp = mode_sums[mode][0] / num_seeds
        avg_sr = mode_sums[mode][1] / num_seeds
        final_results[mode].append((avg_tp, avg_sr))

# 2. 皞?蝜芸??豢?
labels = {1: 'with BACKOFF', 2: 'no BACKOFF'}
colors = {1: 'purple', 2: 'blue'}
markers = {1: 'o', 2: 's'}

# 3. 閮剖??怠?
plt.figure(figsize=(14, 6))

# --- 撌血?嚗verage Throughput 瘥? ---
plt.subplot(1, 2, 1)
for mode in MODES:
    y_tp = [r[0] for r in final_results[mode]]
    plt.plot(rho, y_tp, marker=markers[mode], color=colors[mode], label=labels[mode])

plt.title('Active Rate vs Average Throughput')
plt.xlabel('Active Rate (RHO)')
plt.ylabel('Average Throughput (packets/second)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# --- ?喳?嚗uccess Rate 瘥? ---
plt.subplot(1, 2, 2)
for mode in MODES:
    y_sr = [r[1] for r in final_results[mode]]
    plt.plot(rho, y_sr, marker=markers[mode], color=colors[mode], label=labels[mode])

plt.title('Active Rate vs Success Rate')
plt.xlabel('Active Rate (RHO)')
plt.ylabel('Success Rate')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# 4. 憿舐內蝯?
plt.tight_layout()
plt.show()
'''

'''
MODE隤芣?
MODE1: RL + backoff (Proposed scheme)
MODE2: backoff only (With same satellite selection score)
MODE3: Baseline: No Satellite Selection, Non-state dependent backoff
MODE4: Heuristic satellite selection + backoff
MODE5: Iterative RL training and MODE4
MODE6: backoff + IDEAL SCENARIO, same visibility for all UEs, same satellite selection score
MODE7: Non-state dependent backoff + IDEAL SCENARIO, same visibility for all UEs, same satellite selection score
'''
'''
import matplotlib.pyplot as plt
import numpy as np

# --- 璅⊥??????---
num = 20000
parameter_set = [[3,3,0.01],[4,3,0.01],[6,19,0.01]]  #Mode,頠??Ｘ???航?銵??賊?
results = {}
true_pi = None

for i in range(len(parameter_set)):
    # ?瑁? main.main 銝血?敺??喳?
    # 瘜冽?嚗ㄐ???詨???雿? main ?賢?摰儔?湔撠?
    # ?喳?桀?????賊? i嚗摰芋撘?m (撱箄降?摰???摰芋撘脰?霈??)
    a, b, c, d, e, f, g = main.main(parameter_set[i][2], parameter_set[i][1],30, num, parameter_set[i][0], 42, 50)
    results[i] = {
        'N_tilde': c, 
        'Pi': d, 
        'Loads': a, 
        'PLR': b, 
        'True_Pi': e,  # 撠?銝蝯祕撽??芰? True Pi 摮?靘??銵??貉?鈭??? True Pi 銋?霈?
        'Reward': f
    }

# ?箔???頠??Ｘ??????脣?瘥?(雿輻 colormap 霈?銵冽撠平)
colors = plt.cm.viridis(np.linspace(0, 0.8, len(parameter_set)))
orbit_configs = {i: {'color': colors[idx]} for idx, i in enumerate(range(len(parameter_set)))}


# --- ?” 1嚗?????賊?銝?鈭箏隡啗??嗆?瘥? (N_tilde) ---
plt.figure(figsize=(10, 6))
# ?祕?蜇?冽?訾??嗆?箏??皞?
plt.axhline(y=num, color='black', linestyle='--', label=f'True N ({num})', alpha=0.6)

# 撠???{} ?寧?” []嚗Ⅱ靽揣撘?i ?賣迤蝣箏???
#labels = ['Real Satellite Scenario, rho = 0.01', 'Real Satellite Scenario, rho = 0.005', 'Ideal Case: Uniform Visibility, rho = 0.01', 'Ideal Case: Uniform Visibility, rho = 0.005']
labels = ['Non-state dependent Backoff (Real Satellite)', 'Real Satellite Scenario', 'Ideal Case: Uniform Visibility']
for i in range(len(parameter_set)):
    plt.plot(range(len(results[i]['N_tilde'])), results[i]['N_tilde'], 
             label=labels[i], color=orbit_configs[i]['color'], linewidth=1.5)
plt.xlabel('Time Slot (n)')
plt.ylabel('Estimated Population')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# --- ?” 2嚗?????賊?銝? Pi 隡啗?蝯?隤文榆瘥? (隞?State 0 ?箔?) ---
plt.figure(figsize=(10, 6))
target_state = 0  # 閫撖?State 0 (Idle)

for i in range(len(parameter_set)):
    data_matrix = np.array(results[i]['Pi'])         # 璅⊥銝凋摯閮? Observed Pi 甇瑕蝝??
    current_true_pi = results[i]['True_Pi']          # 閰脤?蝵桐??頂蝯梁?撖?True Pi ?箏???
    
    # 閮?瘥?Time Slot ??撠炊撌殷?|Observed_Pi - True_Pi|
    estimation_error = np.abs(data_matrix[:, target_state] - current_true_pi[target_state])
    
    # 靽格迤??嚗?典?銵剁?銝衣宏?日?銴? color ??linewidth ?
    plt.plot(range(len(estimation_error)), estimation_error, 
             label=labels[i], color=orbit_configs[i]['color'], linewidth=1.5)

# ??瘜??炊撌桀皞?嚗炊撌桃 0嚗?
plt.axhline(y=0, color='black', linestyle='--', alpha=0.6, label='Ideal Estimation (Zero Error)')

plt.title(f'State {target_state} Estimation Absolute Error across Different Orbit Scales', fontsize=12)
plt.xlabel('Time Slot (n)')
plt.ylabel('Absolute Error |Observed - True|')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()

# 憛怠雿?撖行葫?豢?
plr_real = results[0]["PLR"]
plr_real2 = results[1]["PLR"]
plr_ideal = results[2]["PLR"]
#plr_ideal2 = results[3]["PLR"]

plr_values = [plr_real, plr_real2, plr_ideal] #, plr_ideal2]

plt.figure(figsize=(6, 5))
bars = plt.bar(labels, plr_values, width=0.4)

# ?券璇?銝璅酉?詨?
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.01,
        f"{yval:.4f}",
        ha="center",
        va="bottom",
    )

plt.ylabel("Packet Loss Rate (PLR)")
plt.title("PLR Comparison: Real vs. Ideal Case")
plt.ylim(0, max(plr_values) + 0.1)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

print(f"--- Test Complete ---")
print(f"Final PLR:")
print(f"Real satellite scenario, Non-state dependent Backoff: {results[0]['PLR']:.4f}")
print(f"Real satellite scenario: {results[1]['PLR']:.4f}")
print(f"Ideal case: {results[2]['PLR']:.4f}")
print(f"")
'''

import matplotlib.pyplot as plt
import numpy as np
import main


num = 10000
m = 1
USE_REAL_PS = True
results = {}

#def main(RHO, NUM_SAT, SECONDS, NUM_UE,MODE, SEED, NUM_EPOCHS, IMBALANCE_EPSILON=1000)
a, b, c, d, e, f, g = main.main(0.02, 50, num, m, 42, 1, 0.001, USE_REAL_PS=USE_REAL_PS)
results[m] = {
    'N_tilde': c, 
    'Pi': d, 
    'Loads': a, 
    'SuccessRate': b, 
    'Reward': f, 
    'TruePi': e,
    'EpisodeHistory': g
}


plt.figure(figsize=(10, 6))
plt.axhline(y=num, color='black', linestyle='--', label=f'True N ({num})', alpha=0.6)
plt.plot(range(len(results[m]['N_tilde'])), results[m]['N_tilde'], 
         label='MODE 1: RL + Backoff', color='blue', linewidth=1.5)

plt.title('Population Estimation Convergence (N_tilde) - MODE 1 Only')
plt.xlabel('Time Slot (n)')
plt.ylabel('Estimated Population')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(range(len(results[m]['Reward'])), results[m]['Reward'], 
         label='Reward (Negative Variance)', color='blue', linewidth=1.2)


plt.title('Reward Variation over Time Slots')
plt.xlabel('Time Slot (n)')
plt.ylabel('Reward (-Var)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# --- Plot 3: state distribution comparison ---
# --- Plot 3: state distribution over time for all pi_n states ---
pi_history = np.asarray(results[m]['Pi'], dtype=float)
estimated_active_pi = np.asarray(results[m]['TruePi'], dtype=float)
estimated_pi = np.concatenate(([max(0.0, 1.0 - np.sum(estimated_active_pi))], estimated_active_pi))

if pi_history.size > 0:
    if pi_history.ndim == 1:
        pi_history = pi_history.reshape(1, -1)

    record_interval = 10
    time_slots = np.arange(pi_history.shape[0]) * record_interval
    state_count = pi_history.shape[1]

    plt.figure(figsize=(10, 6))
    for state_idx in range(state_count):
        state_label = 'Idle' if state_idx == 0 else f'State {state_idx}'
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
                linestyle='--',
                linewidth=1.0,
                alpha=0.65,
                label=f'Estimated {state_label}',
            )

    plt.title(r'Convergence of State Distributions ($\pi_n$)')
    plt.xlabel('Time Slot (n)')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No pi history was recorded; skip state distribution plot.")

print(f"--- Test Complete ---")
print(f"Packet Loss Rate: {results[m]['SuccessRate']:.4f}")



'''
# --- Epsilon sweep for convex group satellite selection ---
import matplotlib.pyplot as plt
import numpy as np
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
'''
