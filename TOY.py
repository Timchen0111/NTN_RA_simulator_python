import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==========================================
# 1. Data Generation (Same as before)
# ==========================================
T_ref = 10.0
epsilon = 0.1

def calc_f1(g, others, T): return max(0.0, 1 - g/T)
def calc_f2(g, others, T): return max(0.0, (1 - g/T)/(1+len(others)))
def calc_f3(g, others, T, eps): return max(0.0, 1 - (g + eps*sum(others))/T)
def calc_f4(g, others, T, eps): return max(0.0, 1 - (g + eps*(max(others) if others else 0))/T)

# Scenarios
scenarios_general = [
    ("Panic (No Backup)", 1.0, []),
    ("Weak Backup",       1.0, [1.0]),
    ("Strong Backup",     1.0, [5.0]),
    ("Safe (No Backup)",  5.0, []),
    ("Safe (Weak)",       5.0, [1.0]),
    ("Safe (Strong)",     5.0, [5.0]),
]

scenarios_iso = [
    ("Quality Case (1x10s)",  1.0, [10.0]),
    ("Quantity Case (10x1s)", 1.0, [1.0]*10),
]

def get_data(scenarios):
    data = []
    for name, g, others in scenarios:
        data.append({
            "Scenario": name,
            "Baseline (f1)": calc_f1(g, others, T_ref),
            "Count-Aware (f2)": calc_f2(g, others, T_ref),
            "Proposed (f3)": calc_f3(g, others, T_ref, epsilon),
            "Max-Based (f4)": calc_f4(g, others, T_ref, epsilon)
        })
    return pd.DataFrame(data)

df_gen = get_data(scenarios_general)
df_iso = get_data(scenarios_iso)

# ==========================================
# 2. Transpose Data for "Group by Formula"
# ==========================================
# Pivot so that Index is Metric, Columns are Scenarios
df_gen_pivot = df_gen.set_index("Scenario").T
df_iso_pivot = df_iso.set_index("Scenario").T

# ==========================================
# 3. Plotting
# ==========================================
plt.rcParams.update({'font.size': 11})
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: General Comparison (Panic vs Safe)
# Group by Formula (x-axis), Bars are Scenarios
df_gen_pivot.plot(kind="bar", ax=axes[0], width=0.8, colormap="viridis")

axes[0].set_title("Behavior of Metrics across Different Scenarios")
axes[0].set_ylabel("Urgency Score")
axes[0].set_xlabel("Urgency Metric Formulations")
axes[0].set_ylim(0, 1.1)
axes[0].legend(title="Scenarios", loc='upper right', bbox_to_anchor=(1.15, 1))
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=0)

# Plot 2: Iso-Capacity (Quality vs Quantity)
df_iso_pivot.plot(kind="bar", ax=axes[1], width=0.6, color=['#1f77b4', '#ff7f0e']) # Blue and Orange

axes[1].set_title("Iso-Capacity Analysis: Quality (1x10s) vs Quantity (10x1s)")
axes[1].set_ylabel("Urgency Score")
axes[1].set_xlabel("Urgency Metric Formulations")
axes[1].set_ylim(0, 1.1)
axes[1].legend(title="Scenarios", loc='upper right', bbox_to_anchor=(1.15, 1))
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('urgency_score_comparison_grouped_by_formula.png')
print("Plots generated.")