import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ==================== Config ====================
CONFIG = {
    "N": 64,
    "max_ticks": 250,
    "diag_interval": 10,
    "ensemble_seeds": [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021],
    "runs": {
        "A-placebo": {"payload_bits": 0.0, "pm_weight": 0.0, "echo_offset": None, "max_ticks": 200},
        "B-1.5":     {"payload_bits": 1.5, "pm_weight": 0.5, "echo_offset": None},
        "C-2.0":     {"payload_bits": 2.0, "pm_weight": 0.0, "echo_offset": None},
        "D-2.5":     {"payload_bits": 2.5, "pm_weight": 0.5, "echo_offset": None},
        "E-2.0-echo40": {"payload_bits": 2.0, "pm_weight": 0.0, "echo_offset": 40},
        "F-2.5-echo40": {"payload_bits": 2.5, "pm_weight": 0.5, "echo_offset": 40},
    }
}

# Save config
os.makedirs("results", exist_ok=True)
with open("results/config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

# ==================== Margolus Fredkin Gate ====================
def margolus_apply(grid, even=True):
    """Apply one Margolus partition with a reversible Fredkin-style rule."""
    n = grid.shape[0]
    new_grid = grid.copy()
    offset = 0 if even else 1
    for i in range(offset, n, 2):
        for j in range(offset, n, 2):
            i1 = (i + 1) % n
            j1 = (j + 1) % n
            a, b, d, c = grid[i,j], grid[i,j1], grid[i1,j1], grid[i1,j]  # clockwise order

            # Conditional clockwise rotation if top-left (a) is 1
            if a == 1:
                new_grid[i,j] = c
                new_grid[i,j1] = a
                new_grid[i1,j] = d
                new_grid[i1,j1] = b

    return new_grid

def fredkin_step(visible):
    """One full CA tick: two partitions."""
    visible = margolus_apply(visible, even=True)
    visible = margolus_apply(visible, even=False)
    return visible

# ==================== Lattice & Diagnostics ====================
class Lattice:
    def __init__(self, N=64, payload_bits=0.0, pm_weight=0.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.N = N
        self.area = N * N
        self.visible = np.random.randint(0, 2, (N, N), dtype=np.uint8)
        self.PM = np.zeros((N, N), dtype=bool)
        self.PM_weight = pm_weight

        if payload_bits > 0.0:
            cx, cy = N//2, N//2
            cx2, cy2 = N//2, (N//2 + 1) % N
            self.visible[cx, cy] = 0
            self.visible[cx2, cy2] = 1
            self.PM[cx, cy] = True
            self.PM[cx2, cy2] = True

def info_budget(lattice):
    vis = np.sum(lattice.visible)
    hidden = np.sum(lattice.PM) * lattice.PM_weight
    return vis + hidden

def paradox_density(lattice):
    return np.mean(lattice.PM)

def resonance(lattice):
    return info_budget(lattice) * paradox_density(lattice)

def free_energy_curvature(history):
    if len(history) < 5:
        return 0.0
    N_half = history[0].area / 2.0
    mags = [np.sum(h.visible) - N_half for h in history[-5:]]
    return np.var(mags)

# ==================== Simulation ====================
def run_simulation(run_id, params, seed):
    lattice = Lattice(N=CONFIG["N"],
                      payload_bits=params["payload_bits"],
                      pm_weight=params["pm_weight"],
                      seed=seed)
    max_ticks = params.get("max_ticks", CONFIG["max_ticks"])
    echo_offset = params["echo_offset"]

    history = []  # only keep recent for kappa
    metrics = []
    echo_state = None

    for t in range(max_ticks + 1):  # include t=0
        if echo_offset is not None and t == 0:
            echo_state = (lattice.visible.copy(), lattice.PM.copy())

        if t > 0:
            lattice.visible = fredkin_step(lattice.visible)

        if echo_offset is not None and t == echo_offset:
            lattice.visible ^= echo_state[0]
            lattice.PM |= echo_state[1]

        if t % CONFIG["diag_interval"] == 0:
            I = info_budget(lattice)
            P = paradox_density(lattice)
            R = resonance(lattice)
            kappa = free_energy_curvature(history)
            bound = lattice.area / (4 * np.log(2))
            EH = I > bound
            metrics.append({
                "run_id": run_id,
                "seed": seed,
                "t": t,
                "I": float(I),
                "P": float(P),
                "R": float(R),
                "kappa": float(kappa),
                "EH": EH,
                "bound": float(bound)
            })

        history.append(lattice)
        if len(history) > 5:
            history.pop(0)

    return metrics

# ==================== Execute All Runs ====================
all_metrics = []
for run_id, params in CONFIG["runs"].items():
    print(f"Running {run_id} ...")
    for seed in CONFIG["ensemble_seeds"]:
        mets = run_simulation(run_id, params, seed)
        all_metrics.extend(mets)
        # Save individual raw CSV
        df_ind = pd.DataFrame(mets)
        df_ind.to_csv(f"results/raw_{run_id}_seed{seed}.csv", index=False)

# Master raw
df_all = pd.DataFrame(all_metrics)
df_all.to_csv("results/all_raw.csv", index=False)

# ==================== Analysis ====================
summary_stats = []  # list of mean DataFrames (with t index)
peak_stats = []
integrated_R = {}
eh_durations = []  # final cumulative EH duration per run

for run_id in CONFIG["runs"]:
    df_run = df_all[df_all["run_id"] == run_id]
    
    # Mean ± std over seeds at each t
    grouped = df_run.groupby("t")
    mean_df = grouped.mean(numeric_only=True)
    std_df = grouped.std(numeric_only=True)
    mean_df["run_id"] = run_id
    mean_df["std_I"] = std_df["I"]
    mean_df["std_P"] = std_df["P"]
    mean_df["std_R"] = std_df["R"]
    mean_df["std_kappa"] = std_df["kappa"]
    mean_df["EH_frac"] = grouped["EH"].mean()  # fraction of seeds violating
    
    # Compute cumulative EH duration (average over ensemble)
    dt = CONFIG["diag_interval"]
    eh_frac_series = mean_df["EH_frac"]
    cum_eh = (eh_frac_series.cumsum() * dt)
    mean_df["EH_duration_cum"] = cum_eh
    
    summary_stats.append(mean_df.reset_index())
    
    # Peaks (on mean R)
    peak_row = mean_df.loc[mean_df["R"].idxmax()]
    peak_stats.append({
        "run_id": run_id,
        "peak_t": int(peak_row.name),
        "peak_R": peak_row["R"],
        "peak_I": peak_row["I"],
        "peak_P": peak_row["P"],
        "peak_kappa": peak_row["kappa"]
    })
    
    # Integrated resonance ≈ sum R * Δt (using mean R)
    integrated = (mean_df["R"] * dt).sum()
    integrated_R[run_id] = integrated
    
    # Final EH duration for this run
    eh_durations.append(cum_eh.iloc[-1])

# Save summaries
pd.concat(summary_stats).to_csv("results/summary_by_run.csv", index=False)
pd.DataFrame(peak_stats).to_csv("results/peaks.csv", index=False)

# ==================== Plotting ====================
fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

for run_id in CONFIG["runs"]:
    df_run = pd.concat(summary_stats)[pd.concat(summary_stats)["run_id"] == run_id]
    
    label = run_id + f" ({CONFIG['runs'][run_id]['payload_bits']} bits)"
    axs[0].plot(df_run["t"], df_run["I"], label=label)
    axs[1].plot(df_run["t"], df_run["P"], label=label)
    axs[2].plot(df_run["t"], df_run["R"], label=label)
    axs[3].plot(df_run["t"], df_run["kappa"], label=label)
    axs[4].plot(df_run["t"], df_run["EH_frac"], label=label)

axs[0].set_ylabel("Info Budget I (bits)")
axs[1].set_ylabel("Paradox Density P")
axs[2].set_ylabel("Resonance R = I × P")
axs[3].set_ylabel("Curvature proxy κ")
axs[4].set_ylabel("EH violation fraction")
axs[4].set_xlabel("Time (ticks)")
for ax in axs:
    ax.legend()
    ax.grid(True)

plt.suptitle("Paradox Dose Sweep – Ensemble Average (10 seeds)")
plt.tight_layout()
plt.savefig("results/all_metrics_timeseries.png")
plt.close()

# Dose vs Resonance plot
dose_map = {k: v["payload_bits"] for k, v in CONFIG["runs"].items()}
df_peaks = pd.DataFrame(peak_stats)
df_peaks["dose"] = df_peaks["run_id"].map(dose_map)
df_peaks["integrated_R"] = df_peaks["run_id"].map(integrated_R)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df_peaks["dose"], df_peaks["peak_R"], 'o-', label="Peak R")
ax1.plot(df_peaks["dose"], df_peaks["integrated_R"], 's-', label="Integrated R")
ax1.set_xlabel("Injected Paradox Dose (bits)")
ax1.set_ylabel("Resonance Metric")
ax1.legend(loc="upper left")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(df_peaks["dose"], eh_durations, 'd--', color='red', label="Avg EH duration (ticks)")
ax2.set_ylabel("Avg EH violation duration (ticks)", color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Resonance vs. Paradox Dose – Optimal Window Identification")
plt.savefig("results/dose_sweep_summary.png")

print("All simulations complete.")
print("Raw data in results/raw_*.csv")
print("Summaries in results/summary_by_run.csv and peaks.csv")
print("Plots in results/*.png")
print("\nDebug fix applied: EH_duration_cum is now correctly computed on the mean DataFrame,")
print("and final EH durations are extracted properly for the dose-sweep plot.")
print("The script should now run without the KeyError.")
