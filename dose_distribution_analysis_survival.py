# ---------------------------
# 0. Set Random Seed for Reproducibility
# ---------------------------
random_seed = 42

import random
import numpy as np
random.seed(random_seed)
np.random.seed(random_seed)

from scipy import stats
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

g = torch.Generator()
g.manual_seed(random_seed)

# ---------------------------
# 1. Standard Python Libraries
# ---------------------------
import os
from tqdm import tqdm
# ---------------------------
# 2. Data Handling and Visualization
# ---------------------------
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------
# 4. PyTorch and Related Imports
# ---------------------------
from torch.utils.data import Dataset

import scienceplots
plt.style.use(['science', 'grid'])
plt.rcParams['figure.dpi'] = 500
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

SAVE_PATH = '/content/drive/MyDrive/Burdenko/Survival/'
test_path = f"{SAVE_PATH}val_dataset_survival_preprocessed.pt"
train_path = f"{SAVE_PATH}train_dataset_survival_preprocessed.pt"
dose_parameters = f"{SAVE_PATH}PFS_filtered_data.csv"

class GBMDataset3DRaw(Dataset):
    def __init__(self, data_list):
        """
        A “raw” GBM 3D dataset that does no processing whatsoever.

        Args:
            data_list (list): List of samples, each a dict with keys
                'image' (torch.Tensor [3, D, H, W]),
                'label' (int or tensor),
                'clinical_features' (torch.Tensor),
                'patient_id' (any).
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'image': sample['image'],               # raw tensor [C=3, D, H, W]
            'label': sample['label'],               # raw label
            'clinical_features': sample['clinical_features'],
            'patient_id': sample['patient_id']
        }

train_data_list = torch.load(train_path)
test_data_list  = torch.load(test_path)

train_dataset = GBMDataset3DRaw(train_data_list)
test_dataset  = GBMDataset3DRaw(test_data_list)

# --------------------- PARAMETERS --------------------------------------
PROJ_MODE    = 'mean'    # 'mean' or 'sum' along the z‑axis
ALPHA        = 0.05      # p‑value threshold for voxel map (two‑sided)
CI_ALPHA     = 0.10      # p‑value threshold for confidence voxels (two‑sided)
DOSE_THRESH  = 0.01      # minimum mean relative dose per voxel to include
os.makedirs(SAVE_PATH, exist_ok=True)

# --------------------- 1. Build metadata table ------------------------
meta_rows = []
all_samples = train_dataset + test_dataset
for sample in all_samples:
    meta_rows.append({
        "sample": sample,
        "group":   int(sample["label"]),   # 0 = SHORT, 1 = LONG
        "pid":     int(sample["patient_id"])
    })
meta = pd.DataFrame(meta_rows)
group_labels = ['SHORT (<600 d)', 'LONG (≥600 d)']
print("Patients per group:", meta.group.value_counts().to_dict())

# --------------------- 2. Initialize accumulators ----------------------
any_img = meta.iloc[0]["sample"]["image"][2]
_, H, W = any_img.shape
sum_proj      = np.zeros((2, H, W), float)
sum_sq_proj   = np.zeros_like(sum_proj)
counts        = np.zeros(2, int)
sum_head      = np.zeros((H, W), float)
patient_stats = []
coords = np.indices((H, W))  # (y, x) grid

# --------------------- 3. Loop over patients --------------------------
for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Accumulating"):
    sample, grp = row["sample"], row["group"]

    # Dose projection
    dose_vol = sample["image"][2].float().cpu().numpy()
    max_val  = dose_vol.max() if dose_vol.max()>0 else 1.0
    dose_rel = dose_vol / max_val
    proj = dose_rel.mean(axis=0) if PROJ_MODE=='mean' else dose_rel.sum(axis=0)

    stats_dict = {"group": grp}
    mass = proj.sum()
    if mass > 0:
        # center of mass and radius of gyration (within full image)
        y_com = (coords[0]*proj).sum() / mass
        x_com = (coords[1]*proj).sum() / mass
        r2    = (coords[1]-x_com)**2 + (coords[0]-y_com)**2
        r_g   = np.sqrt((proj * r2).sum() / mass)

        # max-dose location and r_g_max
        y_max, x_max = np.unravel_index(proj.argmax(), proj.shape)
        r2_max = (coords[1]-x_max)**2 + (coords[0]-y_max)**2
        r_g_max = np.sqrt((proj * r2_max).sum() / mass)

        # dose-falloff gradient (max |ΔD/Δr|)
        rmap   = np.sqrt((coords[0]-y_max)**2 + (coords[1]-x_max)**2)
        flat_r = rmap.ravel().astype(int)
        flat_p = proj.ravel()
        max_r  = flat_r.max()
        M      = np.zeros(max_r+1, float)
        counts_r = np.zeros_like(M, int)
        for ri, pv in zip(flat_r, flat_p):
            M[ri] += pv
            counts_r[ri] += 1
        M[counts_r>0] /= counts_r[counts_r>0]
        dM = np.diff(M)
        falloff_grad = np.max(np.abs(dM))

        stats_dict.update({
            "mean_rel_dose": float(dose_rel.mean()),
            "xCOM": float(x_com),    "yCOM": float(y_com),
            "r_g": float(r_g),
            "xMAX": float(x_max),    "yMAX": float(y_max),
            "r_g_max": float(r_g_max),
            "falloff_grad": float(falloff_grad)
        })

    patient_stats.append(stats_dict)

    # accumulate for group‑wise projection
    sum_proj[grp]    += proj
    sum_sq_proj[grp] += proj**2
    counts[grp]      += 1

    # accumulate head MR average
    mr0 = sample["image"][0].float().cpu().numpy()
    mr1 = sample["image"][1].float().cpu().numpy()
    sum_head += (mr0.mean(axis=0) + mr1.mean(axis=0)) / 2

patient_stats = pd.DataFrame(patient_stats)

# --------------------- 4. Compute group & head stats -----------------
mean_proj = sum_proj / counts[:, None, None]
std_proj  = np.sqrt(sum_sq_proj/counts[:, None, None] - mean_proj**2)
mean_head = sum_head / len(meta)
boundary  = mean_head.mean()
head_mask = mean_head >= boundary

# mask outside head for all images and stats
mean_proj[:, ~head_mask] = np.nan
std_proj[:, ~head_mask]  = np.nan

# voxel‑wise t‑test
m0, m1 = mean_proj[0], mean_proj[1]
s0, s1 = std_proj[0],   std_proj[1]
n0, n1 = counts[0],     counts[1]
var    = s0**2/n0 + s1**2/n1
with np.errstate(divide='ignore', invalid='ignore'):
    t_map = (m1 - m0) / np.sqrt(var)
    df    = var**2 / ((s0**2/n0)**2/(n0-1) + (s1**2/n1)**2/(n1-1))
    p_map = stats.t.sf(np.abs(t_map), df) * 2
# mask outside head
t_map[~head_mask] = np.nan
p_map[(~head_mask) | (var==0) | (df<1) | np.isnan(p_map)] = 1.0

# apply dose threshold
dose_mask = (mean_proj[0] >= DOSE_THRESH) | (mean_proj[1] >= DOSE_THRESH)

# significance masks
sig_mask = (p_map < ALPHA) & dose_mask
sig_pos  = sig_mask & (t_map > 0)
sig_neg  = sig_mask & (t_map < 0)

# confidence‑interval masks
ci_pos = (p_map < CI_ALPHA) & (t_map > 0) & dose_mask
ci_neg = (p_map < CI_ALPHA) & (t_map < 0) & dose_mask

# --------------------- 5. Summary of falloff_grad -------------------
print("Dose-falloff gradient by group (abs ΔD/Δr):")
for grp, label in enumerate(group_labels):
    vals = patient_stats.loc[patient_stats.group == grp, 'falloff_grad']
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"{label:20s} n={len(vals):2d}, mean={vals.mean():.3f}, "
          f"median={vals.median():.3f}, 95% CI=({lo:.3f},{hi:.3f})")

# r_g_max summary
results = []
for grp, label in enumerate(group_labels):
    vals = patient_stats.loc[patient_stats.group == grp, 'r_g_max'].values
    mean_val, med_val = vals.mean(), np.median(vals)
    ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
    results.append((label, len(vals), mean_val, med_val, ci_lo, ci_hi))
print(f"\n{'Group':<20s}{'N':>5s}{'Mean':>10s}{'Median':>10s}{'95% CI':>15s}")
for label, n, mean_val, med_val, lo, hi in results:
    print(f"{label:<20s}{n:5d}{mean_val:10.2f}{med_val:10.2f}({lo:.2f}, {hi:.2f})")

# --------------------- plotting helper ------------------------------
def draw_head_outline(ax):
    ax.contour(mean_head, levels=[boundary], colors='black', linewidths=1.2)

mean_props   = {'marker':'^','markerfacecolor':'orange','markeredgecolor':'black','markersize':10}
median_props = {'color':'green','linewidth':2}
mean_handle  = Line2D([], [], **mean_props, label='Mean')
median_handle= Line2D([], [], **median_props, label='Median')

# --------------------- 6. SAVE & SHOW PLOTS ----------------------------
# 1) Mean relative dose – SHORT
fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
im = ax.imshow(mean_proj[0], cmap='jet', vmin=0, vmax=1, origin='upper')
draw_head_outline(ax)
ax.set_title("Mean Relative Dose\nSHORT (<600 d)")
ax.axis('off')
fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label='Relative dose')
fig.savefig(f"{SAVE_PATH}/mean_rel_dose_SHORT.pdf", bbox_inches='tight')
plt.close(fig)

# 2) Mean relative dose – LONG
fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
im = ax.imshow(mean_proj[1], cmap='jet', vmin=0, vmax=1, origin='upper')
draw_head_outline(ax)
ax.set_title("Mean Relative Dose\nLONG (≥600 d)")
ax.axis('off')
fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label='Relative dose')
fig.savefig(f"{SAVE_PATH}/mean_rel_dose_LONG.pdf", bbox_inches='tight')
plt.close(fig)

# 3) Dose difference – LONG minus SHORT (pp)
diff = (mean_proj[1] - mean_proj[0]) * 100
diff[~head_mask] = np.nan
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(diff, cmap='seismic', vmin=-100, vmax=100, origin='upper')
draw_head_outline(ax)
ax.set_title("Dose Difference\nLONG – SHORT (pp)", pad=15)
ax.axis('off')
cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cax, label='Percentage points')
fig.savefig(f"{SAVE_PATH}/dose_diff_LONG_minus_SHORT.pdf", bbox_inches='tight')
plt.close(fig)

# 4) Significant voxels
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(mean_head, cmap='gray', origin='upper')
draw_head_outline(ax)
ax.imshow(np.where(sig_mask, 1, np.nan), cmap='Reds', alpha=0.6, origin='upper')
ax.set_title(f"Significant Voxels (p<{ALPHA}, dose≥{DOSE_THRESH})", pad=15)
ax.axis('off')
fig.savefig(f"{SAVE_PATH}/significance_map.pdf", bbox_inches='tight')
plt.close(fig)

# 5) Directional significance with CI dashed
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(mean_head, cmap='gray', origin='upper')
draw_head_outline(ax)
ys_p, xs_p = np.where(sig_pos)
ys_n, xs_n = np.where(sig_neg)
ax.scatter(xs_p, ys_p, c='red',  s=5, label='Higher in LONG')
ax.scatter(xs_n, ys_n, c='blue', s=5, label='Higher in SHORT')
ax.contour(ci_pos, levels=[0.5], colors='red', linestyles='--', linewidths=1.2)
ax.contour(ci_neg, levels=[0.5], colors='blue', linestyles='--', linewidths=1.2)
ax.set_title("Directional Significance\np<0.05; (p<0.10 dashed)", pad=15)
ax.axis('off')
ax.legend(loc='upper left', bbox_to_anchor=(1.02,1), scatterpoints=1, markerscale=5)
fig.savefig(f"{SAVE_PATH}/directional_significance.pdf", bbox_inches='tight')
plt.close(fig)

# 6) Max-dose locations per patient
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(mean_head, cmap='gray', origin='upper')
draw_head_outline(ax)
for grp, color in zip((0,1), ('blue','red')):
    pts = patient_stats[patient_stats.group == grp]
    ax.scatter(pts.xMAX, pts.yMAX, label=f"Max dose, {group_labels[grp]}", color=color, s=50, marker='X', alpha=0.7)
ax.set_xlabel('xMAX (px)'); ax.set_ylabel('yMAX (px)')
ax.set_title('Max‑Dose Location per Patient', pad=15)
ax.legend(loc='upper left', bbox_to_anchor=(1.02,1))
plt.close(fig)

# 7) Center of mass per patient
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(mean_head, cmap='gray', origin='upper')
draw_head_outline(ax)
for grp, color in zip((0,1), ('blue','red')):
    pts = patient_stats[patient_stats.group == grp]
    ax.scatter(pts.xCOM, pts.yCOM, label=f"COM, {group_labels[grp]}", color=color, s=40, edgecolors='white', alpha=0.7)
ax.set_xlabel('xCOM (px)'); ax.set_ylabel('yCOM (px)')
ax.set_title('Center of Mass per Patient', pad=15)
ax.legend(loc='upper left', bbox_to_anchor=(1.02,1))
plt.close(fig)

# 8) Spread around COM (radius of gyration)
fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
bp = ax.boxplot(
    [patient_stats.loc[patient_stats.group==g, 'r_g'] for g in (0,1)],
    labels=group_labels, showmeans=True,
    meanprops={'marker':'^','markerfacecolor':'orange'},
    medianprops={'color':'green','linewidth':2}
)
ax.set_ylabel('Radius of Gyration (px)')
ax.set_title('Spread Around COM', pad=10)
ax.legend(handles=[mean_handle, median_handle], loc='upper right', frameon=True)
plt.close(fig)

# 9) Spread around max-dose voxel (r_g_max)
fig, ax = plt.subplots(figsize=(6,6))
ax.boxplot(
    [patient_stats.loc[patient_stats.group==g, 'r_g_max'] for g in (0,1)],
    labels=group_labels, showmeans=True,
    meanprops={'marker':'^','markerfacecolor':'orange','markersize':10},
    medianprops={'color':'green','linewidth':2}
)
ax.set_ylabel('Radius of Gyration around Max Dose (px)')
ax.set_title('Spread Around Max Dose', pad=10)
ax.set_box_aspect(1)
ax.legend(handles=[mean_handle, median_handle], loc='upper left', bbox_to_anchor=(1.02,1))
plt.close(fig)

# 10) Dose-falloff gradient
fig, ax = plt.subplots(figsize=(6,6))
ax.boxplot(
    [patient_stats.loc[patient_stats.group==g, 'falloff_grad'] for g in (0,1)],
    labels=group_labels, showmeans=True,
    meanprops={'marker':'^','markerfacecolor':'orange','markersize':10},
    medianprops={'color':'green','linewidth':2}
)
ax.set_ylabel('Dose‑Falloff Gradient (abs ΔD/Δr)')
ax.set_title('Radial Dose‑Falloff Gradient', pad=10)
ax.set_box_aspect(1)
ax.legend(handles=[mean_handle, median_handle], loc='upper left', bbox_to_anchor=(1.02,1))
plt.close(fig)
# --------------------- 11. Between-group significance tests ---------------------
def cohens_d(x, y):
    """Unequal-size, unequal-var Cohen’s d (UVSD)."""
    nx, ny   = len(x), len(y)
    vx, vy   = x.var(ddof=1), y.var(ddof=1)
    s_pool   = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return (x.mean() - y.mean()) / s_pool

def rank_biserial(u_stat, n1, n2):
    """Convert Mann-Whitney U to rank-biserial r."""
    return 1 - 2*u_stat / (n1*n2)

def bootstrap_ci(data, fn=np.mean, n_boot=10_000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    boots = [fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

metrics = {
    'r_g'          : 'Radius of gyration (COM)',
    'r_g_max'      : 'Radius of gyration (max-dose voxel)',
    'falloff_grad' : 'Dose-falloff gradient'
}

print("\n===== Between-group significance tests =====")
for col, pretty in metrics.items():
    g0 = patient_stats.loc[patient_stats.group==0, col].values
    g1 = patient_stats.loc[patient_stats.group==1, col].values

    # 1. Normality
    p_norm0 = stats.shapiro(g0).pvalue if len(g0)>=3 else 0
    p_norm1 = stats.shapiro(g1).pvalue if len(g1)>=3 else 0
    normal  = (p_norm0>=0.05) and (p_norm1>=0.05)

    # 2. Equal variances (only meaningful if normal)
    p_lev   = stats.levene(g0, g1).pvalue
    equal_var = p_lev >= 0.05

    # 3. Choose test
    if normal:
        t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)  # Welch always safe
        eff   = cohens_d(g0, g1)
        eff_s = f"Cohen's d = {eff:+.2f}"
    else:
        u_stat, p_val = stats.mannwhitneyu(g0, g1, alternative='two-sided')
        eff   = rank_biserial(u_stat, len(g0), len(g1))
        eff_s = f"Rank-biserial r = {eff:+.2f}"

    # 4. Bootstrap CIs
    ci0 = bootstrap_ci(g0, fn=np.median if not normal else np.mean)
    ci1 = bootstrap_ci(g1, fn=np.median if not normal else np.mean)

    print(f"\n{pretty}:")
    print(f"  n₀={len(g0):2d}, n₁={len(g1):2d}")
    print(f"  Normality p-vals: SHORT={p_norm0:.3f}, LONG={p_norm1:.3f}")
    if normal:
        print(f"  Levene equal-var p={p_lev:.3f}")
    test_name = "Welch t-test" if normal else "Mann-Whitney U"
    print(f"  {test_name}: p = {p_val:.4f};  {eff_s}")
    print(f"  95 % CI SHORT: {ci0[0]:.3f} – {ci0[1]:.3f}")
    print(f"  95 % CI LONG : {ci1[0]:.3f} – {ci1[1]:.3f}")

print(f"All figures have been saved at {SAVE_PATH}")
