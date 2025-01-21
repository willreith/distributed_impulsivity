"""
Make some plots
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#%%
# Load data, metadata, baseline data
# ----------------------------------

# Load data
early_lick = np.load('data/earlyLick_trialStart_all_data.npy')
hit = np.load('data/hit_trialStart_all_data.npy')
miss = np.load('data/miss_trialStart_all_data.npy')

# Load metadata
early_lick_metadata = np.load('data/earlyLick_trialStart_all_metadata.npy', allow_pickle=True)
hit_metadata = np.load('data/hit_trialStart_all_metadata.npy', allow_pickle=True)
miss_metadata = np.load('data/miss_trialStart_all_metadata.npy', allow_pickle=True)

# Load baseline data
hit_baseline = np.load('data/baseline_mean_std_1sBeforeTrialStart-trialStart_hit_all.npy')
hit_means = hit_baseline[:, 0]
hit_stds = hit_baseline[:, 1]

# Define temporal window of interest
# Bins of 10ms; 0 is 3s before trial start; 300 is trial start
min_time = 0
max_time = 600

# Get min and max for soft-normalisation
early_lick_min, early_lick_max = np.min(early_lick, axis=1), np.max(early_lick, axis=1)
hit_min, hit_max = np.min(hit, axis=1), np.max(hit, axis=1)
miss_min, miss_max = np.min(miss, axis=1), np.max(miss, axis=1)


#%%
# Make and apply some normalisation functions
# -------------------------------------------

def softnorm(data_arr, window=(0, 600)):
    data_range =  np.max(data_arr[:, window[0]:window[1]], axis=1) - np.min(data_arr[:, window[0]:window[1]], axis=1)
    return data_arr[:, window[0]:window[1]] / (7 + data_range[:, None])

def centre_across_cells(data_arr):
    cell_mean = np.nanmean(data_arr, axis=0)
    return data_arr - cell_mean

def centre_across_time(data_arr):
    time_mean = np.nanmean(data_arr, axis=1)
    return data_arr - time_mean[:, None]

def centre_data(data_arr):
    cell_centred = centre_across_cells(data_arr)
    return centre_across_time(cell_centred)

def softnorm_centre(data_arr):
    return centre_data(softnorm(data_arr))

def mean_subtract(data_arr, mean_arr):
    return data_arr - mean_arr[:, None]

def std_divide(data_arr, std_arr):
    return data_arr / std_arr[:, None]

def zscore(data_arr, mean_std_arr):
    return std_divide(mean_subtract(data_arr, mean_std_arr[:, 0]), mean_std_arr[:, 1])

# Get mean-centred data
early_lick_meancen = mean_subtract(early_lick, hit_means)[:, min_time:max_time]
hit_meancen = mean_subtract(hit, hit_means)[:, min_time:max_time]
miss_meancen = mean_subtract(miss, hit_means)[:, min_time:max_time]

# Get z-scored data
early_lick_zscored = zscore(early_lick, hit_baseline)[:, min_time:max_time]
hit_zscored = zscore(hit, hit_baseline)[:, min_time:max_time]
miss_zscored = zscore(miss, hit_baseline)[:, min_time:max_time]

#%% 
# Get firing rate differences between conditions
# ----------------------------------------------

# Get differences in mean-centred firing rates
diff_early_lick_hit_meancen = early_lick_meancen - hit_meancen
diff_early_lick_miss_meancen = early_lick_meancen - miss_meancen
diff_hit_miss_meancen = hit_meancen - miss_meancen

# Get differences in z-scored firing rates
diff_early_lick_hit_zscored = early_lick_zscored - hit_zscored
diff_early_lick_miss_zscored = early_lick_zscored - miss_zscored
diff_hit_miss_zscored = hit_zscored - miss_zscored

#%%
# Define ROIs and get differences for each region
# -----------------------------------------------

ROIs = ['ACA', 'CB', 'CP', 'FRP', 'GPe', 'GPi',  'ILA', 'MD', 'MOp', 'MOs', 'MRN', 'ORB', 'PL', 'VISp']

# Get differences by region
regions_combined = hit_metadata[:,4]
fr_diff_by_region = {}
fr_diff_by_region['early_lick-hit'], fr_diff_by_region['early_lick-miss'], fr_diff_by_region['hit-miss'] = {}, {}, {}
for region in np.unique(regions_combined):
    reg_mask = regions_combined == region
    fr_diff_by_region['early_lick-hit'][region] = diff_early_lick_hit_zscored[reg_mask]
    fr_diff_by_region['early_lick-miss'][region] = diff_early_lick_miss_zscored[reg_mask]
    fr_diff_by_region['hit-miss'][region] = diff_hit_miss_zscored[reg_mask]

#%%
# Plot activity difference across conditions by region
# ----------------------------------------------------

diff_key = 'early_lick-hit'
for key, value in fr_diff_by_region[diff_key].items():
    if key in ROIs:
        mean_values = np.nanmean(value, axis=0)
        x = np.arange(value.shape[1])
        std_error = np.nanstd(value, axis=0, ddof=1) / np.sqrt(value.shape[0])

        plt.plot(x, mean_values)
        plt.fill_between(x, mean_values - std_error, mean_values + std_error, alpha=0.2)
        plt.axvline(300, c='k', ls='--')
        plt.axhline(0, c='k', ls='--')
        plt.title(f'{key}, n={value.shape[0]}')
        plt.xticks(range(0,601,100), range(-3,4))
        plt.ylim((-0.4, 0.4))
        plt.show()
        
#%%
# Get firing rate differences by MOs subregions
# ---------------------------------------------

diff_key = 'early_lick-hit'
region = 'MOs'
MOs_diff = fr_diff_by_region[diff_key][region]
MOs_ap_position = hit_metadata[hit_metadata[:,4]==region, 6].astype(np.float64)
MOs_diff = MOs_diff[MOs_ap_position > 0.5]
MOs_ap_position = MOs_ap_position[MOs_ap_position > 0.5]

atMOs_mask = MOs_ap_position >= 2.5
cMOs_mask = (1.5 <= MOs_ap_position) & (MOs_ap_position < 2.5)
pMOs_mask = (0.5 <= MOs_ap_position) & (MOs_ap_position < 1.55)

atMOs, cMOs, pMOs = MOs_diff[atMOs_mask], MOs_diff[cMOs_mask], MOs_diff[pMOs_mask]
atMOs_mean, cMOs_mean, pMOs_mean = np.nanmean(atMOs, axis=0), np.nanmean(cMOs, axis=0), np.nanmean(pMOs, axis=0)
atMOs_std = np.nanstd(atMOs, axis=0, ddof=1) / np.sqrt(atMOs.shape[0])
cMOs_std = np.nanstd(cMOs, axis=0, ddof=1) / np.sqrt(cMOs.shape[0])
pMOs_std = np.nanstd(pMOs, axis=0, ddof=1) / np.sqrt(pMOs.shape[0])

#%%
# Plot firing rate differences by MOs subregions
# ----------------------------------------------
x = np.arange(min_time, max_time)
plt.plot(range(min_time, max_time), atMOs_mean, label=f'atMOs, n={sum(atMOs_mask)}')
plt.plot(range(min_time, max_time), cMOs_mean, label=f'cMOs, n={sum(cMOs_mask)}')
plt.plot(range(min_time, max_time), pMOs_mean, label=f'pMOs, n={sum(pMOs_mask)}')
plt.fill_between(x, atMOs_mean - atMOs_std, atMOs_mean + atMOs_std, alpha=0.2)
plt.fill_between(x, cMOs_mean - cMOs_std, cMOs_mean + cMOs_std, alpha=0.2)
plt.fill_between(x, pMOs_mean - pMOs_std, pMOs_mean + pMOs_std, alpha=0.2)
plt.axvline(300, c='k', ls='--')
plt.axhline(0, c='k', ls='--')
plt.xticks(range(min_time,max_time+1,100), range(-3,4))
plt.ylim((-0.15, 0.15))
plt.xlabel("Time from trial start (s)", fontsize=14)
plt.ylabel('$r_{EL}$-$r_{hit}$', fontsize=14)
plt.title('Early lick vs. hit trials', fontsize=18)
plt.legend()
plt.show()

#%%
# Plot firing rate differences as a function of AP position
# ---------------------------------------------------------

# Remove rows with nan and get mean firing rate difference for each cell
nan_mask = ~np.isnan(MOs_diff).any(axis=1)
MOs_ap_position_clean = MOs_ap_position[nan_mask]
MOs_mean_diff_by_cell = np.nanmean(MOs_diff, axis=1)
MOs_mean_diff_by_cell = MOs_mean_diff_by_cell[nan_mask]

# Plot mean firing difference for each cell across the AP axis across whole time window
corr, p_val = sp.stats.pearsonr(MOs_ap_position_clean, MOs_mean_diff_by_cell)
p_str = f'p={p_val:.3f}' if p_val > 0.001 else "p < 0.001"
x, y = MOs_ap_position_clean, MOs_mean_diff_by_cell
xseq = np.linspace(np.min(x), np.max(x), 100)
b, a = np.polyfit(x, y, deg=1)
fig = plt.figure(figsize=(4,5))
plt.plot(xseq, a + b * xseq, color="k", lw=2.5)
plt.title(f'Difference in early lick and hit firing rates\nby AP position, trial start ± 3s', fontsize=16)
plt.xlabel("AP distance from bregma (mm)", fontsize=16)
plt.ylabel('$r_{EL}$-$r_{hit}$', fontsize=16)
plt.scatter(MOs_ap_position_clean, MOs_mean_diff_by_cell, s=3)
plt.text(np.min(MOs_ap_position_clean), 
         np.max(MOs_mean_diff_by_cell), 
         f'r: {corr:.3f}\n{p_str}',
         horizontalalignment='left',
         verticalalignment='top',
         fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

#%%
# Plot firing rate differences as a function of AP position for shorter time windows
# ----------------------------------------------------------------------------------

# Bin sizes where 1 bin = 10 ms
bin_size = 50
n_plots = int(max_time/bin_size)
fig, axs = plt.subplots(1, n_plots, figsize=(20, 4), dpi=200, sharey=True, sharex=True, layout='constrained')
fig.suptitle('Difference in early lick and hit firing rates by AP position and time', fontsize=22)
fig.supxlabel('Distance from bregma (mm)', fontsize=22)
fig.supylabel('$r_{EL}$-$r_{hit}$', fontsize=22)

for ax_idx, bin_idx in enumerate(range(min_time, max_time, bin_size)):
    MOs_mean_diff_by_cell_binned = np.nanmean(MOs_diff[:, bin_idx:bin_idx+100], axis=1)
    MOs_mean_diff_by_cell_binned = MOs_mean_diff_by_cell_binned[nan_mask]
    corr, p_val = sp.stats.pearsonr(MOs_ap_position_clean, MOs_mean_diff_by_cell_binned)
    p_str = f'p={p_val:.3f}' if p_val > 0.001 else "p < 0.001"
    x, y = MOs_ap_position_clean, MOs_mean_diff_by_cell_binned
    xseq = np.linspace(np.min(x), np.max(x), 100)
    b, a = np.polyfit(x, y, deg=1)
    ax = axs[ax_idx]
    ax.scatter(MOs_ap_position_clean, MOs_mean_diff_by_cell_binned, s=1)
    ax.plot(xseq, a + b * xseq, color="k", lw=2.5)
    ax.set_title(f'{bin_idx/100 - 3}-\n{bin_idx/100 - 2.5}s', fontsize=16)
    ax.text(np.min(MOs_ap_position_clean), 
         np.max(MOs_mean_diff_by_cell), 
         f'r = {corr:.3f}\n{p_str}',
         horizontalalignment='left',
         verticalalignment='center_baseline',
         fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=16)

    #ax.title(f'MOs, time from trial start: {bin_idx/100 - 3}-{bin_idx/100 - 2.5}s;  \n Pearson\'s R: {corr:.3f}, {p_str}', fontsize=16)
    #ax.xlabel("AP distance from bregma (mm)", fontsize=14)
    #ax.ylabel('$\Delta$((z-scored FR|early lick)\n- (z-scored FR| hit))', fontsize=14)
    #ax.show()


# Get binned mean differences, bins of 1s
# %%
