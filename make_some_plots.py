import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

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


# Define temporal window of interest
# Bins of 10ms; 0 is 3s before trial start; 300 is trial start
min_time = 0
max_time = 600

early_lick_min, early_lick_max = np.min(early_lick, axis=1), np.max(early_lick, axis=1)
hit_min, hit_max = np.min(hit, axis=1), np.max(hit, axis=1)
miss_min, miss_max = np.min(miss, axis=1), np.max(miss, axis=1)

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
    return data_arr - mean_arr

def std_divide(data_arr, std_arr):
    return data_arr / std_arr

def zscore(data_arr, mean_std_arr):
    return std_divide(mean_subtract(data_arr, mean_std_arr[:, 0]), mean_std_arr[:, 1])

# Get softnorm + centred data
early_lick_snc = softnorm(early_lick)
hit_snc = softnorm(hit)
miss_snc = softnorm(miss)



# Get differences in firing rates 
diff_early_lick_hit = early_lick[:, min_time:max_time] - hit[:, min_time:max_time]
diff_early_lick_miss = early_lick[:, min_time:max_time] - miss[:, min_time:max_time]
diff_hit_miss = hit[:, min_time:max_time] - miss[:, min_time:max_time]

# Normalise by sum of firing rates across conditions
diff_early_lick_hit_norm = diff_early_lick_hit / (early_lick[:, min_time:max_time] + hit[:, min_time:max_time])
diff_early_lick_miss_norm = diff_early_lick_miss / (early_lick[:, min_time:max_time] + miss[:, min_time:max_time])
diff_hit_miss_norm = diff_hit_miss / (hit[:, min_time:max_time] + miss[:, min_time:max_time])

# Get difference in softnorm/centred neurons FRs
diff_early_lick_hit = early_lick_snc - hit_snc
diff_early_lick_miss = early_lick_snc - miss_snc
diff_hit_miss = hit_snc - miss_snc


ROIs = ['ACA', 'CB', 'CP', 'FRP', 'GPe', 'GPi',  'ILA', 'MD', 'MOp', 'MOs', 'MRN', 'ORB', 'VISp']

# Get differences by region
regions_combined = hit_metadata[:,4]
fr_diff_by_region = {}
fr_diff_by_region['early_lick-hit'], fr_diff_by_region['early_lick-miss'], fr_diff_by_region['hit-miss'] = {}, {}, {}
for region in np.unique(regions_combined):
    reg_mask = regions_combined == region
    fr_diff_by_region['early_lick-hit'][region] = diff_early_lick_hit[reg_mask]
    fr_diff_by_region['early_lick-miss'][region] = diff_early_lick_miss[reg_mask]
    fr_diff_by_region['hit-miss'][region] = diff_hit_miss[reg_mask]


plot_all = 0
if plot_all:
    for key, value in fr_diff_by_region['early_lick-hit'].items():
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
        

# Get differences by MOs subregions
MOs_diff = fr_diff_by_region['early_lick-hit']['MOs']
#MOs_diff = fr_diff_by_region['early_lick-miss']['MOs']
#MOs_diff = fr_diff_by_region['hit-miss']['MOs']
MOs_ap_position = hit_metadata[hit_metadata[:,4]=='MOs', 6].astype(np.float64)
MOs_diff = MOs_diff[MOs_ap_position > 0.5]
MOs_ap_position = MOs_ap_position[MOs_ap_position > 0.5]

atMOs_mask = MOs_ap_position >= 2.5
cMOs_mask = (1.25 <= MOs_ap_position) & (MOs_ap_position < 2.5)
pMOs_mask = (0.5 <= MOs_ap_position) & (MOs_ap_position < 1.25)

atMOs, cMOs, pMOs = MOs_diff[atMOs_mask], MOs_diff[cMOs_mask], MOs_diff[pMOs_mask]
atMOs_mean, cMOs_mean, pMOs_mean = np.nanmean(atMOs, axis=0), np.nanmean(cMOs, axis=0), np.nanmean(pMOs, axis=0)
atMOs_std = np.nanstd(atMOs, axis=0, ddof=1) / np.sqrt(atMOs.shape[0])
cMOs_std = np.nanstd(cMOs, axis=0, ddof=1) / np.sqrt(cMOs.shape[0])
pMOs_std = np.nanstd(pMOs, axis=0, ddof=1) / np.sqrt(pMOs.shape[0])

x = np.arange(600)
plt.plot(range(min_time, max_time), atMOs_mean, label=f'atMOs, n={sum(atMOs_mask)}')
plt.plot(range(min_time, max_time), cMOs_mean, label=f'cMOs, n={sum(cMOs_mask)}')
plt.plot(range(min_time, max_time), pMOs_mean, label=f'pMOs, n={sum(pMOs_mask)}')
plt.fill_between(x, atMOs_mean - atMOs_std, atMOs_mean + atMOs_std, alpha=0.2)
plt.fill_between(x, cMOs_mean - cMOs_std, cMOs_mean + cMOs_std, alpha=0.2)
plt.fill_between(x, pMOs_mean - pMOs_std, pMOs_mean + pMOs_std, alpha=0.2)
plt.axvline(300, c='k', ls='--')
plt.axhline(0, c='k', ls='--')
plt.xticks(range(0,601,100), range(-3,4))
plt.ylim((-0.4, 0.4))
plt.legend()
plt.show()

MOs_mean_diff_by_cell = np.nanmean(MOs_diff, axis=1)
corr, p_val = sp.stats.pearsonr(MOs_ap_position, MOs_mean_diff_by_cell)
p_str = f'p={p_val:.3f}' if p_val > 0.001 else "p< 0.001"
x, y = MOs_ap_position, MOs_mean_diff_by_cell
xseq = np.linspace(np.min(x), np.max(x), 100)
b, a = np.polyfit(x, y, deg=1)
plt.plot(xseq, a + b * xseq, color="k", lw=2.5)
plt.title(f'MOs, trial start ± 3s; Pearson\'s R: {corr:.3f}, {p_str}', fontsize=18)
plt.xlabel("AP distance from bregma (mm)", fontsize=14)
plt.ylabel('$\Delta$((Softnorm FR|early lick)\n- (Softnorm FR| hit))', fontsize=14)
plt.scatter(MOs_ap_position, MOs_mean_diff_by_cell, s=3)
plt.show()

for i in range(0,600,50):
    MOs_mean_diff_by_cell_binned = np.nanmean(MOs_diff[:, i:i+100], axis=1)
    nan_mask = ~np.isnan(MOs_mean_diff_by_cell_binned)
    corr, p_val = sp.stats.spearmanr(MOs_ap_position[nan_mask], MOs_mean_diff_by_cell_binned[nan_mask])
    p_str = f'p={p_val:.3f}' if p_val > 0.001 else "p< 0.001"
    x, y = MOs_ap_position, MOs_mean_diff_by_cell_binned
    xseq = np.linspace(np.min(x), np.max(x), 100)
    b, a = np.polyfit(x, y, deg=1)
    plt.figure(figsize=(4,3))
    plt.scatter(MOs_ap_position, MOs_mean_diff_by_cell_binned, s=3)
    plt.plot(xseq, a + b * xseq, color="k", lw=2.5)
    plt.title(f'MOs, time from trial start: {i/100 - 3}-{i/100 - 2.5}s;  \n Pearson\'s R: {corr:.3f}, {p_str}', fontsize=16)
    plt.xlabel("AP distance from bregma (mm)", fontsize=14)
    plt.ylabel('$\Delta$((Softnorm FR|early lick)\n- (Softnorm FR| hit))', fontsize=14)
    plt.show()


# Get binned mean differences, bins of 1s