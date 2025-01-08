import numpy as np
import matplotlib.pyplot as plt
from funcs import *

MOUSE_NO = 3
SESS_NO = 1
DATA_FNAME = "/ceph/mrsic_flogel/public/projects/WiRe_20241218_DistributedImpulsivity/DMDM_NPX_curated_VidSes_AllAdded_cleaned_August2022_struct.mat"


# Get data
events, probes, video, behav = get_session_data(DATA_FNAME, MOUSE_NO, SESS_NO)

# Find is_FA, clusters, cluster labels, spike times, motion onset times
spike_times = np.array(probes['st'])
clusters = np.array(probes['clu'])
cluster_gs = probes['cluster_id_good_and_stable']
brain_regions = convert_regions_to_strings(probes['good_and_stab_cl_coord']['brain_region'])
brain_regions_comb = convert_regions_to_strings(probes['good_and_stab_cl_coord']['brain_region_comb'])

# Get trial start, finish and duration, regardless of trial performance
trial_start = events['Baseline_ON']['rise_t'].flatten()
baseline_off = events['Baseline_ON']['fall_t'].flatten()
change_off = events['Change_ON']['fall_t'].flatten()
trial_times = get_trial_times(trial_start, baseline_off, change_off)

# Match clusters to spike times
matched_spikes = match_clusters_and_spike_times(clusters, spike_times)

# Bin and smooth FRs
n_firing_rate_bins = np.max(spike_times)
matched_spikes_binned = bin_spike_times(matched_spikes, n_firing_rate_bins)
smoothed_spikes_binned = smooth_firing(matched_spikes_binned)

# Align FRs to trial start
aligned_spikes = match_unit_firing_to_trial_start(smoothed_spikes_binned, trial_times)

# Remove spikes after and 1s before motion onset
motion_onset_times = np.array(video['MotionOnsetTimes']).flatten()
motion_cleaned_spikes = remove_spikes_before_motion_onset(aligned_spikes, trial_start, motion_onset_times)

# Get FRs on early lick trials
trial_type_data = behav['trials_data_exp']
early_lick_frs = subset_trials(motion_cleaned_spikes, trial_type_data, brain_regions_comb, behav_key="IsFA", region=None)
hit_trials_frs = subset_trials(motion_cleaned_spikes, trial_type_data, brain_regions_comb, behav_key="IsHit", region=None)

# Get averages for each unit and trial type
FA = average_over_trials(early_lick_frs)
hit = average_over_trials(hit_trials_frs)
diff = hit - FA
mean_diff = np.nanmean(diff[:300])

# Make some plots
#plot_fa_vs_hit(diff, diff, MOUSE_NO, SESS_NO, brain_regions_comb, cluster_gs)

el_mos = subset_trials(motion_cleaned_spikes, trial_type_data, brain_regions_comb, behav_key="IsFA", region="MOs")
el_orb = subset_trials(motion_cleaned_spikes, trial_type_data, brain_regions_comb, behav_key="IsFA", region="ORB")
hit_mos = subset_trials(motion_cleaned_spikes, trial_type_data, brain_regions_comb, behav_key="IsHit", region="MOs")
hit_orb = subset_trials(motion_cleaned_spikes, trial_type_data, brain_regions_comb, behav_key="IsHit", region="ORB")
el_mos_avg = average_over_trials(el_mos)
el_orb_avg = average_over_trials(el_orb)
hit_mos_avg = average_over_trials(hit_mos)
hit_orb_avg = average_over_trials(hit_orb)
el_mos_avg, el_orb_avg, hit_mos_avg, hit_orb_avg = el_mos_avg[:, :300], el_orb_avg[:, :300], hit_mos_avg[:, :300], hit_orb_avg[:, :300]
mos_diff = np.nanmean(hit_mos_avg - el_mos_avg, axis=1)
orb_diff = np.nanmean(hit_orb_avg - el_orb_avg, axis=1)

