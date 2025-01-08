import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy

"""
Event keys:  ['Air_puff', 'Baseline_ON', 'Change_ON', 'Eye_cam', 'Front_cam', 'Laser_ON', 
              'Lick_L', 'Lick_R', 'Masking_ON', 'Rot_enc_A', 'Rot_enc_B', 'RunningSpeed',
              'Synch', 'Top_cam', 'Valve_L', 'Valve_R', 'frame_times_tr', 'session_name']
              
Probes keys:  ['BestWaveChannelIdxValid', 'BestWaveChannelRaw', 'NoiseIds', 'WaveForms',
               'clu', 'cluster_goodlabels', 'cluster_id_good_and_stable', 'dat_path',
               'dtype', 'good_and_stab_cl_coord', 'hp_filtered', 'n_channels_dat',
               'offset', 'pcFeat', 'pcFeatInd', 'probe_coord', 'sample_rate', 'st',
               'xcoords', 'ycoords']
               
Video keys:  ['MEthresh', 'MotionOnsetTimes', 'motionEnergy', 'pupilArea']

Behaviour keys:  ['ComputerSettings', 'SessionSettings', 'trials_data_exp']
"""

"""Firing rates were calculated as spike counts averaged in 10 ms bins and smoothened by convolution with two-sided Gaussian with 30 ms s.d. (Khilkevich, 2024)"""

MOUSE_NO = 2 # 1-indexed
SESS_NO = 1 # 1-indexed for consistency
DATA_FNAME = "/ceph/mrsic_flogel/public/projects/WiRe_20241218_DistributedImpulsivity/DMDM_NPX_curated_VidSes_AllAdded_cleaned_August2022_struct.mat"


def load_h5_object(obj, file_handle):
    """
    Recursively convert an HDF5 group/dataset/reference into Python objects
    (dicts for groups, NumPy arrays for numeric datasets, lists or nested structures for references).
    """
    if isinstance(obj, h5py.Group):
        # If it's a group, recurse into each key
        out_dict = {}
        for key in obj.keys():
            out_dict[key] = load_h5_object(obj[key], file_handle)
        return out_dict

    elif isinstance(obj, h5py.Dataset):
        # If it's a dataset, read it
        data = obj[()]

        # If the dataset is an array of references, resolve each reference
        if hasattr(data, 'dtype') and (data.dtype == h5py.ref_dtype or data.dtype == object):
            # Return a list or nested structure, depending on shape
            refs = np.array(data, dtype=object, copy=False)  # interpret as object array
            return _resolve_reference_array(refs, file_handle)

        else:
            # It's a normal numeric (or string) dataset, return as NumPy array (or scalar)
            return data

    else:
        raise TypeError(f"Unsupported HDF5 object type: {type(obj)}")


def _resolve_reference_array(ref_array, file_handle):
    """
    Helper to resolve a NumPy array of object references into a Python list
    (or nested lists if ref_array has >1D).
    """
    # If it's scalar (shape == ()), just a single reference
    if ref_array.shape == ():
        single_ref = ref_array.item()  # get the single reference
        return load_h5_object(file_handle[single_ref], file_handle)

    # Otherwise, iterate over each element. We'll preserve the shape by creating
    # a list-of-lists (or deeper) structure that matches ref_array.shape.
    resolved = np.empty(ref_array.shape, dtype=object)  # store python objects
    it = np.nditer(ref_array, flags=["multi_index", "refs_ok"])
    for ref in it:
        idx = it.multi_index
        resolved[idx] = load_h5_object(file_handle[ref.item()], file_handle)
    return resolved.tolist()  # convert to a nested list structure


def get_session_data(data_fname=DATA_FNAME, mouse_no=MOUSE_NO, sess_no=SESS_NO):
    with h5py.File(data_fname, 'r') as f:
        all_mouse_ids = list(f.keys())
        mouse_id = all_mouse_ids[mouse_no]  # 1-indexed, if #0 is '#refs#'

        mouse_data = f[mouse_id]
        all_sess_ids = list(mouse_data.keys())
        sess_id = all_sess_ids[sess_no - 1]  # 1-indexed for consistency
        sess_data = mouse_data[sess_id]

        # Convert each relevant HDF5 group to a Python object
        events_data = load_h5_object(sess_data['NI_events'], f)
        probes_data = load_h5_object(sess_data['NPX_probes'], f)
        video_data  = load_h5_object(sess_data['Video'], f)
        behav_data  = load_h5_object(sess_data['behav_data'], f)

        # Return them as plain Python objects (dicts + arrays)
        return events_data, probes_data, video_data, behav_data


def convert_regions_to_strings(utf8_dict):
    """Convert UTF-8 encoded region labels to strings"""
    brain_regions = []
    for lst in utf8_dict:
        tmp_str = ''
        for region in lst:
            for letter in region:
                letter = letter[0]
                tmp_str = tmp_str + chr(letter)
            brain_regions.append(tmp_str)
    return brain_regions


def match_clusters_and_spike_times(cluster_indices, spike_times):
    unique_clusters = np.unique(cluster_indices)
    return {
        cluster: spike_times[cluster_indices == cluster]
        for cluster in unique_clusters
    }


def get_trial_times(trial_start, baseline_off, change_off):
    """Get trial start and finish, regardless of trial performance."""
    rise_t = trial_start
    change_off[np.isnan(change_off)] = 0
    baseline_off_and_change_off = np.array([baseline_off, change_off])
    fall_t = np.max(baseline_off_and_change_off, axis=0)
    duration = fall_t - rise_t

    return {'duration': duration, 'rise_t': rise_t, 'fall_t': fall_t}


def bin_spike_times(matched_spikes, session_duration, bin_size_ms=10):
    session_duration *= 1000  # Convert to ms
    n_bins = int(session_duration) // bin_size_ms + 20 # Get number of bins and add (empty) bins at end for ease of convolution
    bins = np.arange(0, n_bins * bin_size_ms, bin_size_ms)
    return {
        unit: np.histogram(spikes * 1000, bins=bins)[0] * 100 # Convert to ms, get spikes per bin and convert to spikes per second (for bins of 10 ms)
        for unit, spikes in matched_spikes.items()
    }

def smooth_firing(matched_spikes_binned, bin_size_ms=10, sigma_ms=30):
    """
    Convolves each neuron's binned spike train with a 1D Gaussian kernel
    with standard deviation sigma_ms. The bin size is bin_size_ms.
    """
    # Convert the desired Gaussian sigma from ms to # of bins
    sigma_bins = sigma_ms / bin_size_ms  
    
    # Decide the width of the kernel (e.g., +/- 5 sigma)
    kernel_radius = int(np.ceil(5 * sigma_bins))
    x = np.arange(-kernel_radius, kernel_radius + 1)
    
    # Construct the Gaussian kernel (L2 normalized to integrate ~1.0)
    kernel = np.exp(-0.5 * (x / sigma_bins)**2)
    kernel = kernel / np.sum(kernel)  # Normalize so total area = 1

    smoothed_spikes_binned = {}
    for unit, binned_spikes in matched_spikes_binned.items():
        # Convolution with 'same' mode to keep the same length
        smoothed_spikes = np.convolve(binned_spikes, kernel, mode='same')
        smoothed_spikes_binned[unit] = smoothed_spikes
    
    return smoothed_spikes_binned


def match_unit_firing_to_trial_start_old(smoothed_spikes, trial_times):
    """

    args:
        smoothed_spikes (dict): dictionary with units as keys and 
            smoothed firing rates in bins of 10 ms as values
        trial_times (dict): dictionary with trial start, duration
            and end times in seconds
    
    returns:
        dict: {unit: {trial: np.array(spikes)}}        
    """
    n_trials = len(trial_times['duration'])
    trial_aligned_spikes = {}
    for unit, spikes in smoothed_spikes.items():
        trial_aligned_spikes[unit] = []
        for trial in range(n_trials):
            window_start = trial_times['rise_t'][trial] - 3 # Start window 3s before trial start
            window_end = trial_times['fall_t'][trial] + 3 # End window 3s after trial end
            start_idx = int(np.round(100 * window_start)) # Get index in 10 ms binned spiking
            end_idx = int(np.round(100 * window_end)) # -||-
            trial_aligned_spikes[unit].append(spikes[start_idx:end_idx])

    return trial_aligned_spikes

def match_unit_firing_to_trial_start(smoothed_spikes, trial_times):
    """

    args:
        smoothed_spikes (dict): dictionary with units as keys and 
            smoothed firing rates in bins of 10 ms as values
        trial_times (dict): dictionary with trial start, duration
            and end times in seconds
    
    returns:
        dict: {unit: {trial: np.array(spikes)}}        
    """
    n_trials = len(trial_times['duration'])
    max_duration = np.max(trial_times['duration']) + 6 # Get max duration + 6 seconds either side
    max_duration_bins = int(np.round(100 * max_duration)) # Convert max duration into bins of 10 ms
    trial_aligned_spikes = {}
    for unit, spikes in smoothed_spikes.items():
        trial_aligned_spikes[unit] = np.empty((n_trials, max_duration_bins))
        for trial in range(n_trials):
            window_start = trial_times['rise_t'][trial] - 3 # Start window 3s before trial start
            window_end = trial_times['fall_t'][trial] + 3 # End window 3s after trial end
            start_idx = int(np.round(100 * window_start)) # Get index in 10 ms binned spiking
            end_idx = int(np.round(100 * window_end)) # -||-
            spike_array = np.array(spikes[start_idx:end_idx])
            pad_length = max_duration_bins - len(spike_array)
            trial_aligned_spikes[unit][trial,:] = np.pad(spike_array, (0, pad_length), constant_values=np.nan)

    return trial_aligned_spikes

def remove_spikes_before_motion_onset(aligned_spikes, trial_start, motion_onset_times, window=1):
    motion_onset_in_trial = motion_onset_times - trial_start # Get motion onset relative to trial start
    motion_onset_in_trial = np.round(motion_onset_in_trial * 100) # Convert to scale compatible with 10 ms spike bins
    motion_cleaned_spikes = copy.deepcopy(aligned_spikes)
    for unit, trial_matrix in aligned_spikes.items():
        # trial_matrix.shape == (n_trials, max_duration_bins)
        for trial_idx in range(trial_matrix.shape[0]):
            mo = motion_onset_in_trial[trial_idx]
            if not np.isnan(mo):
                # Add 300 bins to account for 3 s prior to trial start
                mo = int(mo + 300)
                # Remove spikes from `mo - window_in_bins` to the end of the row
                mo_start = max(mo - 100 * window, 0)
                motion_cleaned_spikes[unit][trial_idx, mo_start:] = np.nan
    return motion_cleaned_spikes

def subset_trials(aligned_spikes, trial_type_data, indexed_brain_regions, behav_key="IsFA", region=None):
    """
    Subset trials by trial type (e.g. "IsFA") and optionally restrict to a given brain region.
    """
    valid_trial_types = [
        'IsAbort', 'IsAbortWithFA', 'IsEarlyBlock', 'IsFA',
        'IsHit', 'IsLateBlock', 'IsMiss', 'IsProbe'
    ]
    
    if behav_key not in valid_trial_types:
        raise KeyError(f"Not a valid trial type: {behav_key}")

    subset_spikes = {}
    trial_filter = np.array(trial_type_data[behav_key]).flatten()  # shape (n_trials,)
    trial_filter = trial_filter.astype(bool)

    for unit_idx, unit in enumerate(aligned_spikes.keys()):
        trial_matrix = aligned_spikes[unit]  # All data for unit; shape == (n_trials, max_duration_bins)
        filtered_matrix = trial_matrix[trial_filter, :] # Mask trials of interest
        
        if region is not None:
            if indexed_brain_regions[unit_idx] == region:
                subset_spikes[unit] = filtered_matrix
        else:
            subset_spikes[unit] = filtered_matrix

    return subset_spikes


def average_over_trials(spiking_data):
    mean_frs = {unit: np.nanmean(spiking_data[unit], axis=0) for unit in spiking_data.keys()}
    return np.array([mean_frs[unit] for unit in mean_frs.keys()])


def plot_fa_vs_hit(FA, hit):
    for i in range(FA.shape[0]):
        plt.figure(figsize=(4,2))
        FA_to_plot = FA[i,:600]
        hit_to_plot = hit[i,:600]
        plt.plot(np.arange(0,6000,10)/1000, FA_to_plot, label="Early lick")
        plt.plot(np.arange(0,6000,10)/1000, hit_to_plot, label="Hit")
        plt.title(f'{brain_regions_comb[i]}, M: {MOUSE_NO}, S: {SESS_NO}, U: {cluster_gs[0][i]}')
        plt.vlines(3, min(np.concatenate((FA_to_plot, hit_to_plot))), max(np.concatenate((FA_to_plot, hit_to_plot))), 'k', '--')
        plt.legend()


if __name__ == "__main__":
    # Get data
    events, probes, video, behav = get_session_data()

    # Find is_FA, clusters, cluster labels, spike times, motion onset times
    #is_FA = np.array(behav['trials_data_exp']['IsFA']).flatten() # Get IsFA and convert to one (n_trials,) array
    trial_type_data = behav['trials_data_exp']
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

    # Remove spikes before motion onset

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

    # Make some plots
    plot_fa_vs_hit(FA, hit)


# Remove spikes before motion onset
# Average over FA and Hit trials
# Plot average +/- 3s for each cell by brain region

# Mouse, session, unit, time-lock, trial type, region
