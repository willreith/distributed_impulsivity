import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy

"""
Event keys:  ['Air_puff', 'Baseline_ON', 'Change_ON', 'Eye_cam', 'Front_cam', 'Laser_ON', 
              'Lick_L', 'Lick_R', 'Masking_ON', 'Rot_enc_A', 'Rot_enc_B', 'RunningSpeed',
              'Synch', 'Top_cam', 'Valve_L', 'Valve_R', 'frame_times_tr', 'session_name']
              
Probe keys:  ['BestWaveChannelIdxValid', 'BestWaveChannelRaw', 'NoiseIds', 'WaveForms',
               'clu', 'cluster_goodlabels', 'cluster_id_good_and_stable', 'dat_path',
               'dtype', 'good_and_stab_cl_coord', 'hp_filtered', 'n_channels_dat',
               'offset', 'pcFeat', 'pcFeatInd', 'probe_coord', 'sample_rate', 'st',
               'xcoords', 'ycoords']
               
Video keys:  ['MEthresh', 'MotionOnsetTimes', 'motionEnergy', 'pupilArea']

Behaviour keys:  ['ComputerSettings', 'SessionSettings', 'trials_data_exp']
"""

"""Firing rates were calculated as spike counts averaged in 10 ms bins and smoothened by convolution with two-sided Gaussian with 30 ms s.d. (Khilkevich, 2024)"""


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


def get_session_data(data_fname, mouse_no, sess_no):
    with h5py.File(data_fname, 'r') as f:
        # 1-indexed, if #0 is '#refs#'
        sess_data = f[mouse_no][sess_no]

        # Convert each relevant HDF5 group to a Python object
        events_data = load_h5_object(sess_data['NI_events'], f)
        probes_data = load_h5_object(sess_data['NPX_probes'], f)
        video_data  = load_h5_object(sess_data['Video'], f)
        behav_data  = load_h5_object(sess_data['behav_data'], f)

        # Return them as plain Python objects (dicts + arrays)
        return events_data, probes_data, video_data, behav_data
    

def get_probes_metadata_only(data_fname, mouse_no, sess_no):
    """
    Returns cluster IDs and brain region info for all probes in a session.
    Handles multiple probes by returning a list of cluster IDs for each probe.
    """
    with h5py.File(data_fname, 'r') as f:
        sess_data = f[mouse_no][sess_no]
        
        # Access NPX_probes group
        probes_grp = sess_data['NPX_probes']
        n_probes = len(probes_grp['offset'][()])
        coord_grp = probes_grp['good_and_stab_cl_coord']
        cluster_coords = load_h5_object(coord_grp, f)
        
        # Initialize storage for multiple probes
        if n_probes == 1:
            cluster_ids = probes_grp['cluster_id_good_and_stable'][()]
        else:
            cluster_ids = load_h5_object(probes_grp['cluster_id_good_and_stable'], f)

        return cluster_ids, cluster_coords, n_probes



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


def match_unit_firing_to_trial_start(smoothed_spikes, trial_times):
    """

    args:
        smoothed_spikes (dict): dictionary with units as keys and 
            smoothed firing rates in bins of 10 ms as values
        trial_times (dict): dictionary with trial start, duration
            and end times in seconds
    
    returns:
        trial_aligned_spikes (dict): {unit: {trial: np.array(spikes)}}        
    """
    n_trials = len(trial_times['duration'])
    max_duration = np.max(trial_times['duration']) + 6 # Get max duration + 6 seconds either side
    max_duration_bins = int(np.round(100 * max_duration)) # Convert max duration into bins of 10 ms
    print(max_duration_bins)
    # Cap trial lengths at 25s to account for aberrant durations
    if max_duration_bins > 2500:
        max_duration_bins = 2500
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
            if pad_length < 0:
                trial_aligned_spikes[unit][trial,:] = spike_array[:max_duration_bins]
            else:
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
    
    if isinstance(behav_key, str):
        assert behav_key in valid_trial_types, f'Invalid trial type. Trial type must be one of {valid_trial_types}'
        trial_filter = np.array(trial_type_data[behav_key]).flatten()  # shape (n_trials,)
        trial_filter = trial_filter.astype(bool)
    
    elif isinstance(behav_key, list) or isinstance(behav_key, tuple):
        assert all([item in valid_trial_types for item in behav_key]), f'Invalid trial type. Trial type must be one of {valid_trial_types}'
        n_trials, n_keys = np.array(trial_type_data[behav_key[0]]).flatten(), len(behav_key)
        
        # Make empty array to contain the filter for each trial type and fill with trial filters
        trial_filter_arr = np.empty((n_trials, n_keys)) 
        for idx, key in enumerate(behav_key):
            trial_filter_arr[:, idx] = np.array(trial_type_data[key]).flatten()
        trial_filter_arr = trial_filter_arr.astype(bool)
        
        # Final trial filter is of shape (n_trials,) where True if all individual filters are True
        trial_filter = np.sum(trial_filter_arr) == n_keys
        print(trial_filter.shape, n_trials, n_keys)

    subset_spikes = {}

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


def plot_fa_vs_hit(FA, hit, mouse_no, sess_no, brain_regions, unit_ids):
    for i in range(FA.shape[0]):
        plt.figure(figsize=(4,2))
        FA_to_plot = FA[i,:600]
        hit_to_plot = hit[i,:600]
        plt.plot(np.arange(-3, 3, 0.01), FA_to_plot, label="Early lick")
        plt.plot(np.arange(-3, 3, 0.01), hit_to_plot, label="Hit")
        plt.title(f'{brain_regions[i]}, M: {mouse_no}, S: {sess_no}, U: {unit_ids[0][i]}')
        plt.vlines(0, min(np.concatenate((FA_to_plot, hit_to_plot))), max(np.concatenate((FA_to_plot, hit_to_plot))), 'k', '--')
        plt.legend()



if __name__ == "__main__":

    MOUSE_NO = 1 # 1-indexed
    SESS_NO = 1 # 1-indexed for consistency

    # Get data
    events, probes, video, behav = get_session_data(DATA_FNAME, MOUSE_NO, SESS_NO)

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
    plot_fa_vs_hit(FA, hit, MOUSE_NO, SESS_NO, brain_regions_comb, cluster_gs)


# Remove spikes before motion onset
# Average over FA and Hit trials
# Plot average +/- 3s for each cell by brain region

# Mouse, session, unit, time-lock, trial type, region
