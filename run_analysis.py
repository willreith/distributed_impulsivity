import numpy as np
import matplotlib.pyplot as plt
from funcs import *


DATA_FNAME = "/ceph/mrsic_flogel/public/projects/WiRe_20241218_DistributedImpulsivity/DMDM_NPX_curated_VidSes_AllAdded_cleaned_August2022_struct.mat"

mice = [
        'x1074894',
        'x1078733',
        'x1103204',
        'x1108393',
        'x1116760',
        'x1116763',
        'x1116764',
        'x1116765',
        'x1117910',
        'x1117915',
        'x1119408',
        'x1119409',
        'x1119541',
        'x1119544',
        'x1120567',
        ]

sessions = {'x1074894': ['ML_1074894_M2_S01'], 
               'x1078733': ['ML_1078733_S01', 'ML_1078733_S02', 'ML_1078733_S03', 'ML_1078733_S04', 'ML_1078733_S05'], 
               'x1103204': ['AK_1103204_S01_V1', 'AK_1103204_S02_V1', 'AK_1103204_S03_CB_V1', 'AK_1103204_S04_Hyp_V1', 'AK_1103204_S05_SNr_RS', 'AK_1103204_S06_RS_latV1', 'AK_1103204_S07_STN_Sub', 'AK_1103204_S08_Sub'], 
               'x1108393': ['AK_1108393_S01_V1', 'AK_1108393_S02_PAG_V1', 'AK_1108393_S03_SNr_RS', 'AK_1108393_S04_RL_SC', 'AK_1108393_S05_SNr_V1', 'AK_1108393_S06_MRN_V1', 'AK_1108393_S07_PEG_V1', 'AK_1108393_S08', 'AK_1108393_S09_Pons', 'AK_1108393_S10'], 
               'x1116760': ['AK_1116760_S01_M2', 'AK_1116760_S02_M2', 'AK_1116760_S03', 'AK_1116760_S04_M2', 'AK_1116760_S05_M2', 'AK_1116760_S06_M2', 'AK_1116760_S07_M2_V1', 'AK_1116760_S08_M2_V1', 'AK_1116760_S09_M2_V1', 'AK_1116760_S10_V1'], 
               'x1116763': ['ML_1116763_S01_AM', 'ML_1116763_S02_VM_M2', 'ML_1116763_S03_VM_M1', 'ML_1116763_S04_VA', 'ML_1116763_S05_VA', 'ML_1116763_S06_thal', 'ML_1116763_S07_thal', 'ML_1116763_S08_thalv2', 'ML_1116763_S09_thal', 'ML_1116763_S10_thal'], 
               'x1116764': ['ML_1116764_S01_V1', 'ML_1116764_S02_M2_V1', 'ML_1116764_S03_M2_V1', 'ML_1116764_S04_M2_SNr', 'ML_1116764_S05_V1', 'ML_1116764_S06_V1', 'ML_1116764_S07_V1'], 
               'x1116765': ['AK_1116765_S01_GPe_V1', 'AK_1116765_S02_Sub', 'AK_1116765_S03_V1RL_Str', 'AK_1116765_S04_M1_SNr', 'AK_1116765_S05_M2', 'AK_1116765_S06_M1_MRN', 'AK_1116765_S07_GPe', 'AK_1116765_S08_AMHyp', 'AK_1116765_S09_M1_HypPF', 'AK_1116765_S10_AMLPPF'], 
               'x1117910': ['AK_1117910_S01_CB', 'AK_1117910_S02_CB_M2', 'AK_1117910_S03_CB_M2', 'AK_1117910_S04_CB_M2', 'AK_1117910_S05_CB', 'AK_1117910_S06_CB_M2', 'AK_1117910_S07_CB_M2', 'AK_1117910_S08_CB_M2', 'AK_1117910_S09_CB_M2', 'AK_1117910_S10_CB'], 
               'x1117915': ['AK_1117915_S01_M2', 'AK_1117915_S02_M2_CB', 'AK_1117915_S03_M2_CB', 'AK_1117915_S04_CB_M2'], 
               'x1119408': ['ML_1119408_MD_S11', 'ML_1119408_RE_S12', 'ML_1119408_RL_S04_v2', 'ML_1119408_SNr_GPe_S09', 'ML_1119408_Str_S13', 'ML_1119408_thal_S01', 'ML_1119408_thal_S02', 'ML_1119408_thal_S03', 'ML_1119408_thal_S06', 'ML_1119408_thal_S07_v2', 'ML_1119408_thal_S08'], 
               'x1119409': ['ML_1119409_S01', 'ML_1119409_S02', 'ML_1119409_S03', 'ML_1119409_S04', 'ML_1119409_S05', 'ML_1119409_S06'], 
               'x1119541': ['AK_1119541_S01_CB', 'AK_1119541_S02_M1_CB', 'AK_1119541_S03_GPe_CB', 'AK_1119541_S04_GPe_CB', 'AK_1119541_S05_Th_CB', 'AK_1119541_S06_GPe_CB', 'AK_1119541_S07_M1_CB', 'AK_1119541_S08_M1', 'AK_1119541_S09_M1_CB', 'AK_1119541_S10_M1_CB'], 
               'x1119544': ['AK_1119544_S01_CB', 'AK_1119544_S02_CB', 'AK_1119544_S03', 'AK_1119544_S04_CB', 'AK_1119544_S05_CB_V1', 'AK_1119544_S06_CB_Ent', 'AK_1119544_S07_Ent', 'AK_1119544_S08_CB'], 
               'x1120567': ['ML_1120567_S01', 'ML_1120567_S02', 'ML_1120567_S03', 'ML_1120567_S04']}


MAX_TRIAL_DURATION = 2500
N_CELLS = 15406


def process_probe_data(probe_data, n_probes, events, video, motion_corr=True):
    """ 
    Function to aggregate post-processing steps (matching spikes to clusters, binning, smoothing 
    and removing motion onset times).

    args:
        probe_data (dict): dictionary of probe info including spike time and cluster info
        n_probes (int): number of probes
        events (dict): events dict
        video (dict): video dict
        motion_corr (bool): if True, removes data from 1s before motion onset
        time_window (2-tuple): start and end time in seconds of data to be returned

    returns:
        cleaned_spikes (dict): post-processed spike data for the given probe
    """
    # Get trial start, finish and duration, regardless of trial performance and number of probes
    trial_times = get_trial_times(
                                  events['Baseline_ON']['rise_t'].flatten(),
                                  events['Baseline_ON']['fall_t'].flatten(),
                                  events['Change_ON']['fall_t'].flatten(),
                                  )

    def _bin_smooth_align_clean(clusters, spike_times):
        # Bin and smooth spike times
        session_duration = np.max(spike_times)
        matched_spikes = match_clusters_and_spike_times(clusters, spike_times)
        matched_spikes_binned = bin_spike_times(matched_spikes, session_duration)
        smoothed_spikes_binned = smooth_firing(matched_spikes_binned)
        
        # Align and clean spike data
        aligned_spikes = match_unit_firing_to_trial_start(smoothed_spikes_binned, trial_times)
        if motion_corr:
            motion_onset_times = np.array(video['MotionOnsetTimes']).flatten()
            return remove_spikes_before_motion_onset(aligned_spikes, trial_times['rise_t'], motion_onset_times)
        else: 
            return aligned_spikes

    if n_probes == 1:
        spike_times = np.array(probe_data['st'])
        clusters = np.array(probe_data['clu'])
        cleaned_spikes = _bin_smooth_align_clean(clusters, spike_times)
        
    elif n_probes > 1:
        cleaned_spikes = []
        for probe_no in range(n_probes):
            spike_times = np.array(probe_data['st'][probe_no])
            clusters = np.array(probe_data['clu'][probe_no])
            cleaned_spikes.append(_bin_smooth_align_clean(clusters, spike_times))
    
    else:
        raise ValueError("n_probes must be a positive integer")
    
    return cleaned_spikes


def _get_n_cells(data_dict):
    """Helper function to get number of cells in dictionary"""   
    n_cells = 0
    for mouse in data_dict.keys():
        for sess in data_dict[mouse].keys():
            for probe in data_dict[mouse][sess].keys():
                n_cells += data_dict[mouse][sess][probe].shape[0]
    return n_cells


def aggregate_data(data, dim0, dim1, dtype=np.float64, pad=True):
    """Reorganise results as homogeneous array"""
    cell_array = np.empty((dim0, dim1), dtype=dtype)
    curr_start_index = 0
    for mouse in data.keys(): 
        for sess in sessions[mouse]:
            for probe in data[mouse][sess]:
                curr_data = data[mouse][sess][probe]
                curr_n_cells, curr_max_trial_duration = curr_data.shape[0], curr_data.shape[1]
                if pad:
                    pad_length = dim1 - curr_max_trial_duration
                    curr_data = np.pad(curr_data, pad_width=((0, 0), (0, pad_length)), constant_values=np.nan)
                curr_end_index = curr_start_index + curr_n_cells
                cell_array[curr_start_index:curr_end_index] = curr_data
                curr_start_index = curr_end_index
    return cell_array


def get_results(mice, sessions, behav_key="IsFA", brain_region=None, brain_region_list_key='brain_region_comb'):
    results = {}
    for mouse in mice: 
        results[mouse] = {}
        for sess in sessions[mouse]:
            results[mouse][sess] = {}
            print(mouse, sess)
            events, probes, video, behav = get_session_data(DATA_FNAME, mouse, sess)
            trial_type_data = behav['trials_data_exp']
            n_probes = len(probes['offset'])
            spikes_cleaned = process_probe_data(probes, n_probes, events, video)

            max_dur = 0

            # Handle single probe
            if n_probes == 1:
                # Get spike data
                curr_brain_regions = convert_regions_to_strings(probes['good_and_stab_cl_coord'][brain_region_list_key])
                trials_of_interest = subset_trials(spikes_cleaned, trial_type_data, curr_brain_regions, behav_key=behav_key, region=brain_region)
                n_timepoints = trials_of_interest.shape[0]
                max_dur = n_timepoints if n_timepoints > max_dur else max_dur
                results[mouse][sess][0] = average_over_trials(trials_of_interest)

            # Or handle multiple probes
            elif n_probes > 1:
                for probe in range(n_probes):
                    # Get spike data
                    curr_brain_regions = convert_regions_to_strings(probes['good_and_stab_cl_coord'][probe][0][brain_region_list_key])
                    trials_of_interest = subset_trials(spikes_cleaned[probe], trial_type_data, curr_brain_regions, behav_key=behav_key, region=brain_region)
                    max_dur = n_timepoints if n_timepoints > max_dur else max_dur
                    results[mouse][sess][probe] = average_over_trials(trials_of_interest)

    max_dur = max_dur if max_dur < 2500 else 2500 # Set to prevent too large array from erroneously timed trials
    n_cells_total = _get_n_cells(results)
    return aggregate_data(results, n_cells_total, max_dur)


def get_metadata(mice, sessions, behav_key='IsFA', time_lock='trialStart'):
    info = {}
    n_cols = 9
    for mouse in mice: 
        info[mouse] = {}
        for sess in sessions[mouse]:
            print(mouse, sess)
            info[mouse][sess] = {}
            cluster_ids, cluster_coords, n_probes = get_probes_metadata_only(DATA_FNAME, mouse, sess)

            # Handle single probe
            if n_probes == 1:
                # Get metadata
                brain_regions = convert_regions_to_strings(cluster_coords['brain_region'])
                brain_regions_comb = convert_regions_to_strings(cluster_coords['brain_region_comb'])
                n_cells = len(brain_regions_comb)                
                ap_position = cluster_coords['y']                
                curr_info = np.empty((n_cells, n_cols), dtype='object')
                curr_info[:, :3] = mouse, sess, 0
                curr_info[:, 3] = cluster_ids
                curr_info[:, 4] = brain_regions_comb
                
                curr_info[:, 5] = brain_regions
                curr_info[:, 6] = ap_position
                curr_info[:, 6] = [
                    ap[0].item() if isinstance(ap, list) else ap
                    for ap in curr_info[:, 6]
                ]
                curr_info[:, 7] = behav_key
                curr_info[:, 8] = time_lock
                info[mouse][sess][0] = curr_info

            # Or handle multiple probes
            elif n_probes > 1:
                for probe in range(n_probes):
                    # Get metadata
                    curr_cluster_ids = cluster_ids[probe][0]
                    brain_regions = convert_regions_to_strings(cluster_coords[probe][0]['brain_region'])
                    brain_regions_comb = convert_regions_to_strings(cluster_coords[probe][0]['brain_region_comb'])
                    n_cells = len(brain_regions_comb)
                    ap_position = cluster_coords[probe][0]['y']
                    curr_info = np.empty((n_cells, n_cols), dtype='object')
                    curr_info[:, :3] = mouse, sess, probe
                    curr_info[:, 3] = curr_cluster_ids
                    curr_info[:, 4] = brain_regions_comb
                    curr_info[:, 5] = brain_regions
                    curr_info[:, 6] = ap_position
                    curr_info[:, 6] = [
                        ap[0].item() if isinstance(ap, list) else ap
                        for ap in curr_info[:, 6]
                    ]
                    curr_info[:, 7] = behav_key
                    curr_info[:, 8] = time_lock
                    info[mouse][sess][probe] = curr_info

    n_cells_total = _get_n_cells(info)
    return aggregate_data(info, n_cells_total, n_cols, dtype='object')


def flatten_baseline_dict(baseline_dict, mice, sessions):
    """
    Flatten a nested dictionary of shape:
    
        baseline_dict[mouse][session][probe] = array_of_shape_n_by_2
    
    into a single concatenated array of shape (sum_of_n, 2).

    The order is determined by the order of `mice` and `sessions[mouse]`.
    """
    cat_arrays = []
    
    for mouse in mice: 
        for sess in sessions[mouse]:
            for _, arr in baseline_dict[mouse][sess].items():
                print(arr.shape)
                cat_arrays.append(arr)
    
    # Now stack all of these n-by-2 arrays on top of each other
    if len(cat_arrays) > 0:
        final_array = np.vstack(cat_arrays)
    else:
        # Handle the edge case if there's nothing to stack
        final_array = np.array([]).reshape(0, 2)
    
    return final_array

def get_baseline_mean_and_std(mice, sessions, behav_key='IsHit', brain_region=None, brain_region_list_key='brain_region_comb', time_idxs=(200, 300)):
    """
    Get baseline mean and standard deviation
    
    args:
        mice (list): mice
        sessions (dict): mapping of mice to (list of) sessions
        behav_key (str or list(str)): key for trial type(s)
        brain_region (str): specify specific brain region, default None
        brain_region_list_key (str): specify key to extract brain regions
        time_idxs (tuple): specify time indicies (bins of 10ms, starting
            from -3s before trial start) from which to extract baseline
            
    returns:
        np.array: array of cells x (mean, std)"""

    results = {}
    for mouse in mice: 
        results[mouse] = {}
        for sess in sessions[mouse]:
            results[mouse][sess] = {}
            print(mouse, sess)
            events, probes, video, behav = get_session_data(DATA_FNAME, mouse, sess)
            trial_type_data = behav['trials_data_exp']
            n_probes = len(probes['offset'])
            spikes_cleaned = process_probe_data(probes, n_probes, events, video)
            
            # Handle single probe
            if n_probes == 1:
                # Subset spike data
                curr_brain_regions = convert_regions_to_strings(probes['good_and_stab_cl_coord'][brain_region_list_key])
                trials_of_interest = subset_trials(spikes_cleaned, trial_type_data, curr_brain_regions, behav_key=behav_key, region=brain_region)

                # Get means and stds
                means, stds = [], []
                for value in trials_of_interest.values():
                    curr_trials = value[:, time_idxs[0]:time_idxs[1]] if hasattr(time_idxs, '__iter__') else value
                    means.append(np.nanmean(curr_trials))
                    stds.append(np.nanstd(curr_trials))
                results[mouse][sess][0] = np.array((means, stds)).T
                
            # Or handle multiple probes
            elif n_probes > 1:
                for probe in range(n_probes):
                    # Subset spike data
                    curr_brain_regions = convert_regions_to_strings(probes['good_and_stab_cl_coord'][probe][0][brain_region_list_key])
                    trials_of_interest = subset_trials(spikes_cleaned[probe], trial_type_data, curr_brain_regions, behav_key=behav_key, region=brain_region)
                    
                    # Get means and stds
                    means, stds = [], []
                    for value in trials_of_interest.values():
                        curr_trials = value[:, time_idxs[0]:time_idxs[1]] if hasattr(time_idxs, '__iter__') else value
                        means.append(np.nanmean(curr_trials))
                        stds.append(np.nanstd(curr_trials))
                    results[mouse][sess][probe] = np.array((means, stds)).T

    return flatten_baseline_dict(results, mice, sessions)



hit_all = 0
if hit_all:
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
    session_duration = np.max(spike_times)
    matched_spikes_binned = bin_spike_times(matched_spikes, session_duration)
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

functionless = False
if functionless:
    results = {}
    info = {}
    behav_key = "IsFA"
    time_lock = 'trialStart'
    for mouse in mice: 
        results[mouse] = {}
        info[mouse] = {}
        for sess in sessions[mouse]:
            results[mouse][sess] = {}
            info[mouse][sess] = {}
            print(mouse, sess)
            events, probes, video, behav = get_session_data(DATA_FNAME, mouse, sess)
            trial_type_data = behav['trials_data_exp']
            n_probes = len(probes['offset'])
            spikes_cleaned = process_probe_data(probes, n_probes, events, video)

            # Handle single probe
            if n_probes == 1:
                # Get spike data
                brain_regions = convert_regions_to_strings(probes['good_and_stab_cl_coord']['brain_region'])
                brain_regions_comb = convert_regions_to_strings(probes['good_and_stab_cl_coord']['brain_region_comb'])
                trials_of_interest = subset_trials(spikes_cleaned, trial_type_data, brain_regions_comb, behav_key=behav_key, region=None)
                results[mouse][sess][0] = average_over_trials(trials_of_interest)

                # Get metadata
                n_cells = len(brain_regions_comb)
                ap_position = probes['good_and_stab_cl_coord']['y']
                curr_info = np.empty((n_cells, 8), dtype='object')
                curr_info[:, :3] = mouse, sess, 0
                curr_info[:, 3] = brain_regions_comb
                curr_info[:, 4] = brain_regions
                curr_info[:, 5] = ap_position
                curr_info[:, 6] = behav_key
                curr_info[:, 7] = time_lock
                info[mouse][sess][0] = curr_info

            # Or handle multiple probes
            elif n_probes > 1:
                for probe in range(n_probes):
                    # Get spike data
                    brain_regions = convert_regions_to_strings(probes['good_and_stab_cl_coord'][probe][0]['brain_region'])
                    brain_regions_comb = convert_regions_to_strings(probes['good_and_stab_cl_coord'][probe][0]['brain_region_comb'])
                    trials_of_interest = subset_trials(spikes_cleaned[probe], trial_type_data, brain_regions_comb, behav_key=behav_key, region=None)
                    results[mouse][sess][probe] = average_over_trials(trials_of_interest)

                    # Get metadata
                    n_cells = len(brain_regions_comb)
                    ap_position = probes['good_and_stab_cl_coord'][probe][0]['y']
                    curr_info = np.empty((n_cells, 8), dtype='object')
                    curr_info[:, :3] = mouse, sess, probe
                    curr_info[:, 3] = brain_regions_comb
                    curr_info[:, 4] = brain_regions
                    curr_info[:, 5] = ap_position
                    curr_info[:, 6] = behav_key
                    curr_info[:, 7] = time_lock
                    info[mouse][sess][probe] = curr_info




# np.savez_compressed('/data/earlyLick_trialStart_all', early_lick_data=results, metadata=metadata)
# fname = "data/baseline_mean_std_1sBeforeTrialStart-trialStart_hit_all"
# np.save(fname, baseline)