#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:23:19 2021

@author: jtm
"""
import os
import os.path as op
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal


SAMPLE_RATE = 250

def pupil_preprocessing_fig(nrows, subject, **kwargs):
    f, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(14,9))
    for ax in axs:
        ax.set_ylabel('Pupil size')    
        ax.set_xlabel('Timestamp')
    f.suptitle(f'Preprocessing for subject: {subject}')
    return f, axs


def new_dataset(edf):
    basename = op.basename(edf).split('.')[0]
    print('{}\n{:*^120s}\n{}'.format('*'*120, ' ' + basename + ' ', '*'*120))
    return basename


def print_details(samps, blinks, events):
    print(f'> Loaded {len(samps)} samples ({len(samps)/SAMPLE_RATE} s recording)')
    print(f'> Parsed {len(events)} experimental events')
    print(f'> Keys detected: {len(events[events.event=="key"])}')
    print(f'> Blinks detected: {len(blinks)} (average duration = {round(blinks.duration.mean())} ms)')
    
    
def read_dataset(edf):
    basename = op.splitext(edf)[0] ## remove .edf from filename
    fname_samples = op.join(basename + '_samples.asc')
    fname_events = op.join(basename + '_events.asc')
    
    print(f'> Attempt loading {fname_samples} and {fname_events}')
    ## read samples-file
    df_samps = pd.read_table(
        fname_samples, sep='\t', index_col='time',
        names=['time', 'x', 'y', 'p', 'samp_warns'])
    df_samps.replace('   .', np.NaN, inplace=True) # dodgy string values prevent float conversion
    df_samps[['x','y']] = df_samps[['x','y']].astype('float')
    
    # read the whole file into variable `events` (list with one entry per line)
    with open(fname_events) as f:
        events = f.readlines()
    
    # only events from start of experiment
#    experiment_start_index = np.where(
#        ['still_gabors' in ev for ev in events])[0][0]
#    events = events[experiment_start_index:]
    
    # Get the blinks
    blinks = [ev for ev in events if ev.startswith('EBLINK')]
    df_blinks = pd.DataFrame([b.split() for b in blinks])
    df_blinks.columns = ['event','eye','start','end','duration']
    df_blinks[['start','end','duration']] = df_blinks[['start','end','duration']].astype('int')
    
    
    # Get the events
    events = [ev for ev in events if ev.startswith('MSG')]
    df_events = pd.DataFrame([ev.split() for ev in events])
    df_events = (df_events.loc[((df_events[3].str.contains('DISPLAY_target', na=False))|
                                (df_events[3].str.contains('KEYBOARD_response', na=False)))])
    df_events = df_events[[1,2,3]].copy()
    df_events.columns=['time', 'key', 'event']
    df_events.loc[df_events['key']=='KEYBOARD_response', 'event'] = 'key'
    df_events = df_events[['time','event']]
    df_events.time = df_events.time.astype('int')
    
    print_details(df_samps, df_blinks, df_events)
    
    return df_samps, df_blinks, df_events


# def adjust_eyelink_recov_idxs(
#         samples, blinks, z_thresh=.1, window=250, kernel_size=25):
#     """ Extends event endpoint until the z-scored derivative of 'field's timecourse drops below thresh
#     We will try to extend *every* event passed in.
#     Parameters
#     ----------
#     samples (list of dicts)
#         A Samples object
#     events (list of dicts)
#         An Events object
#     z_thresh (float)
#         The threshold below which the z-score of the timecourse's gradient
#         must fall before we'll consider the event over.
#     field (string)
#         The field to use.
#     window (int)
#         The number of indices you'll search through for z-threshold
#     kernel_size (int)
#         The number of indices we'll average together at each measurement. So
#         what you really get with this method is the index of the first voxel
#         whose gradient value, when averaged together with that of the
#         n='kernel' indices after it, then z-scored, is below the given z
#         threshold.
#     """
#     #breakpoint()
#     # use pandas to take rolling mean. pandas' kernel looks backwards, so we need to pull a reverse...
#     dfs = np.gradient(samples['p'].values)
#     reversed_dfs = dfs[::-1]
#     reversed_dfs_ravg = (pd.DataFrame(reversed_dfs).rolling(
#             window=kernel_size, min_periods=1).mean())
#     dfs_ravg = reversed_dfs_ravg[::-1]
#     dfs_ravg = np.abs((dfs_ravg - np.mean(dfs_ravg)) / np.std(dfs_ravg))
#     samp_count = len(samples)
#     # search for drop beneath z_thresh after end index
#     blinks.index = blinks.start
#     new_durs = []
#     for idx, dur in blinks.duration.items():
#         #print(idx, dur)
#         try:
#             s_pos = samples.index.get_loc(idx + dur) - 1
#             e_pos = samples.index[min(s_pos + window, samp_count - 1)]
#         except Exception as e:
#             # can't do much about that
#             s_pos = e_pos = 0
#         if s_pos == e_pos:
#             new_durs.append(dur)
#             continue
#         e_dpos = np.argmax(dfs_ravg[s_pos:e_pos] < z_thresh)  # 0 if not found
#         new_end = samples.index[min(s_pos + e_dpos, samp_count - 1)]
#         new_durs.append(new_end - idx)
#     blinks.duration = new_durs
#     return blinks
    
def mask_blinks(samples, blinks):
    extend_blinks_by = (SAMPLE_RATE/1000)*100 # 100 ms
    samps = samples.copy()
    # if find_recovery:
    #     blinks = adjust_eyelink_recov_idxs(samples, blinks)
    for idx, row in blinks.iterrows():
        # adjust start and end idx by 100 ms to account for artefactual saccades
        start, end = int(row.start)-extend_blinks_by, int(row.end)+extend_blinks_by
        samps.loc[start:end, 'p'] = np.NaN
        samps['masked'] = np.where(np.isnan(samps['p'].values), 1, 0)
    print('> Masked {} samples due to blinks ({} s)'.format(
        len(samps[samps['masked']==1]), len(samps[samps['masked']==1])/SAMPLE_RATE))
    return samps


def butterworth_series(samples,
                       fields=['p'], 
                       filt_order=3,
                       cutoff_freq=4/(SAMPLE_RATE/2),
                       inplace=False):
    '''Applies a Butterworth filter to the given fields. 

    Parameters
    ----------
    samples : `pandas.DataFrame`
        DataFrame of samples containing the pupil data.
    fields : list, optional
        List of columns to be filtered. The default is ['diameter'].
    filt_order : int, optional
        Order of the filter. The default is 3.
    cutoff_freq : float
        Normalised cut-off frequency in hertz. For 4 Hz cut-off, this should 
        4/(sample_rate/2). The default is .01.
    inplace : bool, optional
        Whether to modify `samples` in place. The default is False.

    Returns
    -------
    samps : 
        The samples.

    '''
    samps = samples if inplace else samples.copy(deep=True)
    B, A = signal.butter(filt_order, cutoff_freq, output='BA')
    samps[fields] = samps[fields].apply(
        lambda x: signal.filtfilt(B, A, x), axis=0)
    print(f'> Applied butterworth filter to pupil data (order = {filt_order}, cut_off = {cutoff_freq})')
    return samps


#TODO: optimse and debug
def extract(samples, 
            events, 
            offset=0, 
            duration=0,
            borrow_attributes=[]):
    '''
    Extracts ranges from samples based on event timing and sample count.
    
    Parameters
    ----------
    samples : pandas.DataFrame
        The samples from which to extract events. Index must be timestamp.
    events : pandas.DataFrame
        The events to extract. Index must be timestamp.
    offset : int, optional
        Number of samples to offset from baseline. The default is 0.
    duration : int, optional
        Duration of all events in terms of the number of samples. Currently 
        this has to be the same for all events, but could use a 'duration' 
        column in events DataFrame if needs be. The default is 0.
    borrow_attributes : list of str, optional
        List of column names in the events DataFrame whose values should be
        copied to the respective ranges. For each item in the list, a
        column will be created in the ranges dataframe - if the column does
        not exist in the events dataframe, the values in the each
        corresponding range will be set to float('nan'). This is uesful for 
        marking conditions, grouping variables, etc. The default is [].
        
    Returns
    -------
    df : pandas.DataFrame
        Extracted events complete with hierarchical multi-index.
        
    '''
    # negative duration should raise an exception
    if duration <= 0:
        raise ValueError('Duration must be >0')

    # get the list of start time indices
    event_starts = events.index.to_series()

    # find the indexes of the event starts, and offset by sample count
    range_idxs = np.searchsorted(
        samples.index, event_starts.iloc[:], 'left') + offset
    range_duration = duration
    
    # make a hierarchical index
    samples['orig_idx'] = samples.index
    midx = pd.MultiIndex.from_product(
        [list(range(len(event_starts))), list(range(range_duration))],
        names=['event', 'onset'])
    
    # get the samples
    df = pd.DataFrame()
    idx = 0
    for start_idx in range_idxs:
        # get the start time and add the required number of indices
        end_idx = start_idx + range_duration - 1  # .loc indexing is inclusive
        new_df = deepcopy(
            samples.loc[samples.index[start_idx] : samples.index[end_idx]])
        for ba in borrow_attributes:
            new_df[ba] = events.iloc[idx].get(ba, float('nan'))
        df = pd.concat([df, new_df])
        idx += 1
    df.index = midx
    print('> Extracted ranges for {} events'.format(len(events)))
    return df

def reject_bad_trials(ranges, interp_thresh=20, drop=False):
    '''Drop or markup trials which exceed a threshold of interpolated data.
    
    Parameters
    ----------
    ranges : pandas.DataFrame
        Extracted event ranges with hierarchical pd.MultiIndex.
    interp_thresh : int, optional
        Percentage of interpolated data permitted before trials are marked for
        rejection / dropped. The default is 20.
    drop : bool, optional
        Whether to drop the trials from the ranges. The default is False.
        
    Returns
    -------
    ranges : pandas.DataFrame
        Same as ranges but with a column identifying trials marked for
        rejection (drop = False) or with those trials dropped from the 
        DataFrame (drop = True).
        
    '''
    if not isinstance(ranges.index, pd.MultiIndex):
        raise ValueError('Index of ranges must be pd.MultiIndex')
        
    pct_interp = ranges.groupby(by='event').agg(
        {'masked': lambda x: float(x.sum())/len(x)*100})
    print(f'> Average percentage of data interpolated for each trial: {round(pct_interp.mean()[0])}')
    
    reject_idxs = (pct_interp.loc[pct_interp['masked'] > interp_thresh]
                             .index.to_list())
    ranges['reject'] = 0
    if reject_idxs:
        ranges.loc[reject_idxs, 'reject'] = 1
    if drop:
        ranges = ranges.drop(index=reject_idxs)
        print('> {} trials were dropped from the DataFrame'.format(
            len(reject_idxs)))
    else:
        print('> {} trials were marked for rejection'.format(len(reject_idxs)))
    return ranges