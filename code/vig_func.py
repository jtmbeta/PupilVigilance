#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for analysing data from the vigilance experiment
==========================================================

Credit to Ben Acland (cili) and Matthias Mitner (pypillometry), whose code we
have adapted.

@author: jtm

"""

import os
import os.path as op
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from numpy import nanmedian

# Sampling rate of the eye tracker
SAMPLE_RATE = 250


def pupil_preprocessing_fig(nrows, subject):
    """
    Return a figure with axes for showing the processing steps of pupil data.

    Parameters
    ----------
    nrows : int
        How many rows.
    subject : str
        The subject ID. Used for the title of the figure.

    Returns
    -------
    fig : plt.Figure
        Figure object.
    axs : np.array
        Array of matplotlib axes objects.

    """
    fig, axs = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(14, 9))
    for ax in axs:
        ax.set_ylabel('Pupil size')
        ax.set_xlabel('Timestamp')
    fig.suptitle(f'Preprocessing for subject: {subject}')
    return fig, axs


def new_dataset(edf):
    """
    Determine the subject ID from the name of the EDF file a print some stars.

    Parameters
    ----------
    edf : str
        Path to EDF.

    Returns
    -------
    basename : str
        The subject ID.

    """
    basename = op.basename(edf).split('.')[0]
    print('{}\n{:*^60}\n{}'.format('*'*60, ' ' + basename + ' ', '*'*60))
    return basename


def print_details(samps, blinks, events):
    print(f'> Loaded {len(samps)} samples ({len(samps)/SAMPLE_RATE} s)')
    print(f'> Parsed {len(events)} experimental events')
    print(f'> Keys detected: {len(events[events.event=="key"])}')
    print(f'> Blinks detected: {len(blinks)}')
    print(f'\taverage duration = {round(blinks.duration.mean())} ms)')


def read_dataset(edf):
    """
    Read the EDF and return separate DataFrames for samples, blinks and events.

    Parameters
    ----------
    edf : str
        Path to EDF.

    Returns
    -------
    df_samps : pd.DataFrame
        EyeLink data samples.
    df_blinks : pd.DataFrame
        EyeLink blink events.
    df_events : pd.DataFrame
        EyeLink events.

    """
    # Get paths to the ASCII files for samples and events
    basename = op.splitext(edf)[0]  # remove .edf from path
    fname_samples = op.join(basename + '_samples.asc')
    fname_events = op.join(basename + '_events.asc')
    print(f'> Attempt loading {fname_samples} and {fname_events}')

    # Read samples-file
    df_samps = pd.read_table(
        fname_samples, sep='\t', index_col='time',
        names=['time', 'x', 'y', 'p', 'samp_warns'])

    # Set strange string values preventing float conversion to NaN
    df_samps.replace('   .', np.NaN, inplace=True)
    df_samps[['x', 'y']] = df_samps[['x', 'y']].astype('float')

    # Read whole file into variable `events` (list with one entry per line)
    with open(fname_events) as f:
        events = f.readlines()

    # Only events from start of experiment. This is marked by a 'still_gabors'
    # message in the ASCII file.
    experiment_start_index = np.where(
        ['still_gabors' in ev for ev in events])[0][0]
    events = events[experiment_start_index:]

    # Get the blinks
    blinks = [ev for ev in events if ev.startswith('EBLINK')]
    df_blinks = pd.DataFrame([b.split() for b in blinks])
    df_blinks.columns = ['event', 'eye', 'start', 'end', 'duration']
    df_blinks[['start', 'end', 'duration']] = df_blinks[[
        'start', 'end', 'duration']].astype('int')

    # Get the MSG events
    events = [ev for ev in events if ev.startswith('MSG')]
    df_events = pd.DataFrame([ev.split() for ev in events])
    df_events = (  # The three types of event we are interesed in
        df_events.loc[((df_events[2].str.contains('KEY_PRESS_FOUND')) |
                       (df_events[3].str.contains('NORMAL', na=False)) |
                       (df_events[3].str.contains('RARE', na=False)))]
    )

    # Get rid of the junk and give sensible names
    df_events = df_events[[1, 2, 3]].copy()
    df_events.columns = ['time', 'key', 'msg']
    df_events.loc[df_events['key'] == 'KEY_PRESS_FOUND', 'event_type'] = 'key'
    df_events.loc[df_events['msg'].str.contains(
        'NORMAL', na=False), 'event_type'] = 'normal'
    df_events.loc[df_events['msg'].str.contains(
        'RARE', na=False), 'event_type'] = 'rare'
    df_events['event'] = df_events['msg'].str.extract('(\d\d\d\d)')
    df_events['event'] = df_events['event'].str.lstrip('0')
    df_events = df_events[['time', 'event_type', 'event']]
    df_events.time = df_events.time.astype('int')
    df_events.index = df_events.time.astype('int')

    # Print summary of loaded data
    print_details(df_samps, df_blinks, df_events)

    return df_samps, df_blinks, df_events


def mask_blinks(samples, blinks, extend_by=100):
    """
    Mask data affected by blinks with NaN.

    Parameters
    ----------
    samples : pd.DataFrame
        DataFrame of samples.
    blinks : pd.DataFrame
        DataFrame of blink events.
    extend_by : int, optional
        Number of miliseconds to extend the onset and offset by. The default
        is 100, which accounts well for artefactual saccades that wrap the
        blinks and seems to be a common standard in the literature.

    Returns
    -------
    samps : pd.DataFrame
        Samples with masked blinks.

    """
    # Work out how many milliseconds to extend by
    extend_blinks_by = (SAMPLE_RATE/1000) * extend_by
    samps = samples.copy()

    # Iterate over blink events
    for idx, row in blinks.iterrows():

        # Adjust start and end idx to account for artefactual saccades
        start, end = (int(row.start)
                      - extend_blinks_by, int(row.end) + extend_blinks_by)
        samps.loc[start:end, 'p'] = np.NaN

        # A column to show which samples were masked so we can refer back to it
        # after the data have been interpolated.
        samps['masked'] = np.where(np.isnan(samps['p'].values), 1, 0)
    print('> Masked {} percent of samples due to blinks'.format(
        round(samps.masked.value_counts(normalize=True)[1] * 100, 2)))
    return samps


def butterworth_series(samples,
                       fields=['p'],
                       filt_order=3,
                       cutoff_freq=4/(SAMPLE_RATE/2),
                       inplace=False):
    """Apply Butterworth filter to specified fields.

    Parameters
    ----------
    samples : pd.DataFrame
        DataFrame of samples containing the pupil data.
    fields : list, optional
        List of columns to be filtered. The default is ['p'].
    filt_order : int, optional
        Order of the filter. The default is 3.
    cutoff_freq : float
        Normalised cut-off frequency in hertz. For 4 Hz cut-off, this should
        4/(SAMPLE_RATE/2).
    inplace : bool, optional
        Whether to modify samples in place. The default is False.

    Returns
    -------
    samps : pd.DataFrame
        The samples.

    """
    # Copy of samples
    samps = samples if inplace else samples.copy(deep=True)

    # Coefficients for the filter
    B, A = signal.butter(filt_order, cutoff_freq, output='BA')

    # Apply filter to specified fields
    samps[fields] = samps[fields].apply(
        lambda x: signal.filtfilt(B, A, x), axis=0)
    print(f'> Applied butterworth filter to {fields}.')
    print(f'> Order = {filt_order}, cut_off = {cutoff_freq}')
    return samps


def mad(data):
    """
    Function to compute the median absolute deviation. See Leys et al. (2013).

    """
    med = nanmedian(data)
    return nanmedian(abs(data - med)) * 1.4826


def calculate_RTs(events):
    """
    Assign keys to events and calculate RTs.

    Parameters
    ----------
    events : pd.DataFrame
        Events returned by read_dataset.

    Returns
    -------
    rts : pd.DataFrame
        Same as events but with added RT columns.

    """
    # Copy of the events DataFrame
    rts = events.copy(deep=True)
    last_rare_idx = None
    last_normal_idx = None

    # Loop over rows and check what event we are dealing with
    for idx, row in rts.iterrows():
        if row['event_type'] == 'rare':
            last_rare_idx = idx
        if row['event_type'] == 'normal':
            last_normal_idx = idx
        if row['event_type'] == 'key':
            if last_rare_idx == None:  # Edge case
                # A key was pressed before the first rare event
                rts.loc[idx, 'RT'] = -1  # Set RT to -1
                rts.loc[idx, 'last_normal_from_key_idx'] = last_normal_idx
            else:
                # Calculate RT, which is the time elapsed between the last
                # rare event and the current keypress
                rts.loc[idx, 'RT'] = (
                    row.time - rts.loc[last_rare_idx, 'time'])
                rts.loc[idx, 'event'] = rts.loc[last_rare_idx, 'event']
                
                # Update last rare / normal idx variables
                rts.loc[idx, 'last_rare_from_key_idx'] = last_rare_idx
                rts.loc[idx, 'last_normal_from_key_idx'] = last_normal_idx
    print('> Assigned keys to events and calculate RTs')
    return rts


def calculate_sd_outcomes(events):
    """
    Classify events as hits, misses, false alarms or correct rejections.

    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame, as returned by calculate_RTs.

    Returns
    -------
    evs : pd.DataFrame
        Same as events but with added outcome column.

    """
    evs = events.copy(deep=True)

    # Get the median and median-absolute-deviation of RT. We only consider
    # RTs less than 6000 ms, which is the minimum time between rare events.
    # RTs longer than this will be automatically designated as false alarms.
    #med_RT = evs.loc[evs.RT <= 6000, 'RT'].median()
    #mad_RT = mad(evs.loc[evs.RT <= 6000, 'RT'])

    # Hits are within 2 MADs either side of the group-level median RT
    evs.loc[((evs.RT <= 1149) & (evs.RT >= 225)), 'outcome'] = 'H'

    # False alarms are outside of this range
    evs.loc[((evs.RT >= 1149) | (evs.RT <= 225)), 'outcome'] = 'FA'

    # Edge case (button pressed before any rare events)
    (evs.loc[evs.RT.notna()
             & evs.last_rare_from_key_idx.isna(), 'outcome']) = 'FA'

    # Get the rare event indices for all hits so we can mark the event itself
    # as a hit and have a record of the relevant RT.
    hs = evs.loc[evs.outcome == 'H', 'last_rare_from_key_idx'].values
    evs.loc[hs, ['outcome', 'RT']] = (evs.loc[evs.outcome ==
                                              'H', ['outcome', 'RT']].values)
    # As above but for false alarms
    fas = evs.loc[evs.outcome == 'FA', 'last_normal_from_key_idx'].values
    evs.loc[fas, ['outcome', 'RT']] = (evs.loc[evs.outcome ==
                                               'FA', ['outcome', 'RT']].values)

    # A miss is when a rare event does not have a valid RT
    evs.loc[((evs.event_type == 'rare') & (evs.RT.isnull())), 'outcome'] = 'M'

    # All other events are correct rejections
    evs.loc[evs.outcome.isna(), 'outcome'] = 'CR'
    print('> Calculated signal detection outcomes')
    return evs


def extract(samples,
            events,
            offset=0,
            duration=0,
            borrow_attributes=[]):
    """
    Extracts ranges from samples based on event timing and sample count.

    Parameters
    ----------
    samples : pd.DataFrame
        The samples from which to extract events. Index must be timestamp.
    events : pd.DataFrame
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
    df : pd.DataFrame
        Extracted events complete with hierarchical multi-index.

    """
    # Negative duration should raise an exception
    if duration <= 0:
        raise ValueError('Duration must be >0')

    # Get the list of start time indices
    event_starts = events.index.to_series()

    # find the indexes of the event starts, and offset by sample count
    range_idxs = np.searchsorted(
        samples.index, event_starts.iloc[:], 'left') + offset
    range_duration = duration

    # Make hierarchical index
    samples['orig_idx'] = samples.index
    midx = pd.MultiIndex.from_product(
        [list(range(len(event_starts))), list(range(range_duration))],
        names=['event', 'onset'])

    # Get the samples
    df = pd.DataFrame()
    idx = 0
    for start_idx in range_idxs:
        # Get the start time and add the required number of indices
        end_idx = start_idx + range_duration - 1  # .loc indexing is inclusive
        new_df = deepcopy(
            samples.loc[samples.index[start_idx]: samples.index[end_idx]])

        # Borrow the specified attributes from master events DataFrame
        for ba in borrow_attributes:
            new_df[ba] = events.iloc[idx].get(ba, float('nan'))

        # Concatenate the DataFrames
        df = pd.concat([df, new_df])
        idx += 1

    # Set the multi index
    df.index = midx
    print('> Extracted ranges for {} events'.format(len(events)))
    return df


def reject_bad_trials(ranges, interp_thresh=20, drop=False):
    """Markup trials which exceed a threshold of interpolated data.

    Parameters
    ----------
    ranges : pd.DataFrame
        Extracted event ranges with hierarchical pd.MultiIndex.
    interp_thresh : int, optional
        Percentage of interpolated data permitted before trials are marked for
        rejection / dropped. The default is 20.
    drop : bool, optional
        Whether to drop the trials from the ranges. The default is False.

    Returns
    -------
    ranges : pd.DataFrame
        Same as ranges but with a column identifying trials marked for
        rejection (drop = False) or with those trials dropped from the
        DataFrame (drop = True).

    """
    # Require a muti index
    if not isinstance(ranges.index, pd.MultiIndex):
        raise ValueError('Index of ranges must be pd.MultiIndex')

    # Get the percentage of interpolated data
    pct_interp = ranges.groupby(by='event').agg(
        {'masked': lambda x: float(x.sum())/len(x)*100})
    print('> Average percentage of data interpolated for each trial:')
    print(f'\t{round(pct_interp.mean()[0])}')

    # Indices for trials to reject
    reject_idxs = (pct_interp.loc[pct_interp['masked'] > interp_thresh]
                             .index.to_list())

    # Mark up reject trials in the ranges DataFrame
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
