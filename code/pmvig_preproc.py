#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:14:49 2021

@author: jtm545
"""

#%% Imports
import sys
sys.path.insert(0, '../../code')
import glob

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pmvig_func import (new_dataset,
                      read_dataset, 
                      mask_blinks,
                      butterworth_series, 
                      pupil_preprocessing_fig,
                      extract,
                      reject_bad_trials)

sns.set(context='paper', style='ticks', font='sans-serif', font_scale=2.1)

#%% Preprocessing

# Get edf files
edfs = glob.glob('../data/**/*.edf')
edfs.sort()

# Loop over the EDFs
for edf in edfs:
    # Get subject id and read data
    subject = new_dataset(edf)
    fig, axs = pupil_preprocessing_fig(3, edf)
    samps, blinks, events = read_dataset(edf)
    blinks.to_hdf('../analysis/blinks.h5', key=subject)
    samps.p.plot(title='Raw', ax=axs[0], legend=True)
    
    # Mask blinks, interpolate and plot again
    samps = mask_blinks(samps, blinks) 
    samps.p.interpolate(inplace=True, limit_direction='both')
    samps.p.plot(title='Masked blinks / interpolated', ax=axs[1], legend=False)
    
    # Smooth and plot again
    samps = butterworth_series(
        samps,
        fields=['p'],
        filt_order=3,
        cutoff_freq=4/(250/2))
    samps.p.plot(
        title='3rd order Butterworth filter with 4 Hz cut-off', 
        ax=axs[2], legend=False)
    
    # Close figure
    plt.close(fig)

    # Save samples
    samps.to_hdf('../analysis/samples_250hz.h5', key=subject)
    
    # Downsample and save again
    samps = samps[::5]
    samps.to_hdf('../analysis/samples_50hz.h5', key=subject)
    
    # Calculate RTs
    target = (events.loc[events.event.str.contains('target')]
              .reset_index(drop=True))
    response = (events.loc[events.event.str.contains('response')]
                .reset_index(drop=True))
    response['RT'] = (response.time - target.time)
    response['1/RT'] = (1 / response['RT']) * 1000
    medRT = response.RT.median()
    response['Lapse'] = 0
    response.loc[response.RT > medRT * 2, 'Lapse'] = 1
    TrialGroup = []
    for t in range(1,6):
        TrialGroup.append([t] * 18)
    TrialGroup = [tg for sl in TrialGroup for tg in sl]
    response['TrialGroup'] = TrialGroup * 3
    response['Block'] = pd.cut(
        response.time.values, 3, labels=[1, 2, 3]).astype('int')
    response['Quintile'] = pd.qcut(
        response.RT.values, 5, labels=[1, 2, 3, 4, 5]).astype('int')
    response['Subject'] = subject
    target = target.merge(
        response[['Block', 'TrialGroup', 'Quintile', 'Subject']], 
        left_index=True, right_index=True)
    
    target.set_index('time', inplace=True)
    response.set_index('time', inplace=True)
    print(f'> Average RT: {int(response.RT.mean())}')
    response.to_hdf('../analysis/RTs.h5', key=subject)
    
    # Extract pupil data timelocked to key response
    event_ranges = extract(
            samps, 
            response,  # using the keypress message
            offset=-50, 
            duration=150,
            borrow_attributes=['Block', 'TrialGroup', 'Quintile', 'Subject'])
    
    # Extract baselines
    event_baselines = extract(
            samps, 
            target,  # using the target message
            offset=-50, 
            duration=50,
            borrow_attributes=['Block', 'TrialGroup', 'Quintile', 'Subject']
            ).groupby(level=0).mean()
    
    # Calculate baseline z score
    event_baselines['pz'] =  ((event_baselines.p 
                              - event_baselines.p.mean()) 
                              / event_baselines.p.std())
    
    # Baseline correct the pupil traces
    event_ranges['p_div'] = (event_ranges.p / event_baselines.p - 1).values
    event_ranges['p_sub'] = (event_ranges.p - event_baselines.p).values
    event_ranges['p_pc'] = (event_ranges.p_sub / event_baselines.p * 100).values
    event_ranges['pz'] = (event_ranges.p.sub(
            event_ranges.p.mean())).div(
            event_ranges.p.std())

    # Exclude trials with more than 50 % interpolated data or a blink
    # in the baseline
    event_ranges = reject_bad_trials(event_ranges, interp_thresh=50)
    event_ranges.rename(columns={'reject': 'pct_interp>50'}, inplace=True)
    event_ranges['base_blink'] = event_ranges.index.get_level_values(0).map(
        event_baselines['masked'])
    
    # Save
    event_ranges.to_hdf('../analysis/event_ranges_50hz.h5', key=subject)
    event_baselines.to_hdf('../analysis/event_baselines.h5', key=subject)
