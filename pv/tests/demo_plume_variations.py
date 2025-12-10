#!/usr/bin/env python

#
# Test/demonstrate generation of random plume variations and, optionally, show
# plume mask components.
#

import matplotlib.pyplot as plt
import numpy as np
import yaml

import pv

#
# Somewhat arbitrarily-chosen plume of interest (plume 'CH4_PlumeComplex-3727',
# scene 'emit20241130t180334'):
#

# data:
cfg_src = './config.yaml'
plume_data_src = './data/previous_manual_annotation_oneback.json'
cfg = yaml.safe_load(open(cfg_src))

#
# switches that can be used to exercise various code paths:
#

# number of test plume variations:
N = 100
# include combined mask effects (clouds + water + 'no ch4_mf data'):
include_combined_mask = True
#include_combined_mask = False
# include resulting plume mask plots:
#include_plume_mask_plots = True
include_plume_mask_plots = False

# plume instance:
ch4_plume = pv.emit_plume.EMITPlume(
    plume_id='CH4_PlumeComplex-3727',
    plume_data=plume_data_src,
    cfg=cfg)

#
# Plot N in-scene plume variations against original plume:
#

fig, axs = plt.subplots()

# plot plume variations against ch4 mf results:
ch4_mf = pv.emit_file.EMITMatchedFilterFile(
    root=cfg['emit_matched_filter_dataproducts_root'],
    id=ch4_plume.plume['fids'].iloc[0][0],
    type='ch4_mf')
axs.imshow(np.where(ch4_mf.data[:]>cfg['NO_DATA_VALUE'],ch4_mf.data[:],0.))

if include_combined_mask:
    # 'True' for features that should be excluded:
    l2a_mask = pv.emit_file.EMITAcquisitionFile(
        root=cfg['emit_acquisition_dataproducts_root'],
        id=ch4_plume.fid, level='l2a', type = cfg['emit_l2a_mask_type'])
    combined_mask = np.sum(l2a_mask.data[...,:3],axis=-1) > 0                       # clouds, surface water
    combined_mask[np.squeeze(ch4_plume.ch4_mf.data)<=cfg['NO_DATA_VALUE']] = True   # plus missing ch4_mf data
else:
    # 'False' everywhere:
    combined_mask = np.zeros(ch4_mf.data.shape,dtype=bool)

# original plume:
axs.plot(
    ch4_plume.boundary_pixels.get_coordinates().x,
    ch4_plume.boundary_pixels.get_coordinates().y,
    linestyle='-', color='yellow')
if include_plume_mask_plots:
    original_plume_mask = np.squeeze(ch4_plume.mask())

# random rotation+translation variations:
if include_plume_mask_plots:
    all_plume_masks = original_plume_mask & ~combined_mask
for i in range(N):
    ch4_plume.new_random_variation()
    axs.plot(
        ch4_plume.random_variation.get_coordinates().x,
        ch4_plume.random_variation.get_coordinates().y,
        linestyle='-', color='red')
    if include_plume_mask_plots:
        all_plume_masks = all_plume_masks | (np.squeeze(ch4_plume.mask(random_variation=True)) & ~combined_mask)

if include_plume_mask_plots:
    axs.imshow(all_plume_masks)

plt.title(f'{ch4_plume.plume['Plume ID'].iloc[0]}; {ch4_plume.plume['fids'].iloc[0][0]}')
axs.set_xlabel('Acquisition Samples')
axs.set_ylabel('Acquisition Lines')
plt.show()

