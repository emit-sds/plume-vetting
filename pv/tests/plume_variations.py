#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import yaml

import pv

#
# Somewhat arbitrary-chosen plume of interest (plume 'CH4_PlumeComplex-3727',
# scene 'emit20241130t180334'):
#

# data:
cfg_src = './config.yaml'
plume_data_src = './data/previous_manual_annotation_oneback.json'
cfg = yaml.safe_load(open(cfg_src))

# number of test plume variations:
N = 1000

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

# original plume:
axs.plot(
    np.array(ch4_plume.boundary_pixels.get_coordinates())[:,0],
    np.array(ch4_plume.boundary_pixels.get_coordinates())[:,1],
    linestyle='-', color='yellow')

# random rotation+translation variations:
for i in range(N):
    ch4_plume.new_random_variation()
    axs.plot(
        np.array(ch4_plume.random_variation.get_coordinates())[:,0],
        np.array(ch4_plume.random_variation.get_coordinates())[:,1],
        linestyle='-', color='red')

plt.title(f'{ch4_plume.plume['Plume ID'].iloc[0]}; {ch4_plume.plume['fids'].iloc[0][0]}')
axs.set_xlabel('Acquisition Samples')
axs.set_ylabel('Acquisition Lines')
plt.show()

