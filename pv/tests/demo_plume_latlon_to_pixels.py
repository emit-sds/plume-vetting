#!/usr/bin/env python

#
# Test/demonstrate conversion of lat/lon plume boundary to pixel coordinates,
# visually confirm via retrieval and orthorectified image plots.
#

import geojson
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

import pv


cfg_src = './config.yaml'
plume_data_src = './data/previous_manual_annotation_oneback.json'

#
# Step 1: Get some test plume data:
#

cfg = yaml.safe_load(open(cfg_src))

# somewhat arbitrary choice:
plume_ids = ['CH4_PlumeComplex-3727']

plume_data = geojson.load(open(plume_data_src))     # geojson.feature.FeatureCollection
gpd_plume_data_all = gpd.GeoDataFrame.from_features(plume_data['features'])
gpd_plume_data_selected = gpd_plume_data_all[       # geopandas.geodataframe.GeoDataFrame
    [plume_id in plume_ids for plume_id in gpd_plume_data_all['Plume ID']]]

# just use the first acquisition file set:
fid = gpd_plume_data_selected['fids'].iloc[0][0]

#
# Step 2: Get associated L1B GLT and (for example) radiance files:
#

l1b_glt = pv.emit_file.EMITAcquisitionFile(
    root=cfg['emit_acquistion_dataproducts_root'],
    id=fid, level='l1b', type = cfg['emit_l1b_glt_type'])
#print(f'l1b_glt.data.shape: {l1b_glt.data.shape}')

l1b_radiance = pv.emit_file.EMITAcquisitionFile(
    root=cfg['emit_acquistion_dataproducts_root'],
    id=fid, level='l1b', type = cfg['emit_l1b_radiance_type'])
#print(f'l1b_radiance.data.shape: {l1b_radiance.data.shape}')

#
# Step 3: Convert plume latlon coordinates to L1B GLT coordinates, which are
# then used to look up L1B acquisition product(s) pixels:
#

l1b_glt_xy_pixels, l1b_line_sample_lookup = pv.utils.geo_to_pix(
    gpd_plume_data_selected['geometry'].get_coordinates(),l1b_glt)

#
# Step 4: Plot, along with acquisition *.png files for reference, and to verify
# plume outlines:
#

fig, axs = plt.subplots(2,2)

fig.suptitle(
    f'{gpd_plume_data_selected["Plume ID"].iloc[0]}, ({" ".join(gpd_plume_data_selected["fids"].iloc[0])})')

[rad_png,ortho_png] = sorted(glob.glob(l1b_radiance.filestr+'png'))
axs[0,0].imshow(plt.imread(ortho_png))
axs[0,0].set_title('L1B Radiance, Orthographic Projection')
axs[0,1].imshow(plt.imread(rad_png))
axs[0,1].set_title('L1B Radiance, Pixel Space')

# plot plume in xy pixel space against nonzero lookup table terms:
im0 = axs[1,0].imshow((l1b_glt.data[:,:,0]>0) & (l1b_glt.data[:,:,1]>0))
axs[1,0].plot(l1b_glt_xy_pixels[:,0],l1b_glt_xy_pixels[:,1],linestyle='-', color='red')
fig.colorbar(im0,ax=axs[1,0])
axs[1,0].set_title('Plume boundary, Orthographic Projection')

# plot plume in emit acquisition space where x->sample and y->line, against
# methane wavelengths:
l1b_wavelengths = np.array(l1b_radiance.hdr['wavelength']).astype('float')
ch4_wavelengths = ((l1b_wavelengths>1640) & (l1b_wavelengths<1690))
#ch4_wavelengths = ((l1b_wavelengths>1640) & (l1b_wavelengths<1690)) | ((l1b_wavelengths>2100) & (l1b_wavelengths<2440))
im1 = axs[1,1].imshow(
    l1b_radiance.data[:,:,ch4_wavelengths].sum(axis=-1))
axs[1,1].plot(l1b_line_sample_lookup[:,1],l1b_line_sample_lookup[:,0],linestyle='-', color='red')
fig.colorbar(im1,ax=axs[1,1])
axs[1,1].set_title('Plume boundary, Pixel Space')

plt.tight_layout()
plt.show()

