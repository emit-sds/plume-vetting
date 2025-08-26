#!/usr/bin/env python

import argparse
import geojson
import geopandas as gpd
import logging
import numpy as np
import yaml

#tmp:
import matplotlib.pyplot as plt

from .emit_file import EMITAcquisitionFile, EMITMatchedFilterFile
from .emit_plume import EMITPlume
from . import utils

logging.basicConfig(
    format = '%(levelname)-10s %(asctime)s %(message)s')
log = logging.getLogger('pv')


def create_parser():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help="""
        (Path and) name of configuration file (yaml format) providing
        parameters and other defaults; see examples.""")
    parser.add_argument('--plume_data', help="""
        (Path and) name of GeoJSON FeatureCollection object data file.""")
    parser.add_argument('--plume_ids', nargs='*', help="""
        --plume_data plume ids of interest, e.g., 'CH4_PlumeComplex-0'
        'CH4_PlumeComplex-1', etc. (default: all)""")
    parser.add_argument('-l','--log', dest='log_level',
        choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
        default='WARNING', help="""
        Set logging level (default: %(default)s).""")
    return parser


def plume_vetting( plume_data=None, plume_id=None, cfg=None, log_level=None):
    """Refactored and updated methane plume vetting implementation, originally
    based on publication, "Identification of false methane plumes for orbital
    imaging spectrometers: A case study with EMIT" by Xiang, et al.

    Args:
        plume_data (geopandas.geodataframe.GeoDataFrame): GeoDataFrame
            containing plume_id.
        plume_id (str): Single plume of interest, identified as one of the
            elements in plume_data['Plume ID'].
        cfg (dict): Parsed plume vetting configuration data (see examples).
        log_level (str): log_level choices per Python logging module
            ('DEBUG','INFO','WARNING','ERROR' or 'CRITICAL'; default='WARNING').

    Returns:
        TODO: Define useful screening/ranking plume vetting metrics

    """
    log = logging.getLogger('edp.'+__name__)
    if log_level:
        log.setLevel(log_level)

    plume = EMITPlume( plume_id=plume_id, plume_data=plume_data, cfg=cfg)
    print(f'type(plume): {type(plume)}')

    log.info("running plume vetting metrics on plume %s, fid %s...",
        plume.plume_id, plume.fid)

    #
    # quantities common to all plume experiments:
    #

    # "combined" mask (clouds + water + 'no ch4_mf data'),
    # 'True' for features that should be excluded:
    l2a_mask = EMITAcquisitionFile(
        root=cfg['emit_acquistion_dataproducts_root'],
        id=plume.fid, level='l2a', type = cfg['emit_l2a_mask_type'])
    combined_mask = np.sum(l2a_mask.data[...,:3],axis=-1) > 0                   # clouds, surface water
    combined_mask[np.squeeze(plume.ch4_mf.data)<=cfg['NO_DATA_VALUE']] = True   # plus missing ch4_mf data
    #plt.imshow(combined_mask)
    #plt.show()


    #
    # original, and shifted plume experiments:
    #

    for plume_variation_i in range(cfg['NUM_PLUME_VARIATIONS']):

        # first plume experiment is with respect to original plume, while
        # subsequent experiments are with respect to randomly shifted plumes:

        if plume_variation_i==0:
            is_random_variation = False
        else:
            is_random_variation = True

        # "basic" plume mask, to which modifications will be applied (note that
        # in this context, 'True' implies data values we intend to keep, 'False'
        # are those to exclude):

        plume_mask = np.copy(np.squeeze(plume.mask(is_random_variation)))

        # apply "combined" mask:
        plume_mask = plume_mask & ~combined_mask

        # "raw" (i.e., for entire plume) min/max/mean metrics:
        in_plume_mf_mean = np.mean(
            np.squeeze(plume.ch4_mf.data)[plume_mask])
        in_plume_mf_max = np.max(
            np.squeeze(plume.ch4_mf.data)[plume_mask])
        in_plume_mf_min = np.min(
            np.squeeze(plume.ch4_mf.data)[plume_mask])

        # exclude points with MF values greater than nth percentile:
        nth_percentile = np.percentile(
            np.squeeze(plume.ch4_mf.data)[plume_mask],
            cfg['CH4_MF_EXCLUDE_PERCENTILE'])

        log.info(f'in-plume matched filter min/max/mean/nth_percentile: {in_plume_mf_min}/{in_plume_mf_max}/{in_plume_mf_mean}/{nth_percentile}')

        # exclude extreme points:
        plume_mask = (plume.ch4_mf.data<nth_percentile) & plume_mask

        # TODO: from this point on...

        # select N points with highest matched filter values:

        # apply pixel dilation:

        # select only those points that have nonzero matched filter values greater than zero:

        # similarity matrix:

        # generate new random (rotation+translation) variation:
        plume.new_random_variation()

#
#   The following is obsolete first-draft code, retained only for (ongoing)
#   migration purposes (most operations are now in file and plume classes):
#
#    for idx,fid in enumerate(gpd_plume_data['fids']):
#
#        #
#        # standard EMIT products:
#        #
#
#        print(f'---------> fid: {fid}')
#
#        l1b_radiance = emit_file.EMITAcquisitionFile(
#            root=cfg['emit_acquistion_dataproducts_root'],
#            id=fid, level='l1b', type = cfg['emit_l1b_radiance_type'])
#        print(f'l1b_radiance.data.shape: {l1b_radiance.data.shape}')
#
#        l1b_glt = emit_file.EMITAcquisitionFile(
#            root=cfg['emit_acquistion_dataproducts_root'],
#            id=fid, level='l1b', type = cfg['emit_l1b_glt_type'])
#        print(f'l1b_glt.data.shape: {l1b_glt.data.shape}')
#
#        l2a_mask = emit_file.EMITAcquisitionFile(
#            root=cfg['emit_acquistion_dataproducts_root'],
#            id=fid, level='l2a', type = cfg['emit_l2a_mask_type'])
#        print(f'l2a_mask.data.shape: {l2a_mask.data.shape}')
#
#        #
#        # methane matched filter results:
#        #
#
#        ch4_mf = emit_file.EMITMatchedFilterFile(
#            root=cfg['emit_matched_filter_dataproducts_root'],
#            id=fid, type='ch4_mf')
#        print(f'ch4_mf.filestr: {ch4_mf.filestr}')
#        print(f'ch4_mf.data.shape: {ch4_mf.data.shape}')
#
#        #
#        # quantities common to each shifted plume experiment:
#        #
#
#        # cloud, surface water, and missing pixel binary mask ('True' == masked):
#        combined_mask = np.sum(l2a_mask.data[...,:3],axis=-1) > 0           # clouds, surface water
#        combined_mask[np.squeeze(ch4_mf.data)<=cfg['NO_DATA_VALUE']] = True # plus missing ch4_mf data
#        print(f'combined_mask.sum(): {combined_mask.sum()}')
#
#        # matched filter background threshold mask:
#        background_mask = combined_mask.copy()
#        background_mask[
#            (np.squeeze(ch4_mf.data) < -cfg['CH4_MF_THRESHOLD']) |
#            (np.squeeze(ch4_mf.data) >  cfg['CH4_MF_THRESHOLD'])  ] = True
#        print(f'background_mask.sum(): {background_mask.sum()}')
#        #plt.imshow(background_mask); plt.colorbar(); plt.show()
#
#        #--> next: plume countours, and pull in mask and plume plot capabilities...
#
#        # for purposes of plume overlap identification, use the l1b geographic
#        # lookup table to convert lat/lon plume definition points to acquisition
#        # file sample/line pixel locations: --see new func def at bottom of this
#        # file; move to a utility later on...


def main():
    """Command-line and Python script entry point.

    """
    parser = create_parser()
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg))

    # repeatability testing:
    if 'RANDOM_SEED' in cfg:
        np.random.seed(cfg['RANDOM_SEED'])

    gpd_plume_data = gpd.GeoDataFrame.from_features(
        geojson.load(open(args.plume_data))['features'])

    if (args.plume_ids):
        # specific user-selected plumes:
        plume_ids = args.plume_ids
    else:
        # all plumes:
        plume_ids = list(gpd_plume_data['Plume ID'])

    for plume_id in plume_ids:
        plume_vetting(
            plume_data=gpd_plume_data, plume_id=plume_id, cfg=cfg, log_level=args.log_level)


if __name__=='__main__':
    main()

