#!/usr/bin/env python

import argparse
import functools
import geojson
import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .emit_file import EMITAcquisitionFile, EMITMatchedFilterFile
from .emit_plume import EMITPlume
from . import utils

logging.basicConfig(
    format = '%(levelname)-10s %(asctime)s %(message)s')
log = logging.getLogger('pv')


def create_parser():
    """Application-level argument definitions.

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


def plume_vetting(plume_data=None, plume_id=None, cfg=None,
                  out_inoutplume_file=None, out_spectralmatch_file=None,
                  out_ch4target_basefile=None, log_level=None):
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
    log = logging.getLogger('pv.'+__name__)
    if log_level:
        log.setLevel(log_level)

    # plume instance:
    plume = EMITPlume(plume_id=plume_id, plume_data=plume_data, cfg=cfg)

    log.info("running plume vetting metrics on plume %s, fids %s...",
             plume.plume_id, plume.fids)

    #
    # quantities common to all experiments for this plume:
    #

    # radiance data:
    l1b_radiance = EMITAcquisitionFile(
        root=cfg['emit_acquisition_dataproducts_root'],
        ids=plume.fids, level='l1b',
        type=cfg['emit_l1b_radiance_type'],
        ext='hdr')

    # Obs data:
    l1b_obs = EMITAcquisitionFile(
        root=cfg['emit_acquisition_dataproducts_root'],
        ids=plume.fids, level='l1b',
        type=cfg['emit_l1b_obs_type'],
        ext='hdr')

    # Loc data:
    l1b_loc = EMITAcquisitionFile(
        root=cfg['emit_acquisition_dataproducts_root'],
        ids=plume.fids, level='l1b',
        type=cfg['emit_l1b_loc_type'],
        ext='hdr')

    # ATM data:
    # l2a_atm = EMITAcquisitionFile(
    #     root=cfg['emit_acquisition_dataproducts_root'],
    #     ids=plume.fids, level='l2a',
    #     type=cfg['emit_l2a_atm_type'],
    #     ext='hdr')

    # "combined" mask (clouds + water + 'no ch4_mf data'),
    # 'True' for features that should be excluded:
    l2a_mask = EMITAcquisitionFile(
        root=cfg['emit_acquisition_dataproducts_root'],
        ids=plume.fids, level='l2a',
        type=cfg['emit_l2a_mask_type'], ext='hdr')

    combined_mask = np.sum(l2a_mask.data[:, :, :3], axis=-1) > 0                   # clouds, surface water
    combined_mask[np.squeeze(plume.ch4_mf.data) <= cfg['NO_DATA_VALUE']] = True   # plus missing ch4_mf data

    #
    # original, and shifted plume experiments:
    #

    log.info("performing %d shifted plume experiments (cfg['NUM_PLUME_VARIATIONS'])...",
        cfg['NUM_PLUME_VARIATIONS'])

    results_list = []
    for plume_variation_i in range(cfg['NUM_PLUME_VARIATIONS'] + 1):

        # first plume experiment is with respect to original plume, while
        # subsequent experiments are with respect to randomly shifted plumes:

        if plume_variation_i == 0:
            is_random_variation = False
        else:
            # generate new random (rotation+translation) variation:
            plume.new_random_variation()
            is_random_variation = True

        #
        # "basic" plume mask, to which modifications will be applied (in this,
        # and all subsequent, 'True' for features to include, 'False' for
        # features to exclude):
        #

        plume_mask = np.copy(np.squeeze(plume.mask(
            random_variation=is_random_variation)))

        # apply "combined" mask:
        plume_mask = plume_mask & ~combined_mask

        # "raw" (i.e., for entire plume) min/max/mean metrics:
        in_plume_mf_mean = np.mean(
            np.squeeze(plume.ch4_mf.data)[plume_mask])
        in_plume_mf_max = np.max(
            np.squeeze(plume.ch4_mf.data)[plume_mask])
        in_plume_mf_min = np.min(
            np.squeeze(plume.ch4_mf.data)[plume_mask])

        #
        # determine "optimal" target and background pixel pairs:
        #

        # plume target pixels ('True' for retained target pixels):

        # mask out pixels with MF values greater than N-th percentile:
        nth_percentile = np.percentile(
            np.squeeze(plume.ch4_mf.data)[plume_mask],
            cfg['CH4_MF_EXCLUDE_PERCENTILE'])
        target_pixel_mask = np.logical_and(
            plume_mask, np.squeeze(plume.ch4_mf.data) <= nth_percentile)

        if log.getEffectiveLevel() == logging.DEBUG:
            if plume_variation_i == 0:
                log.debug('plume %s high-level MF metrics: '
                          'variation#/min/max/mean/nth-percentile:', plume.fid)
            log.debug('%d / %f / %f / %f / %f', plume_variation_i,
                      in_plume_mf_min, in_plume_mf_max, in_plume_mf_mean, nth_percentile)

        # reduce target_pixel_mask to NUM_SEED_PIXELS:
        tmp = np.squeeze(plume.ch4_mf.data)[target_pixel_mask]
        tmp.sort()
        seed_threshold = tmp[-cfg['NUM_SEED_PIXELS']]
        target_pixel_mask = np.logical_and(
            target_pixel_mask,
            (np.squeeze(plume.ch4_mf.data)*target_pixel_mask) >= seed_threshold)

        # "blur" pixel mask to form "region" of target pixels:
        dilated_target_pixel_mask = gaussian_filter(
            target_pixel_mask.astype(float),
            sigma=cfg['GAUSSIAN_FILTER_SIGMA']) > cfg['GAUSSIAN_FILTER_RESULTS_THRESHOLD']

        # and reapply combined mask in case Gaussian filtering has reintroduced pixels
        # that should be excluded:
        dilated_target_pixel_mask = dilated_target_pixel_mask & ~combined_mask

        # option to retain only those target pixels corresponding to positive ch4
        # matched filter values:
        if cfg['POSITIVE_TARGET_PIXELS_ONLY']:
            dilated_target_pixel_mask = np.logical_and(
                dilated_target_pixel_mask, np.squeeze(plume.ch4_mf.data) > 0.)

        # background pixels ('True' for candidate background pixels):

        bpe = cfg['BACKGROUND_PAIRING_EXTENTS_IN_PIXELS']       # notational
        mf_delta = cfg['BACKGROUND_PAIRING_CH4_MF_THRESHOLD']   # convenience

        background_pixel_mask = np.zeros(plume_mask.shape, dtype=bool)
        num_lines, num_samples = background_pixel_mask.shape

        # consider background region based on target pixel extents...:
        line_indices, sample_indices = np.nonzero(dilated_target_pixel_mask)
        background_pixel_mask[
            max(0, min(line_indices)-bpe): min(num_lines, max(line_indices)+bpe),
            max(0, min(sample_indices)-bpe): min(num_samples, max(sample_indices)+bpe)] = True

        # as is the case with target pixels, make sure water and missing features are excluded:
        background_pixel_mask = background_pixel_mask & ~combined_mask

        # remove target pixels from consideration:
        background_pixel_mask = background_pixel_mask & ~dilated_target_pixel_mask

        # select pixels with little to no ch4 absorption (i.e., filter according
        # to +/- MF small value range):
        background_pixel_mask = np.logical_and(
            background_pixel_mask,
            np.logical_and(
                np.squeeze(plume.ch4_mf.data) > -mf_delta,
                np.squeeze(plume.ch4_mf.data) < mf_delta))

        # "optimal" target and background pixel pairing based on non-ch4 wavelengths:

        dilated_target_pixel_mask_indices = np.where(dilated_target_pixel_mask)
        background_pixel_mask_indices = np.where(background_pixel_mask)

        wl = np.array(l1b_radiance.hdr['wavelength'], dtype=float)
        ch4_rngs = np.array(cfg['ch4_absorption_ranges'])
        ch4_wl_indices = []
        # accommodate any number of ch4_rngs range pairs:
        for i in range(ch4_rngs.shape[0]):
            ch4_wl_indices.extend(list(np.where((wl >= ch4_rngs[i, 0]) &
                                                (wl <= ch4_rngs[i, 1]))[0]))
        # and, just to be sure, make sure none of the indices are repeated, and
        # sort for convenience:
        ch4_wl_indices = list(set(ch4_wl_indices))
        ch4_wl_indices.sort()
        non_ch4_wl_indices = list(set(range(len(wl))).difference(set(ch4_wl_indices)))
        target_non_ch4_spectra = l1b_radiance.data[dilated_target_pixel_mask][:, non_ch4_wl_indices]
        background_non_ch4_spectra = l1b_radiance.data[background_pixel_mask][:, non_ch4_wl_indices]

        # L1-normalized Euclidian distance spectral similarity matrix
        # (target pixel rows x background pixel columns):
        similarity_matrix = cdist(
            target_non_ch4_spectra, background_non_ch4_spectra, 'euclidean')

        # use optimal assignment to determine non-repeated index pairs (for each
        # target pixel, corresponding "best fit" background pixel, no background
        # pixel used more than once):
        target_row_indices, background_column_indices = linear_sum_assignment(similarity_matrix)

        # for convenience, gather indices, metrics in DataFrame for subsequent
        # operations, analysis:
        similarity_results_df = pd.DataFrame(columns=[
            'similarity_matrix_target_index',
            'similarity_matrix_background_index',
            'similarity_matrix_coefficient',
            'target_pixel_indices',
            'background_pixel_indices',
            'target_mf',
            'background_mf',
            'target_mean_radiance',
            'background_mean_radiance'])
        for idx, (tgt_idx, bg_idx) in enumerate(zip(target_row_indices, background_column_indices)):
            similarity_results_df.loc[tgt_idx] = [
                tgt_idx,
                bg_idx,
                similarity_matrix[tgt_idx,bg_idx],
                (dilated_target_pixel_mask_indices[0][tgt_idx],dilated_target_pixel_mask_indices[1][tgt_idx]),
                (background_pixel_mask_indices[0][bg_idx],background_pixel_mask_indices[1][bg_idx]),
                plume.ch4_mf.data[(dilated_target_pixel_mask_indices[0][tgt_idx], dilated_target_pixel_mask_indices[1][tgt_idx])][0],
                plume.ch4_mf.data[(background_pixel_mask_indices[0][tgt_idx], background_pixel_mask_indices[1][tgt_idx])][0],
                np.mean(l1b_radiance.data[(dilated_target_pixel_mask_indices[0][tgt_idx], dilated_target_pixel_mask_indices[1][tgt_idx]),:]),
                np.mean(l1b_radiance.data[(background_pixel_mask_indices[0][tgt_idx], background_pixel_mask_indices[1][tgt_idx]),:]),
            ]

        # To account for scene inhomogeneity, consider only a defined percentage
        # of pairs having the lowest (i.e., "best") similarity scores:
        sorted_similarity_results_df = similarity_results_df.sort_values(by='similarity_matrix_coefficient')
        sorted_similarity_results_df = sorted_similarity_results_df[
            :int(len(sorted_similarity_results_df)*cfg['SPECTRAL_SIMILARITY_FRACTION_RETAINED'])]

        # Generate plot for showing target and background pixel groups
        if plume_variation_i == 0:
            # Only generating plot for the original plume
            cmf_no_data = float(plume.ch4_mf.hdr['data ignore value'])
            cmf_mask = np.where(plume.ch4_mf.data == cmf_no_data, 1, 0)
            cmf_data = plume.ch4_mf.data / 8000.
            cmf_data = np.where(cmf_mask == 1, 0, cmf_data)
            target_line_indices = [index[0] for index in sorted_similarity_results_df['target_pixel_indices']]
            target_samp_indices = [index[1] for index in sorted_similarity_results_df['target_pixel_indices']]
            background_line_indices = [index[0] for index in sorted_similarity_results_df['background_pixel_indices']]
            background_samp_indices = [index[1] for index in sorted_similarity_results_df['background_pixel_indices']]

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(cmf_data, vmin=0, vmax=0.3, cmap='inferno')

            poly = Polygon(plume.boundary_pixels.geometry.get_coordinates(),
                           closed=True, facecolor='none', edgecolor='red')
            ax.add_patch(poly)
            ax.scatter(target_samp_indices, target_line_indices, s=1, color='red',
                       label='In-plume')
            ax.scatter(background_samp_indices, background_line_indices, s=1,
                       color='green', label='Out-plume')
            plt.legend()
            plt.savefig(out_inoutplume_file, dpi=300)

        # ref. ../../tests/demo_plume_pixel_pairing.ipynb for examples of
        # metrics that could be drawn directly from sorted_similarity_results_df

        # final paired radiances:
        target_radiances = l1b_radiance.data[
            [index[0] for index in sorted_similarity_results_df['target_pixel_indices']],   # line indices
            [index[1] for index in sorted_similarity_results_df['target_pixel_indices']],   # sample indices
            :]
        background_radiances = l1b_radiance.data[
            [index[0] for index in sorted_similarity_results_df['background_pixel_indices']],   # line indices
            [index[1] for index in sorted_similarity_results_df['background_pixel_indices']],   # sample indices
            :]

        target_background_radiance_ratio = np.mean(target_radiances, axis=0) / np.mean(background_radiances, axis=0)

        utils.ghg_process_old.main([
            l1b_radiance.filename, l1b_obs.filename, l1b_loc.filename,
            out_ch4target_basefile
        ])
        ch4_eps = pd.read_csv(out_ch4target_basefile + '_ch4_target',
                              sep='\\s+', names=['index', 'lambda', 'eps'], header=None)
        ch4_eps['eps'] *= -1.
        # since these values have been provided on a slightly different wavelength grid,
        # interpolate to EMIT spectra:
        ch4_eps_interp = np.interp(wl, ch4_eps['lambda'], ch4_eps['eps'])
        # to match eventual dataflow, just overwrite dataframe with ch4
        # interpolated values:
        ch4_eps = ch4_eps_interp

        #
        # generalized plume transmittance model fit:
        #

        ch4_fitting_rngs = np.array(cfg['ch4_fitting_absorption_ranges'])
        ch4_fitting_wl_indices = []
        for i in range(ch4_fitting_rngs.shape[0]):
            ch4_fitting_wl_indices.extend(list(np.where((wl >= ch4_fitting_rngs[i,0]) & (wl <= ch4_fitting_rngs[i,1]))[0]))
        # and, just to be sure, make sure none of the indices are repeated, and
        # sort for convenience:
        ch4_fitting_wl_indices = list(set(ch4_fitting_wl_indices))
        ch4_fitting_wl_indices.sort()

        transmittance_model_fixed_epsilon = functools.partial(
            utils.transmittance_model,epsilon=ch4_eps[ch4_fitting_wl_indices])

        popt, _ = curve_fit(
            transmittance_model_fixed_epsilon,
            wl[ch4_fitting_wl_indices],
            target_background_radiance_ratio[ch4_fitting_wl_indices],
            p0 = [1.] + [0.]*(cfg['TRANSMITTANCE_MODEL_POLYNOMIAL_DEGREE']+1))

        # "Goodness" of fit metrics:
        # what follows is "as implemented" in the original plume vetting code, using the
        # options as described in Chuchu et al. Much could be done to improve the
        # approach, beginning with questioning its rationale.

        (transmittance_model_polynomial, transmittance_model_exponential) = \
            utils.transmittance_model_components(
                wl[ch4_fitting_wl_indices],
                *popt,
                epsilon=ch4_eps[ch4_fitting_wl_indices])

        # estimated target/background radiance ratio, "normalized" by transmittance_model_polynomial:
        normalized_target_background_radiance_ratio = \
            target_background_radiance_ratio[ch4_fitting_wl_indices]/transmittance_model_polynomial

        # modeled target/background radiance ratio, "normalized" by
        # transmittance_model_polynomial (which, by definition, leaves just the 
        # exponential portion of the fit):
        normalized_modeled_target_background_radiance_ratio = transmittance_model_exponential

        dist = np.mean(np.abs(
            normalized_target_background_radiance_ratio -
            normalized_modeled_target_background_radiance_ratio))

        mag = np.mean(np.abs(
            normalized_target_background_radiance_ratio -
            np.mean(normalized_target_background_radiance_ratio)))

        # Xiang, et al. eq. 11:
        normalized_dist = dist/mag

        # Store results
        results_list.append([normalized_dist, popt[0] * 1.e5])

        log.debug('      D_norm: %f   model concentration length (*1e5): %f',
            normalized_dist, popt[0]*1.e5)

        # Generate spectral match plot for plume vetting
        if plume_variation_i == 0:
            # Only generating for the original plume
            fig, axes = plt.subplots(2, 1, constrained_layout=True, squeeze=False)
            axes = axes.flatten()

            wl = wl[ch4_fitting_wl_indices]
            yy = np.polyval(popt[1:], wl)
            y1 = target_background_radiance_ratio[ch4_fitting_wl_indices]
            y2 = y1 * np.exp(popt[0] * ch4_eps[ch4_fitting_wl_indices])
            ax_upper = axes[0]

            ax_upper.plot(wl, yy, label='Continuum function', color='green',
                          linestyle='--', alpha=0.5)
            line1 = ax_upper.plot(wl, y1, label='Measurement')[0]
            line2 = ax_upper.plot(wl, y2, label='Model')[0]
            color1 = line1.get_color()
            color2 = line2.get_color()

            ax_lower = axes[1]
            y1 = y1 / yy
            y2 = y2 / yy
            ax_lower.plot(wl, y1, label='Measurement / Continuum function',
                          color=color1, linestyle='--')
            ax_lower.plot(wl, y2, label='Model / Continuum function',
                          color=color2, linestyle='--')
            ax_lower.set_xlabel('Wavelength')

            handles_upper, labels_upper = ax_upper.get_legend_handles_labels()
            handles_lower, labels_lower = ax_lower.get_legend_handles_labels()
            combined_handles = handles_upper + handles_lower
            combined_labels = labels_upper + labels_lower
            plt.legend(combined_handles, combined_labels, fontsize=8)
            plt.savefig(out_spectralmatch_file, dpi=300)

    log.info("...completed %d shifted plume experiments (cfg['NUM_PLUME_VARIATIONS']).",
        cfg['NUM_PLUME_VARIATIONS'])

    return results_list

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

    if args.plume_ids:
        # specific user-selected plumes:
        plume_ids = args.plume_ids
    else:
        # all plumes:
        plume_ids = list(gpd_plume_data['Plume ID'])

    for plume_id in plume_ids:
        plume_vetting(
            plume_data=gpd_plume_data,
            plume_id=plume_id,
            cfg=cfg,
            out_inoutplume_file=args.out_inoutplume_file,
            out_spectralmatch_file=args.out_spectralmatch_file,
            out_ch4target_basefile=args.out_ch4target_basefile,
            log_level=args.log_level
        )


if __name__ == '__main__':
    main()

