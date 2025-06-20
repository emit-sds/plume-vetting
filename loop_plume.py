#import emit_data
#e = emit_data.EMIT('emit20230424t060859', load_rdn = True, load_baseline_mf=True, load_data = False, methane_name_string='ch4_mf', methane_path='/beegfs/scratch/jfahlen/run_comparison_segment_poly_smoothing_modsigmod_bias_corrected/mod_plume/', load_mf=True)
import numpy as np
import pickle
import scipy.spatial.distance as SSD
import os
from osgeo import gdal
from scipy.interpolate import interp1d
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import ImageGrid
import json
from matplotlib.path import Path
#import gis_utils
import emit_data_c
import sfun
import ghg_process_old
import ghg_process_new
from scipy.optimize import least_squares
from utils import envi_header
from spectral.io import envi
import multiprocessing
import matplotlib.gridspec as gridspec
from shapely.geometry import Polygon, Point
from scipy.stats import percentileofscore
from functools import partial


def process_data(data_per_plume, fids, plume_coords):
        np.random.seed(42)

        ############ Parameters for users to adjust ###############
        num_ite=1000  # The first iteration calculate the original plume, and the rest iterations calculate shifted plumes
        mf_threshold=30 # The range of matched filter values around zero for the background pixels
        num_pts=40 # number of seed pixels with the highest MF values inside the plume
        radius = 200 # radius of the region where background pixels are searched
        sig_flag=1 # 1: consider dependence of absorption coefficient on concentration length, i.e., use local slope of log(radiance); 0: not consider dependence on cl
        average_over_pix_flag=1 # 1: generate one signal using the mean atmospheric/observational conditions of the pixels; 0: generate an individual signal for each pixel based on its condition
        dist_opt=1 # 0: euclidean;  1: euclidean/L1;  2: spectral angle; 3: cosine distance 
        full_flag=0  # 1: use full spectrum to calculate spectral similarity;  0: use reduced spectrum (excluding methane absorption band)
        deg_poly=10  # degree of polynomial for the theoretical model

        ####### Options of making plots and saving data: #######
        plot_plume_mask=0 # 1: highlight all plumes and uniform rows/columns in the scene (the shifted plumes avoid overlapping with them); 0: not plot
        plot_pixels=0 # 1: plot target and background pixels;  0: not plot
        plot_six_panel_figure=1 # 1: plot the figure with six panels;  0: not plot
        save_data_for_plotting=1  #  1: save the data for plotting the six panel figure; 0: not save
        ##############################################################

        pics_folder, results_folder, plot_data_folder = 'pic', 'result', 'plot_data'  # folders to store figures and results
        index, fid, plume_coord, plume_id, r1, r2, cfd, sector, sector_cfd= data_per_plume
        try:
            ii=index
 
            (trans_contour, trans_ratio, trans_fit_sig, trans_simi, trans_coef, trans_avg_top_mf, trans_avg_in_mf, trans_pairs, trans_ac) = ([] for _ in range(9))

            other_plume_coords = [pc for pc, f in zip(plume_coords, fids) if f == fid and pc != plume_coord]  # other plumes in the scene
    
            s = emit_data_c.EMIT(fid, load_rdn=True, load_mf=True, load_rfl=True)
            arg_file=s.run_me(False) 
            af=arg_file.split()
            rdn = s.rdn
            mf = s.mf
            rfl = s.rfl
            wl = s.wavelength_nm
            wl_orig=wl
            mask_fn=s.l2a_mask_filename
            ind = np.where(((np.array(wl) >= 1640) & (np.array(wl) <= 1690)) | ((np.array(wl) >= 2100) & (np.array(wl) <= 2440))) # methane absorption range
            ind_fit= np.where((np.array(wl) >= 2100) & (np.array(wl) <= 2440)) # absorption region used for fitting the measurement to the model
            original_ind = np.arange(len(wl))
            if full_flag==1:
                ind_out=original_ind
            else:
                ind_out = np.delete(original_ind, ind)  # index of wavelength outside the methane absorption region (used to calculate spectral similarity)

            wl= np.array(wl)[ind_fit]

            ######### calculate bad pixel mask ################
            clouds_and_surface_water_mask = None
            if mask_fn is not None:
                clouds_and_surface_water_mask = np.sum(envi.open(envi_header(mask_fn)).open_memmap(interleave='bip')[...,:3],axis=-1) > 0

            uni_row, uni_col = sfun.find_uniform_indices(mf)   # find rows and columns with two or fewer unique MF values   
            rowcol_mask = np.full_like(clouds_and_surface_water_mask, False, dtype=bool)
            rowcol_mask[uni_row, :] = True
            rowcol_mask[:, uni_col] = True
            rowcol_mask[mf <= -9999] = True

            # This mask is used for target pixels. Combine the rowcol_mask with the clouds_and_surface_water_mask 
            combined_mask = clouds_and_surface_water_mask | rowcol_mask

            # This mask is used for background pixels
            background_mask=combined_mask.copy()
            background_mask[(mf < -mf_threshold) | (mf > mf_threshold)] = True
           
            ######### find points inside plume ##############      
            orig_pcontour = Polygon(plume_coord)
            minx, miny, maxx, maxy = orig_pcontour.bounds
            orig_points_inside_plume = []
            for y in range(int(miny), int(maxy) + 1):
                for x in range(int(minx), int(maxx) + 1):
                    if orig_pcontour.contains(Point(x, y)):
                        orig_points_inside_plume.append((x, y)) #  x y coordinate, not row column coordinate
            ###################################################
            
            # calculate the plume mask, so that the shifted plumes doesn't overlap with the plumes in the scene or the uniform row/column
            if num_ite>1:  # calculate shifted plumes
                    plume_mask = np.zeros(mf.shape, dtype=bool)
                    plume_mask[uni_row, :] = True
                    plume_mask[:, uni_col] = True
                    for x, y in orig_points_inside_plume:
                        if 0 <= x < plume_mask.shape[1] and 0 <= y < plume_mask.shape[0]:
                            plume_mask[y, x] = True 
                    for plume in other_plume_coords:
                            pcontour = Polygon(plume)
                            minx, miny, maxx, maxy = pcontour.bounds
                            for y in range(int(miny), int(maxy) + 1):  
                                for x in range(int(minx), int(maxx) + 1):  
                                    if pcontour.contains(Point(x, y)):  
                                        plume_mask[y, x] = True  
                    if plot_plume_mask:
                        sfun.plume_mask_figure(plume_mask, ii, pics_folder)


            ite = 0
            max_attempts = 10000 # maximum number of attempts to shift plumes
            num_attempts=0
            while ite < num_ite and num_attempts<max_attempts:
             num_attempts += 1
             try:
                results=sfun.get_radiance_ratio(wl_orig, radius, num_pts, rdn, mf, plume_coord, orig_points_inside_plume, ind_out, ite, ii, combined_mask, background_mask, plume_mask, dist_opt)
                if results==None:
                    continue
                else:
                    contour_coord, similarity_perpix, ratio,top_ind, top_mf, avg_top_mf, avg_in_plume_mf, top_pairs = results               

            #  get the target signal (absorption coefficient)
                if sig_flag==1:
                    sig, *_=ghg_process_new.main(af, l=top_mf, ind=top_ind, average_over_pix_flag=average_over_pix_flag)
                else:
                    sig=ghg_process_old.main(af)
          
                sig=sig[ind_fit]
                ratio=ratio[ind_fit] # target-to-background radiance ratio          
                coef, fit_sig=sfun.calculate_fit(deg_poly, wl, sig, ratio) # fit the model to measurement
                coef[0] *=1e5  # multiply estimated concentration length with the scaling factor

                if ite==0:  # original plume
                    orig_contour=contour_coord
                    orig_ratio=ratio
                    orig_fit_sig=fit_sig
                    orig_simi=similarity_perpix
                    orig_coef=coef
                    orig_avg_top_mf=avg_top_mf  # mean MF of target points
                    orig_avg_in_mf=avg_in_plume_mf # mean MF of all points inside plume  
                    orig_pairs=top_pairs  
                    orig_ac=sig    # absorption coefficient  
                               
                if ite>0:  # shifted plume
                    trans_contour.append(contour_coord)
                    trans_ratio.append(ratio)
                    trans_fit_sig.append(fit_sig)
                    trans_simi.append(similarity_perpix)   
                    trans_coef.append(coef)
                    trans_avg_top_mf.append(avg_top_mf)
                    trans_avg_in_mf.append(avg_in_plume_mf)
                    trans_pairs.append(top_pairs)  
                    trans_ac.append(sig)                       
             
                if ite==0 and plot_pixels:
                    sfun.target_background_pixels_figure(mf, top_pairs, orig_contour, pics_folder, ii)

                ite += 1   
             except Exception as e:
                print(f"Iteration failed with error: {e}")
            
            trans_dist = []  # normalized distance between measurement and model
            trans_cl=[]     # estimated concentration length from fitted model
            trans_mf=[]    # mean MF value of target points 
            for ratio, sig, mf_t, coef_t in zip(trans_ratio, trans_fit_sig, trans_avg_top_mf, trans_coef):
            # if np.mean(ratio) >= 0:  
                polyn= np.polyval(coef_t[1:], wl)
                ratio=ratio/polyn
                sig=sig/polyn
                dist_t = sfun.calculate_dist(ratio, sig, 0)
                mag_t = sfun.calculate_magnitude(ratio, 1)
                dist_t /= mag_t
                trans_dist.append(dist_t)
                trans_cl.append(coef_t[0])
                trans_mf.append(mf_t)
            index1 = np.argmin(trans_dist)  # find the shifted plume with the smallest normalized distance 
            index2 = np.argmax(trans_mf)  # find the shifted plume with the highest MF value

            orig_cl=orig_coef[0]
            polyn= np.polyval(orig_coef[1:], wl)
            ratio=orig_ratio/polyn
            sig=orig_fit_sig/polyn
            orig_dist = sfun.calculate_dist(ratio, sig, 0)
            orig_mag = sfun.calculate_magnitude(ratio, 1)
            orig_dist /= orig_mag
            perc_dist = percentileofscore(trans_dist, orig_dist)
            perc_cl = percentileofscore(trans_cl, orig_cl)
           

            if plot_six_panel_figure: 
               sfun.six_panel_figure(ii, pics_folder, plot_data_folder, s, mf, orig_contour, trans_contour, index1, index2, 
                                     other_plume_coords, fid, plume_id, r1, r2, cfd, sector, sector_cfd, rfl, rdn, wl_orig, 
                                     wl, ind_fit, orig_pairs, orig_ratio, orig_fit_sig, orig_coef, trans_ratio, trans_fit_sig, 
                                     trans_coef, trans_pairs, orig_dist, perc_dist, orig_cl, perc_cl, save_data_for_plotting)         
            
            all_result = {'fid': fid, 'r12': [r1, r2, cfd], 'perc_dist': perc_dist, 'perc_cl': perc_cl,
                        'o_ratio': orig_ratio, 'o_fit_sig': orig_fit_sig, 'o_simi': orig_simi, 'o_coef': orig_coef, 'o_contour': orig_contour,
                        't_ratio': trans_ratio, 't_fit_sig': trans_fit_sig, 't_simi': trans_simi, 't_coef': trans_coef, 't_contour': trans_contour,
                        'o_avg_top_mf': orig_avg_top_mf,'o_avg_in_mf': orig_avg_in_mf, 'o_dist': orig_dist, 'o_ac': orig_ac,
                        't_avg_top_mf': trans_avg_top_mf,'t_avg_in_mf': trans_avg_in_mf, 't_dist': trans_dist, 't_ac':trans_ac}
            np.save(os.path.join(results_folder, f'i{ii}.npy'), all_result)
            plt.close('all')

        except Exception as e:
            print(f"Error: {e}.")
        return (index, "success", None)  


def main():
    ############ Parameters for users to adjust ###############
    list_ind=0  # it indicates the range of plume indices to process
    loop_all_plumes=1  #   1: use parallel computing to loop over all plumes; 0: calculate specific plumes using plumes_IDs;   
    fid_flag=1 # 1: the plume list already exist, just load them; 0: create it from json file
    num_processes = 4 # number of cores for parallel computing
    #############################################################

    if fid_flag==1: 
        with open('plume_list.pkl', 'rb') as f:
            plume_list = pickle.load(f)
    else: 
        json_filename = '/store/brodrick/methane/ch4_plumedir/previous_manual_annotation_oneback.json'
        j = json.load(open(json_filename))
        plume_list= sfun.load_plume(j)

    if list_ind==0:   # create folders to store results
        sfun.create_folders_for_results()

    if loop_all_plumes==1:
        #np.save(f"{list_ind}.npy", np.array([1]))  # indicate this job is running

        plume_coords, fids, plume_IDs, r1s, r2s, r1_visions, r2_visions, cfds, sectors, sector_cfds = map(plume_list.get, 
        ['plume_coords', 'fids', 'plume_IDs', 'r1s', 'r2s', 'r1_visions', 'r2_visions', 'cfds', 'sectors', 'sector_cfds'])

        fids_length = len(fids)
        interval = 500
        plume_indices = [range(i, min(i + interval, fids_length)) for i in range(0, fids_length, interval)]
        plume_indices = [range(0, len(fids))]

        data_for_processing = [(i, fids[i], plume_coords[i], plume_IDs[i], r1_visions[i], r2_visions[i], cfds[i], sectors[i], sector_cfds[i]) 
                               for i in plume_indices[list_ind]]

        with multiprocessing.Pool(processes=num_processes) as pool:
            partial_func = partial(process_data, fids=fids, plume_coords=plume_coords)
            results = pool.map(partial_func, data_for_processing)

        for result in results:
            index, status, error = result
            if status == "success":
                print(f"Processed index {index}")
            else:
                print(f"Failed for index {index} with error: {error}")


    else:
        plume_IDs = plume_list["plume_IDs"]
        plume_id_list = [plume_IDs[0], plume_IDs[1]]
        json_filename = '/store/brodrick/methane/ch4_plumedir/previous_manual_annotation_oneback.json'
        j = json.load(open(json_filename))
        specific_plumes = sfun.load_plume(j, plume_id_list) # it can also be extracted from plume_list. load_plume() is used here for flexibly extracting additional plume properties
        plume_coords, fids, plume_IDs, r1s, r2s, r1_visions, r2_visions, cfds, sectors, sector_cfds=map(specific_plumes.get, 
        ['plume_coords', 'fids', 'plume_IDs', 'r1s', 'r2s', 'r1_visions', 'r2_visions', 'cfds', 'sectors', 'sector_cfds'])
        data_for_processing_all= [(i, fids[i], plume_coords[i], plume_IDs[i], r1_visions[i], r2_visions[i], cfds[i], sectors[i], sector_cfds[i]) for i in range(0,len(fids))]
        fids = plume_list["fids"]
        plume_coords = plume_list["plume_coords"]
        for data_for_processing in data_for_processing_all:
            results = process_data(data_for_processing, fids, plume_coords)

if __name__ == "__main__":
    main()