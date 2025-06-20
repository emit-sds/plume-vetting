import os 
import glob
from functools import partial
import shutil
import time
import pickle
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.interpolate import interp1d
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.path import Path
from scipy.optimize import least_squares
from osgeo import gdal
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.optimize as opt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
#from sklearn.metrics import mean_squared_error

def create_folders_for_results():
    frames_folder = 'pic'
    results_folder = 'result'
    plot_data_folder = 'plot_data'
    
    for folder in [results_folder, frames_folder, plot_data_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        while os.path.exists(folder):
            time.sleep(0.1)
        os.makedirs(folder)

    # for i in range(11):
    #     file_name = f"{i}.npy"
    #     if os.path.exists(file_name):
    #         os.remove(file_name)


def spectral_angle(spec1, spec2):
    dot_product = np.dot(spec1, spec2)
    norm_spec1 = np.linalg.norm(spec1)
    norm_spec2 = np.linalg.norm(spec2)
    return np.arccos(dot_product / (norm_spec1 * norm_spec2))


def fit_exponential(wavelengths, epsilon, *coeffs, ac):
    L0 = np.polyval(coeffs, wavelengths)
    return L0 * np.exp(epsilon * ac)

def fit_exponential2(wavelengths, epsilon, *coeffs, ac):
    L0 = np.polyval(coeffs, wavelengths)
    L0[L0 == 0] = 1e-30
    return L0 * np.exp(epsilon * ac)

def calculate_fit(degree, wl, ac, ratio):
    epsilon_initial = 0.01  
    poly_initial = [1.0] + [0.0] * degree  
    initial_guess = [epsilon_initial] + poly_initial

    lower_bounds = [0] + [-np.inf] * (degree + 1)  
    upper_bounds = [1] + [np.inf] * (degree + 1)  
    
    # use partial to fix ac while calling fit_exponential
    fit_func = partial(fit_exponential, ac=ac)

    popt, _ = opt.curve_fit(fit_func, wl, ratio, p0=initial_guess, bounds=(lower_bounds, upper_bounds))
    optimized_epsilon = popt[0]
    optimized_coeffs = popt[1:]  
    
    poly = np.polyval(optimized_coeffs, wl)
    fit_sig = poly * np.exp(optimized_epsilon * ac)
    
    return popt, fit_sig

# find the best polynomial degree
def calculate_best_fit(wl, ac, ratio, max_degree=25):
    best_degree = None
    best_fit_sig = None
    best_popt = None
    min_mse = np.inf 

    for degree in range(1, max_degree + 1):    
        initial_guess = [0.1] + [1.0] * (degree + 1)  
        lower_bounds = [0] + [-np.inf] * (degree + 1)  
        upper_bounds = [np.inf] * (degree + 2)
        
        fit_func = partial(fit_exponential2, ac=ac)
        try:
            popt, _ = opt.curve_fit(fit_func, wl, ratio, p0=initial_guess, bounds=(lower_bounds, upper_bounds))
            optimized_epsilon = popt[0]
            optimized_coeffs = popt[1:]  
            
            poly = np.polyval(optimized_coeffs, wl)
            fit_sig = poly * np.exp(optimized_epsilon * ac)
            
            mse = np.mean((ratio - fit_sig) ** 2)
            if mse < min_mse:
                min_mse = mse
                best_degree = degree
                best_fit_sig = fit_sig
                best_popt = popt
        
        except Exception as e:
            print(f"Error with degree {degree}: {e}")
            continue  

    return best_degree, best_popt, best_fit_sig


def get_neighboring_points(top_points, square_size, point_inside_plume, remove_center_flag):
    all_neighboring_points = []
    for point in top_points:
        i, j = point
        half_size = square_size // 2
        neighboring_points = [(i + di, j + dj) for di in range(-half_size, half_size + 1) for dj in range(-half_size, half_size + 1)]
        if remove_center_flag==1:
         neighboring_points.remove(point)
        all_neighboring_points.extend(neighboring_points)
    
    all_neighboring_points = list(set(all_neighboring_points)) # remove duplicates
    all_neighboring_points = [(ni, nj) for ni, nj in all_neighboring_points if (ni, nj) in point_inside_plume]

    return all_neighboring_points


def get_radiance_ratio(wl, radius, num_pts, rdn, mf, orig_plume_coord, orig_points_inside_plume, ind, ite, ii, combined_mask, background_mask, plume_mask, dist_opt):

    ###########################################################
    extreme_pts_flag=0 # 1: remove pixels with highest MF values
    dilate_flag=1 # 1: dilate around seed pixels
    pos_flag=0  # 1: exclude target pixels with non-positive MF values
    remove_center_flag=0 # 1: remove seed pixels
    use_all_pts_flag=0 # 1: use all pixels inside plume as target pixels
    allow_repetition=0 # 1: one background spectrum can be paired with multiple target spectra; 0: unique background spectrum for each target spectrum
    ###########################################################

    background_mask=background_mask.copy()

    # ite==0: original plume; ite>0: shifted plumes
    if ite==0:
        points_inside_plume=orig_points_inside_plume
        plume_coord=orig_plume_coord
    else:        
        rotation_angle = np.random.uniform(0, 2 * np.pi)
        rotation_center = find_center(orig_points_inside_plume)  
        rotated_orig_points_inside_plume = rotate_points(orig_points_inside_plume, rotation_center, rotation_angle)  #rotate points inside plume
        rotated_orig_plume_coord = rotate_points(orig_plume_coord, rotation_center, rotation_angle)                  #rotate plume contour
        translated_pts, translated_plume_coord= find_translated_pts(rotated_orig_points_inside_plume, rotated_orig_plume_coord, plume_mask) # translate them   
        if translated_pts is None:
            return None
        else:
            points_inside_plume=translated_pts
            plume_coord=translated_plume_coord

    points_inside_plume = [(int(x), int(y)) for x, y in points_inside_plume]  # convert all points to integer tuples 
    points_inside_plume = [point for point in points_inside_plume if not combined_mask[point[1], point[0]]]  # exclude bad pixels

    in_plume_mf = [mf[point[1], point[0]] for point in points_inside_plume]
    avg_in_plume_mf = np.mean([v for v in in_plume_mf if v != -9999])

    if extreme_pts_flag==1: # exclude points with MF values greater than or equal to the 99th percentile
        mf_values = [mf[point[1], point[0]] for point in points_inside_plume]
        mf_highest = np.percentile(mf_values, 99)
        print("high", mf_highest)
        points_inside_plume = [point for point in points_inside_plume if mf[point[1], point[0]] < mf_highest]

    # select the top num_pts points with the highest MF values 
    sorted_points = sorted(points_inside_plume, key=lambda point: mf[point[1], point[0]], reverse=True)
    num_pts=min(num_pts, len(points_inside_plume))
    if use_all_pts_flag==1:
        num_pts=len(points_inside_plume)
    top_points = sorted_points[:num_pts]

    if dilate_flag==1:  # dilate pixels
       square_size=3
       top_points = get_neighboring_points(top_points, square_size, points_inside_plume, remove_center_flag)

    if pos_flag==1:
       top_points = [point for point in top_points if mf[point[1], point[0]] > 0]

    top_points = [(y, x) for x, y in top_points] # swap x y to so that it is (row, col)
    if not top_points:
       return None
    
    # define bounds for the region to search for background pixels
    x_coords, y_coords = zip(*top_points)
    x_coords, y_coords = np.array(x_coords), np.array(y_coords)
    min_x = max(min(x_coords) - radius, 0)
    max_x = min(max(x_coords) + radius, rdn.shape[0] - 1)
    min_y = max(min(y_coords) - radius, 0)
    max_y = min(max(y_coords) + radius, rdn.shape[1] - 1)

    A_data = rdn[x_coords, y_coords, :]
    A_data=A_data[:,ind]  # target spectra outside methane absorption region

    B_data_full = rdn[min_x:max_x+1, min_y:max_y+1, :]
    B_data_full=B_data_full[:,:,ind]  #candidate background spectra outside methane absorption region

    background_mask[x_coords, y_coords] = True # mask out target pixels from background pixels
    condition_mask=background_mask[min_x:max_x+1, min_y:max_y+1]
    valid_indices = np.where(~condition_mask) # select valid background pixels
    B_data = B_data_full[valid_indices[0], valid_indices[1], :]

    # each row of similarity_matrix corresponds to a target pixel; each column corresponds to a background pixel
    if dist_opt==0:
       similarity_matrix = cdist(A_data, B_data, 'euclidean') # smaller value means higher similarity
    elif dist_opt==1:
        A_data_normalized = A_data / np.sum(np.abs(A_data), axis=1, keepdims=True)
        B_data_normalized = B_data / np.sum(np.abs(B_data), axis=1, keepdims=True)  # L1 normalized
        similarity_matrix = cdist(A_data_normalized, B_data_normalized, metric='euclidean')
    elif dist_opt == 2:
        A_norm = A_data / np.linalg.norm(A_data, axis=1, keepdims=True)
        B_norm = B_data / np.linalg.norm(B_data, axis=1, keepdims=True) # spectral angle 
        similarity_matrix = np.arccos(np.clip(np.dot(A_norm, B_norm.T), -1.0, 1.0))  # smaller value means higher similarity
    elif dist_opt==3:
        A_norm = A_data / np.linalg.norm(A_data, axis=1, keepdims=True)
        B_norm = B_data / np.linalg.norm(B_data, axis=1, keepdims=True)
        similarity_matrix = 1-np.dot(A_norm, B_norm.T)  # smaller value means higher similarity

    if allow_repetition==0:
       row_ind, col_ind = linear_sum_assignment(similarity_matrix) # find the optimal pairs that minimize global cost (highest similarity)
    else:
        row_ind = np.arange(similarity_matrix.shape[0]) # row indices
        col_ind = np.argmin(similarity_matrix, axis=1) # index of the smallest element in each row

    original_B_indices = [(valid_indices[0][i] + min_x, valid_indices[1][i] + min_y) for i in col_ind] # convert back to original index

    # each element in top_pairs has three components: (x, y) of target pixel, (x ,y) of background pixel, similarity score
    top_pairs = [[top_points[i], original_B_indices[i], similarity_matrix[i, col_ind[i]]] for i in row_ind]

    #avg_similarity = sum(similarity for _, _, similarity in top_pairs) / len(top_pairs) if top_pairs else 0 
    all_similarity = [similarity for _, _, similarity in top_pairs] if top_pairs else []

    top_pairs.sort(key=lambda x: x[2]) 
    num_pairs_to_select = int(0.5 * len(top_pairs))
    top_pairs = top_pairs[:num_pairs_to_select]   # choose smallest 50% 

    top_mf = [mf[i, j] for (i, j), _, _, in top_pairs] # MF of target pixels 
    avg_top_mf = np.mean([v for v in top_mf if v != -9999])

    top_rdns = [rdn[i, j, :] for (i, j), _, _, in top_pairs] # radiance of target pixels
    avg_top_rdn = np.mean(top_rdns, axis=0)

    low_rdns = [rdn[i, j, :] for _, (i, j), _, in top_pairs] # radiance of background pixels
    avg_low_rdn = np.mean(low_rdns, axis=0)

    ratio=avg_top_rdn/avg_low_rdn # target-to-background radiance ratio

    top_ind = [(i, j) for (i, j), _, _ in top_pairs] # indices of target pixels
    
    save_data=0  
    if save_data==1: # save data to csv file
       low_mf = [mf[i, j] for _, (i, j), _, in top_pairs] # MF of background pixels
       save_to_csv(wl, top_rdns, low_rdns, top_mf, low_mf, top_pairs, ii)

    results = (plume_coord, all_similarity, ratio, top_ind, top_mf, avg_top_mf, avg_in_plume_mf, top_pairs)
    return results 
    


def find_translated_pts(pts_inside_contour, pcontour, mask, max_iterations=100):
    iterations = 0
    pts_inside_contour = np.array(pts_inside_contour)
    while iterations < max_iterations:
        max_trans_x = mask.shape[1] - max(pts_inside_contour[:, 0])
        min_trans_x = -min(pts_inside_contour[:, 0])
        max_trans_y = mask.shape[0] - max(pts_inside_contour[:, 1])
        min_trans_y = -min(pts_inside_contour[:, 1])

        translation_vector = (
            np.random.randint(min_trans_x, max_trans_x), 
            np.random.randint(min_trans_y, max_trans_y))

        translated_pts = [(x + translation_vector[0], y + translation_vector[1]) for x, y in pts_inside_contour]
        translated_contour = [(x + translation_vector[0], y + translation_vector[1]) for x, y in pcontour]
        translated_pts_array = np.array(translated_pts).astype(int)
        
        valid_x = (0 <= translated_pts_array[:, 0]) & (translated_pts_array[:, 0] < mask.shape[1])
        valid_y = (0 <= translated_pts_array[:, 1]) & (translated_pts_array[:, 1] < mask.shape[0])
        valid_pts = translated_pts_array[valid_x & valid_y]

        if not np.any(mask[valid_pts[:, 1], valid_pts[:, 0]]):  # check if any of the translated points are within plumes or uniform row/columns in the scene 
           # mask[valid_pts[:, 1], valid_pts[:, 0]] = True  # mask out translated points to avoid overlapping of shifted plumes
            return translated_pts, translated_contour
        iterations += 1

    print("Error: unable to find translation vector after", max_iterations, "iterations")
    return None, None



def load_plume(j, plume_id_list=None):
    import pickle
    plumes = []
    for index, feature in enumerate(j['features']):
        try:
            if feature['geometry']['type'] == 'Polygon':
                properties = feature['properties']
                plume_id = properties.get('Plume ID', 'No ID')
                if plume_id_list is not None and plume_id not in plume_id_list:
                    continue  # skip this plume if its ID is not in the list
                scene_fid = properties['fids'][0]
                plume_coords = feature['geometry']['coordinates'][0]
                plume_coords_pix = calculate_plume_coords_pix(scene_fid, plume_coords)
                plume_coords_pix = [(y, x) for (x, y) in plume_coords_pix]
                plumes.append({
                'fid': scene_fid,
                'coords': plume_coords_pix,
                'R1': properties.get('R1 - Reviewed', 'Unknown'),
                'R2': properties.get('R2 - Reviewed', 'Unknown'),
                'R1_VISIONS': properties.get('R1 - VISIONS', 'Unknown'),
                'R2_VISIONS': properties.get('R2 - VISIONS', 'Unknown'),
                'Confidence': properties.get('Confidence', 'Unknown'),
                'Plume ID': plume_id,
                'Sector': properties.get('Sector', 'Unknown'),
                'Sector Confidence': properties.get('Sector Confidence', 'Unknown').strip()})        
                print(f"Processed Plume {index}: {plumes[-1]['R1']}, {plumes[-1]['R2']}")
        except Exception as e:
            print(f"Error with plume {index}: {e}.")          
            continue
    plume_list = {
    'plume_coords': np.array([p['coords'] for p in plumes], dtype=object),
    'fids': [p['fid'] for p in plumes],
    'plume_IDs': [p['Plume ID'] for p in plumes],
    'r1s': [p['R1'] for p in plumes],
    'r2s': [p['R2'] for p in plumes],
    'r1_visions': [p['R1_VISIONS'] for p in plumes],
    'r2_visions': [p['R2_VISIONS'] for p in plumes],
    'cfds': [p['Confidence'] for p in plumes],
    'sectors': [p['Sector'] for p in plumes],
    'sector_cfds': [p['Sector Confidence'] for p in plumes]}

    if plume_id_list is None:
        save_path = '/store/xiang/dat/plume_list.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(plume_list, f)
    
    return plume_list


def find_filename(name_string, level, tag, ext = '.img'):
   basepath = '/beegfs/store/emit/ops/data/acquisitions/'
   run_folder = basepath + f'{name_string[4:12]}/{name_string}' + '/'
   f = glob.glob(run_folder + f'{level}/*{tag}*{ext}')
   #print(self.run_folder)
   if len(f) != 1:
      print('no file')
   return f[0]

def calculate_plume_coords_pix(scene_fid, plume_coords):
    glt_filename=find_filename(scene_fid, 'l1b', 'glt')
    glt_ds = gdal.Open(glt_filename)
    trans = glt_ds.GetGeoTransform()

    plume_coords_pix = []
    for c in plume_coords:
        glt_ypx = int(round((c[1] - trans[3]) / trans[5]))
        glt_xpx = int(round((c[0] - trans[0]) / trans[1]))
        glt = np.squeeze(glt_ds.ReadAsArray(glt_xpx, glt_ypx, 1, 1))
        if glt is not None:
            if isinstance(glt, np.ndarray):
                if glt.size > 1:
                    if glt[0] != 0:
                        try:
                            plume_coords_pix.append((glt[1], glt[0]))
                        except Exception as e:
                            pdb.set_trace()
    return plume_coords_pix


def find_center(points):
    num_points = len(points)
    if num_points == 0:
        return (0, 0)

    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)

    center_x = sum_x / num_points
    center_y = sum_y / num_points

    return (center_x, center_y)


def rotate_points(points, center, angle):
    rotated_points = []
    for point in points:
        # translate plume to the origin for rotation
        translated_point = (point[0] - center[0], point[1] - center[1])
        rotated_x = translated_point[0] * np.cos(angle) - translated_point[1] * np.sin(angle)
        rotated_y = translated_point[0] * np.sin(angle) + translated_point[1] * np.cos(angle)
        # translate plume back
        rotated_point = (rotated_x + center[0], rotated_y + center[1])
        rotated_points.append(rotated_point)
    return rotated_points


def find_uniform_indices(matrix):
    # check if an array has two or fewer unique values
    def has_two_or_fewer_unique_values(arr):
        return len(np.unique(arr)) <= 2

    uniform_rows = []
    for i in range(matrix.shape[0]):
        if has_two_or_fewer_unique_values(matrix[i, :]):
            uniform_rows.append(i)

    uniform_columns = []
    for j in range(matrix.shape[1]):
        if has_two_or_fewer_unique_values(matrix[:, j]):
            uniform_columns.append(j)

    return uniform_rows, uniform_columns

def calculate_magnitude(ratio, mag_opt):
    if mag_opt == 0:
        mag = 1
    elif mag_opt == 1:
        mag = np.mean(np.abs(ratio - np.mean(ratio))) # mean absolute deviation from the mean of the ratio.
    elif mag_opt == 2:
        mag = np.linalg.norm(ratio)
    elif mag_opt == 3:
        mag = np.mean(np.abs(ratio))
    elif mag_opt == 4:
        mag = max(ratio)-min(ratio) 
    elif mag_opt == 5:
        mag = np.std(ratio)      
    return mag


def calculate_dist(ratio, sig, dist_opt):
    if dist_opt==0:
        dist=np.mean(np.abs(ratio - sig))
    if dist_opt==1:
        if np.std(ratio) == 0 or np.std(sig) == 0:
            dist=-1  # avoid division by zero 
        correlation_coefficient = np.corrcoef(ratio, sig)[0, 1]
        dist=1 - correlation_coefficient**2
    return dist


def save_to_csv(wl, top_rdns, low_rdns, top_mf, low_mf, top_pairs, ii):
        num_pixel_pairs = len(top_pairs)
        matrix = np.zeros((len(wl) + 1, 1 + 2 * num_pixel_pairs))
        matrix[1:, 0] = wl  
        for index, (top, low, t_mf, l_mf) in enumerate(zip(top_rdns, low_rdns, top_mf, low_mf)):
            matrix[0, 1 + 2*index] = t_mf  # place MF at the top of the column
            matrix[0, 2 + 2*index] = l_mf  
            matrix[1:, 1 + 2*index] = top  # place radiance data below MF
            matrix[1:, 2 + 2*index] = low  
        np.savetxt(f'{ii}.csv', matrix, delimiter=",", fmt='%g')
        print(f'Data saved to {ii}.csv')


def target_background_pixels_figure(mf, top_pairs, orig_contour, frames_folder, ii):
    fig3, ax = plt.subplots(figsize=(8, 6))  
    label_size = 14
    marker_size=3
    vmax_99th_perc = np.percentile(mf, 99)
    mf_display = ax.imshow(mf, vmin=0, vmax=vmax_99th_perc, cmap='inferno', origin='upper')
    # plot the original plume contour
    ax.plot(*zip(*orig_contour), color='r', label='Original plume')
    # plot the target pixels in red
    for (i, j), _, _ in top_pairs:
        ax.plot(j, i, 'ro', markersize=marker_size, label='Target Pixel')
    # plot the background pixels in green
    for _, (i, j), _ in top_pairs:
        ax.plot(j, i, 'go', markersize=marker_size, label='Background Pixel')

    x_min = min(min(j for (i, j), _, _ in top_pairs), min(j for _, (i, j), _ in top_pairs)) 
    x_max = max(max(j for (i, j), _, _ in top_pairs), max(j for _, (i, j), _ in top_pairs)) 
    y_min = min(min(i for (i, j), _, _ in top_pairs), min(i for _, (i, j), _ in top_pairs)) 
    y_max = max(max(i for (i, j), _, _ in top_pairs), max(i for _, (i, j), _ in top_pairs)) 

    ax.set_xlim(x_min - 10, x_max + 10)
    ax.set_ylim(y_min - 10, y_max + 10)
    ax.invert_yaxis()  # invert the y axis so that it increases downward
    fig3.colorbar(mf_display, ax=ax, label='Matched filter values')
    fig3.savefig(os.path.join(frames_folder, f'pix_{ii}.png'), dpi=300)


def save_plot_data(plot_data_folder, ii, mf, temp, orig_contour, other_plume_coords, rfl_mean, rfl_std, rdn_mean_target, 
                     rdn_std_target, rdn_mean_background, rdn_std_background, fid, plume_id, r1, r2, cfd, sector, sector_cfd):
    plot_data = {
        'mf': mf,
        'rgb': temp,
        'plume_coord': orig_contour,
        'other_plume_coord': other_plume_coords,
        'rfl_mean': rfl_mean,
        'rfl_std': rfl_std,
        'rdn_mean_target': rdn_mean_target,
        'rdn_std_target': rdn_std_target,
        'rdn_mean_background': rdn_mean_background,
        'rdn_std_background': rdn_std_background,
        'fid': fid,
        'plume_id': plume_id,
        'R1': r1,
        'R2': r2,
        'Confidence': cfd,
        'Sector': sector,
        'Sector confidence': sector_cfd}

    file_path = os.path.join(plot_data_folder, f'p{ii}.pkl')
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(plot_data, file)
        
    except Exception as e:
        print(f"Failed to save data: {e}")  

def plume_mask_figure(plume_mask, ii, pics_folder):
    cmap = plt.cm.colors.ListedColormap(['black', 'red'])
    plt.figure(figsize=(8, 6))
    plt.imshow(plume_mask, cmap=cmap)  
    plt.colorbar(label='Plume Presence', ticks=[0, 1])  
    plt.title("Plume Mask")
    plt.savefig(os.path.join(pics_folder, f'plumemask_{ii}.png'), dpi=300)
    plt.close()



def six_panel_figure(ii, pics_folder, plot_data_folder, s, mf, orig_contour, trans_contour,
                           index1, index2, other_plume_coords, fid, plume_id, r1, r2, cfd,
                           sector, sector_cfd, rfl, rdn, wl_orig, wl, ind_fit, orig_pairs,
                           orig_ratio, orig_fit_sig, orig_coef,
                           trans_ratio, trans_fit_sig, trans_coef, trans_pairs,
                           orig_dist, perc_dist, orig_cl, perc_cl, save_data_for_plotting):
       
    fig1 = plt.figure(figsize=(20, 15), dpi=300)
    outer_grid = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    label_size=14

    # panel 1: MF map
    ax = fig1.add_subplot(outer_grid[0, 0])
    vmax_99th_perc = np.percentile(mf, 99)
    mf_display = ax.imshow(mf, vmin = 0, vmax = vmax_99th_perc, cmap = 'inferno')

    ax.plot(*zip(*orig_contour), color='r', label='Original plume')
    ax.plot(*zip(*trans_contour[index1]), color='g', label='Shifted plume wt smallest norm dist')
    ax.plot(*zip(*trans_contour[index2]), color='m', label='Shifted plume wt highest MF')      

    if other_plume_coords:
        for j in range(0, len(other_plume_coords)):                       
            other_plume_coord =  other_plume_coords[j]
            if j == 0:
                ax.plot(*zip(*other_plume_coord), color='b', label='Other plumes')
            else:
                ax.plot(*zip(*other_plume_coord), color='b')

    cbar = fig1.colorbar(mf_display, ax=ax)
    cbar.set_label('Matched filter values')
    ax.legend(bbox_to_anchor=(2.25, 1.65), loc='upper center', fontsize=14)
    ax.annotate(
        f'{fid}\n'
        f'{plume_id}\n'
        f'R1_VISIONS: {r1}\n'
        f'R2_VISIONS: {r2}\n'
        f'Confidence: {cfd}\n'
        f'Sector: {sector}\n'
        f'Sector confidence: {sector_cfd}',
        xy=(0.5, 1),  # center-left of the axes
        xycoords='axes fraction',
        textcoords="offset points",
        xytext=(-80, 80),  # rightward offset from the edge
        fontsize=14,
        ha='left',  
        va='center',  
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='none'),
        annotation_clip=False)

    #panel 2: RGB 
    ax = fig1.add_subplot(outer_grid[0, 1])
    rgb_rfl= s.get_rgb_rfl(r = 35)  
    temp = rgb_rfl.copy()
    ax.imshow(temp)
    for pcoord in trans_contour:                        
            ax.plot(*zip(*pcoord), color='black')
    ax.plot(*zip(*orig_contour), color='r', label='Original plume')
    ax.plot(*zip(*trans_contour[index1]), color='g', label='Shifted plume wt smallest norm dist')
    ax.plot(*zip(*trans_contour[index2]), color='m', label='Shifted plume wt highest MF')
    if other_plume_coords:
        for j in range(0, len(other_plume_coords)):  
            other_plume_coord = other_plume_coords[j]
            if j == 0:
                ax.plot(*zip(*other_plume_coord), color='b', label='Other plumes')
            else:
                ax.plot(*zip(*other_plume_coord), color='b')

    # panel 3: reflectance and radiance
    ax = fig1.add_subplot(outer_grid[0, 2])
    top_pairs=orig_pairs
    top_rfl = [rfl[i, j, :] for (i, j), _, _, in top_pairs]
    data=top_rfl
    rfl_mean = np.mean(data, axis=0)
    rfl_std = np.std(data, axis=0)
    wavelengths = wl_orig[ind_fit]
    rfl_std=rfl_std[ind_fit]
    rfl_mean=rfl_mean[ind_fit]
    ax.plot(wavelengths, rfl_mean, color='black', label='Mean Reflectance of target')
    ax.fill_between(wavelengths, rfl_mean - rfl_std, rfl_mean + rfl_std, color='gray', alpha=0.3, label='Standard deviation of rfl')
    ax.set_xlabel('Wavelength', fontsize=label_size)
    ax.set_ylabel('Reflectance', fontsize=label_size)

    ax.grid(True)
    ax2 = ax.twinx()  # create axes that shares the x-axis
    top_rdns = [rdn[i, j, :] for (i, j), _, _, in top_pairs]
    rdn_mean_target = np.mean(top_rdns, axis=0)
    rdn_std_target = np.std(top_rdns, axis=0)
    rdn_std_target=rdn_std_target[ind_fit]
    rdn_mean_target=rdn_mean_target[ind_fit]
    ax2.plot(wavelengths, rdn_mean_target, color='hotpink', label='Mean radiance of target')
    ax2.set_ylabel('Radiance', fontsize=label_size)

    low_rdns = [rdn[i, j, :] for _, (i, j), _, in top_pairs]
    rdn_mean_background= np.mean(low_rdns, axis=0)
    rdn_std_background = np.std(low_rdns, axis=0)
    rdn_mean_background=rdn_mean_background[ind_fit]
    rdn_std_background=rdn_std_background[ind_fit]
    ax2.plot(wavelengths, rdn_mean_background, color='blue', label='Mean radiance of background')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines + lines2
    all_labels = labels + labels2
    ax.legend(all_lines, all_labels, loc='lower left', bbox_to_anchor=(0.02, 0.02))
    ax.set_title('Original plume', fontsize=label_size)
    ax.annotate(
        f'Normalized distance: {orig_dist:.3f}\n'
        f'Normalized distrance percentile: {perc_dist:.1f}%\n'
        f'Concentration length: {orig_cl:.0f}\n'
        f'Concentration length percentile: {perc_cl:.1f}%',
        xy=(0.5, 1),  
        xycoords='axes fraction',
        textcoords="offset points",
        xytext=(-100, 60),  
        fontsize=14,
        ha='left',  
        va='center', 
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='none'),
        annotation_clip=False)
    
    if save_data_for_plotting:
        save_plot_data(plot_data_folder, ii, mf, temp, orig_contour, other_plume_coords, rfl_mean, rfl_std, rdn_mean_target, 
            rdn_std_target, rdn_mean_background, rdn_std_background, fid, plume_id, r1, r2, cfd, sector, sector_cfd)

                
    # panels 4  5  6: model versus measuement for original plume and two shifted plumes
    for f in [0,1,2]:
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1, f], height_ratios=[1, 1], hspace=0.1)
        ax_upper = fig1.add_subplot(inner_grid[0])
        if f==0:
            y1=orig_ratio
            y2=orig_fit_sig
            oc=orig_coef
            top_pairs=orig_pairs
            ax_upper.set_title('Original plume', fontsize=label_size)

        if f==1:
            y1=trans_ratio[index1]
            y2=trans_fit_sig[index1]
            oc=trans_coef[index1]
            top_pairs=trans_pairs[index1]
            ax_upper.set_title('Shifted plume with smallest norm dist', fontsize=label_size)

        if f==2:
            y1=trans_ratio[index2]
            y2=trans_fit_sig[index2]
            oc=trans_coef[index2]
            top_pairs=trans_pairs[index2]
            ax_upper.set_title('Shifted plume with highest MF', fontsize=label_size)
        
        line1, =ax_upper.plot(wl, y1, label='Measurement')
        line2, =ax_upper.plot(wl, y2, label='Model')
        
        color1 = line1.get_color()
        color2 = line2.get_color()
        
        yy= np.polyval(oc[1:], wl)  # continuum function
        ax_upper.plot(wl, yy, label='Continuum function', color='green', linestyle='--', alpha=0.5)

        ax_lower = fig1.add_subplot(inner_grid[1])          
        y1=y1/yy
        y2=y2/yy
        ax_lower.plot(wl, y1, label='Measurement / continuum function', color=color1, linestyle='--')
        ax_lower.plot(wl, y2, label='Model / continuum function', color=color2, linestyle='--')
        ax_lower.set_xlabel('Wavelength', fontsize=label_size)

        dist = calculate_dist(y1, y2, 0)
        mag = calculate_magnitude(y1, 1)
        ndist=dist/mag

        if f==0:
            handles_upper, labels_upper = ax_upper.get_legend_handles_labels()
            handles_lower, labels_lower = ax_lower.get_legend_handles_labels()
            combined_handles = handles_upper + handles_lower
            combined_labels = labels_upper + labels_lower
            ax_upper.legend(combined_handles, combined_labels, loc='upper center', bbox_to_anchor=(-0.08, 1.8), fontsize=14)
        lines, labels = ax_lower.get_legend_handles_labels()
               
        dd = [rfl[i, j, ind_fit] for (i, j), _, _, in top_pairs]
        dd_all = np.concatenate(dd)
        mean_of_rfl = np.mean(dd_all)
        std_of_rfl = np.std(dd_all)

        dd = [rdn[i, j, ind_fit] for (i, j), _, _, in top_pairs]
        dd_all = np.concatenate(dd)
        mean_of_rdn = np.mean(dd_all)
        std_of_rdn = np.std(dd_all)

        dd = [mf[i, j] for (i, j), _, _, in top_pairs]
        dd_all = dd
        mean_of_mf = np.mean(dd_all)
        std_of_mf = np.std(dd_all)

        dd = [similarity for _, _, similarity in top_pairs]
        dd_all = dd
        mean_of_sim = np.mean(dd_all)
        std_of_sim = np.std(dd_all)

        ax_lower.annotate(  
        f'MF: {mean_of_mf:.0f} ± {std_of_mf:.0f}\n'
        f'Sim: {mean_of_sim:.5f} ± {std_of_sim:.5f}\n'
        f'RDN: {mean_of_rdn:.3f} ± {std_of_rdn:.3f}\n'
        f'RFL: {mean_of_rfl:.3f} ± {std_of_rfl:.3f}\n'
        f'Dist: {dist:.4f}\n'
        f'Norm dist: {ndist:.3f}',
        xy=(0, 0),  
        xycoords='axes fraction',
        textcoords="offset points",
        xytext=(100, -110),  
        fontsize=12,
        ha='right',  
        va='bottom',  
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='none'),
        annotation_clip=False)

    fig1.text(0.02, 0.98, f'{ii}', transform=plt.gcf().transFigure, fontsize=14, va='top', ha='left')
    fig1.savefig(os.path.join(pics_folder, f'frame_{ii}.png'), dpi=300) 