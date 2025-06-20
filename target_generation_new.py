from os.path import exists
import numpy as np
import scipy.ndimage
import argparse
import spectral
#from loop_plume import all_r, center

all_r = np.load('all_r.npy')  # load new LUT
center = np.load('center.npy') # wavelength of each channel

def find_ind(num_table,value_min,value_max,value):
    value = max(value_min, min(value, value_max))  
    ind=(value-value_min)/(value_max-value_min)*(num_table-1)
    return ind


def get_5deg_lookup_index(sensor_za=0, ground=0, water=0, solar_za=30, conc=0):
     num_table=5
     maxz,minz,maxg,ming,maxw,minw,maxs,mins,maxc,minc=180,120,3,0,6,0,60,0,60000,0

     sensor_za=180-sensor_za
     if sensor_za<120:
       sensor_za=120

     z_ind=find_ind(num_table,minz,maxz,sensor_za)
     g_ind=find_ind(num_table,ming,maxg,ground)
     w_ind=find_ind(num_table,minw,maxw,water)
     s_ind=find_ind(num_table,mins,maxs,solar_za)
     c_ind=find_ind(num_table,minc,maxc,conc)  

     idx = np.asarray([[z_ind],[g_ind],[w_ind],[s_ind],[c_ind]])
     return idx


def spline_5deg_lookup(grid_data, sensor_za=0, ground=0, water=0, solar_za=30, conc=0, order=1):
    
    coords = get_5deg_lookup_index(
        sensor_za=sensor_za, ground=ground, water=water, solar_za=solar_za, conc=conc)

    if order == 1:
        coords_fractional_part, coords_whole_part = np.modf(coords)
        coords_near_slice = tuple((slice(int(c), int(c+2)) for c in coords_whole_part))
        near_grid_data = grid_data[coords_near_slice]
        new_coord = np.concatenate((coords_fractional_part * np.ones((1, near_grid_data.shape[-1])),
                                    np.arange(near_grid_data.shape[-1])[None, :]), axis=0)
        lookup = scipy.ndimage.map_coordinates(near_grid_data, coordinates=new_coord, order=1, mode='nearest')
    elif order == 3:
        lookup = np.asarray([scipy.ndimage.map_coordinates(
            im, coordinates=coords_fractional_part, order=order, mode='nearest') for im in np.moveaxis(near_grid_data, 5, 0)])
    return lookup.squeeze()

   

def generate_library(l, sensor_za_list=0, ground_list=0, water_list=0, solar_za_list=30, order=1):

    grid, wave= all_r, center

    rads_all=[] # store radiance
    cons_all=[] 

    for l_pix, sensor_za, ground, water, solar_za in zip(l, sensor_za_list, ground_list, water_list, solar_za_list):
      if all(val != -9999 for val in (l_pix, sensor_za, ground, water, solar_za)):  
        gas_concentration_vals=np.linspace(max(0, l_pix - 1000), min(60000, l_pix + 1000), 8)
        
        rads = np.empty((len(gas_concentration_vals), grid.shape[-1]))

        for i, ppmm in enumerate(gas_concentration_vals):
            rads[i, :] = spline_5deg_lookup(
                grid, sensor_za=sensor_za, ground=ground, water=water, solar_za=solar_za, conc=ppmm, order=order)
    rads_all.append(rads)
    cons_all.append(gas_concentration_vals)
    return rads_all, cons_all, wave


def generate_template_from_bands(l, centers, fwhm, params, **kwargs):

    SCALING = 1e5
    centers = np.asarray(centers)
    fwhm = np.asarray(fwhm)

    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(fwhm)):
        raise RuntimeError(
            'Band Wavelengths Centers/FWHM data contains non-finite data (NaN or Inf).')
    if centers.shape[0] != fwhm.shape[0]:
        raise RuntimeError(
            'Length of band center wavelengths and band fwhm arrays must be equal.')
  
    rads_all, cons_all, wave = generate_library(l, **params)
    
    # sigma = fwhm / ( 2 * sqrt( 2 * ln(2) ) )  ~=  fwhm / 2.355
    """ sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # response = scipy.stats.norm.pdf(wave[:, None], loc=centers[None, :], scale=sigma[None, :])
    # Evaluate normal distribution explicitly
    var = sigma ** 2
    denom = (2 * np.pi * var) ** 0.5
    numer = np.exp(-(np.asarray(wave)[:, None] - centers[None, :])**2 / (2*var))
    response = numer / denom
    # Normalize each gaussian response to sum to 1.
    response = np.divide(response, response.sum(
        axis=0), where=response.sum(axis=0) > 0, out=response)
    # implement resampling as matrix multiply
    resampled = rads.dot(response) """
    target_all=[]
    for rads, concentrations in zip(rads_all, cons_all):  
        resampled=rads   
        lograd = np.log(resampled, out=np.zeros_like(
            resampled), where=resampled > 0)
        slope, _, _, _ = np.linalg.lstsq(
            np.stack((np.ones_like(concentrations), concentrations)).T, lograd, rcond=None)
        spectrum = slope[1, :] * SCALING
        target = np.stack((np.arange(1, spectrum.shape[0]+1), centers, spectrum)).T
        target_all.append(target[:,2])
        #np.save('target_sig.npy', target)
    target=np.mean(target_all, axis=0)
    return target


def main(input_args=None, l=None, sensor_za_list=None, solar_za_list=None, ground_list=None, water_list=None):
    parser = argparse.ArgumentParser(
        description='Create a unit absorption spectrum for specified parameters.')
    parser.add_argument('-z', '--sensor_zenith', nargs='+', type=float, required=True,
                        help='Zenith Angle (in degrees) for generated spectrum.')
    parser.add_argument('-g', '--ground_elevation', type=float,
                        required=True, help='Ground Elevation (in km).')
    parser.add_argument('-w', '--water_vapor', type=float,
                        required=True, help='Column water vapor (in cm).')
    parser.add_argument('-s', '--solar_zenith', type=float,
                        required=True, help='Column solar zenith angle (in degrees).')
    parser.add_argument('--order', choices=(1, 3), default=1,
                        type=int, required=False, help='Spline interpolation degree.')
    gas = parser.add_mutually_exclusive_group(required=False)
    gas.add_argument('--co2', action='store_const', dest='gas', const='co2')
    gas.add_argument('--ch4', action='store_const', dest='gas', const='ch4')
    wave = parser.add_mutually_exclusive_group(required=True)
    wave.add_argument(
        '--hdr', type=str, help='ENVI Header file for the flightline to match band centers/fwhm.')
    wave.add_argument('--txt', type=str,
                      help='Text-based table for band centers/fwhm.')
    parser.add_argument('--source', type=str,
                        choices=['full', 'pca'], default='full')
    parser.add_argument('-o', '--output', type=str,
                        default='generated_uas.txt', help='Output file to save spectrum.')
    parser.add_argument('--concentrations', type=float, default=None,
                        required=False, nargs='+', help='override the ppmm lookup values')
    parser.set_defaults(gas='ch4')
   
    args = parser.parse_args(input_args)
    param = {'sensor_za_list': sensor_za_list,
             # Model uses sensor height above ground
             #'sensor': args.sensor_altitude - args.ground_elevation,
             'ground_list': ground_list,
             'water_list': water_list,
             'solar_za_list': solar_za_list,
             'order': args.order}
    if args.hdr and exists(args.hdr):
        image = spectral.io.envi.open(args.hdr)
        centers = np.array([float(x) for x in image.metadata['wavelength']])
        fwhm = np.array([float(x) for x in image.metadata['fwhm']])
    elif args.txt and exists(args.txt):
        data = np.loadtxt(args.txt, usecols=(0, 1),delimiter=',')
        centers = data[:, 0]
        fwhm = data[:, 1]
    else:
        raise RuntimeError(
            'Failed to load band centers and fwhm from file. Check that the specified file exists.')
    concentrations = args.concentrations

    uas = generate_template_from_bands(l, centers, fwhm, param,
                                       concentrations=concentrations)
    
    # np.savetxt(args.output, uas, delimiter=' ', fmt=('%03d', '% 10.3f', '%.18f'))

    return uas

if __name__ == '__main__':
    main()
