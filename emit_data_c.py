import numpy as np
import glob
import pdb
import pickle
import scipy.spatial.distance as SSD
import os
from osgeo import gdal
#from scipy.interpolate import interp1d
#import scipy.ndimage
import subprocess
import spectral
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
#from mpl_toolkits.axes_grid1 import ImageGrid
import json
from matplotlib.path import Path
import gis_utils

class EMIT():
    def __init__(self, name_string, basepath = '/store/emit/ops/data/acquisitions/', 
                 baseline_methane_path = '/store/brodrick/methane/methane_20221121/',
                 #baseline_methane_path = '/beegfs/scratch/brodrick/methane/methane_20230813/',
                 #methane_path = '/beegfs/scratch/jfahlen/run_comparison_segment_poly_smoothing_modsigmod_bias_corrected/mod_plume/',
                 methane_path='/store/brodrick/methane/methane_20230813/',
                 methane_name_string = 'ch4_mf',
                 mask_path = '/store/brodrick/methane/public_mmgis_masked/',
                 load_data = False,
                 load_rdn = False,
                 load_mf = False,
                 load_baseline_mf = False,
                 load_state = False,
                 load_atm = False,
                 load_bandmask = False,
                 load_wavelength_um = False, 
                 load_target_mask = False,
                 load_injected_mf = False,
                 load_before_sum = False,
                 load_injected_before_sum = False,
                 load_l2b_abun = False,
                 load_rfl = False):

        self.name_string = name_string
        self.basepath = basepath
        self.baseline_methane_path = baseline_methane_path
        self.methane_path = methane_path
        self.methane_name_string = methane_name_string

        self.baseline_methane_filename = baseline_methane_path + f'{name_string}_ch4_mf' 
        #self.methane_filename = methane_path + f'{name_string}_ch4_mf' 
        date_string = name_string[4:12]  # Extracts '20230620' from the name_string
        self.methane_filename = (f'{methane_path}{date_string}/{name_string}_ch4_mf')
        # if methane_name_string is not None:
        #     self.methane_filename = methane_path + f'{name_string}_{methane_name_string}' 
        self.injected_before_sum_filename = methane_path + f'{name_string}_injected_ch4_mf_before_sum' 
        self.before_sum_filename = methane_path + f'{name_string}_ch4_mf_before_sum' 

        self.run_folder = self.basepath + f'{self.name_string[4:12]}/{self.name_string}' + '/'

        # Standard EMIT products
        self.rdn_filename = self._find_file('l1b', 'rdn')
        self.rdn_filename_hdr = '.'.join(self.rdn_filename.split('.')[:-1]) + '.hdr'

        self.obs_filename = self._find_file('l1b', 'obs')
        self.loc_filename = self._find_file('l1b', 'loc')
        self.glt_filename = self._find_file('l1b', 'glt')
        self.rfl_filename = self._find_file('l2a', 'rfl_')
        self.atm_filename = self._find_file('l2a', 'atm')
        self.l2b_abun_filename = self._find_file('l2b', 'l2b_abun_b0106')
        self.state_subs_filename = self._find_file('l2a', 'statesubs_b0106')
        self.atm_filename = self._find_file('l2a', 'atm_b0106')

        self.load_data = load_data
        self.load_mf = load_mf
        self.load_baseline_mf = load_baseline_mf
        self.load_wavelength_um = load_wavelength_um
        self.load_target_mask = load_target_mask
        self.load_state = load_state
        self.load_atm = load_atm
        self.load_rdn = load_rdn
        self.load_rfl = load_rfl
        self.load_injected_mf = load_injected_mf
        self.load_before_sum = load_before_sum
        self.load_injected_before_sum = load_injected_before_sum
        self.load_bandmask = load_bandmask
        self.load_l2b_abun = load_l2b_abun

        self.target_mask_filename = os.path.join(mask_path, name_string + '_ch4_mask.tif')

        if self.load_data:
            self.load_mf = True
            self.load_baseline_mf = True
            self.load_target_mask = True
            self.load_rdn = True
            self.load_state = True
            self.load_atm = True
            self.load_bandmask = True
            
        
        if self.load_wavelength_um:
            self.load_rdn = True

        if self.load_rdn:
            self.rdn_hdr, self.rdn, self.wavelength_nm = self._load_data(self.rdn_filename, get_wavelengths = True)
            self.wavelength_um = self.wavelength_nm / 1000.
            self.na, self.nc, self.ns = self.rdn.shape

        if self.load_rfl:
              self.rfl_hdr, self.rfl, _ = self._load_data(self.rfl_filename, get_wavelengths = True)


        # if self.load_baseline_mf:
        #     self.baseline_mf_hdr, self.baseline_mf = gis_utils.read_envi(self.baseline_methane_filename)
        #     self.baseline_mf = np.squeeze(self.baseline_mf[:,:,:])

        if self.load_mf:
            print(self.methane_filename)
            if os.path.exists(self.methane_filename):
                self.mf_hdr, self.mf = gis_utils.read_envi(self.methane_filename)
                self.mf = np.squeeze(self.mf[:,:,:])       
        
        if self.load_injected_mf:
            self.mf_injected_hdr, self.mf_injected = gis_utils.read_envi(self.injected_methane_filename)
            self.mf_injected = np.squeeze(self.mf_injected[:,:,:])
        
        if self.load_injected_before_sum:
            self.mf_injected_before_sum_hdr, self.mf_injected_before_sum = gis_utils.read_envi(self.injected_before_sum_filename)

        if self.load_l2b_abun:
            self.l2b_abun_hdr, self.l2b_abun = gis_utils.read_envi(self.l2b_abun_filename)

        if self.load_before_sum:
            self.mf_before_sum_hdr, self.mf_before_sum = gis_utils.read_envi(self.before_sum_filename)

        if self.load_state:
            self.state_subs = self._load_data(self.state_subs_filename)
        
        if self.load_atm:
            self.atm_hdr, self.atm = self._load_data(self.atm_filename)
            self.H2OSTR = self.atm[:,:, self.atm_hdr['band names'].index('H2OSTR')]

        if self.load_target_mask:
            try:
                gd = gdal.Open(self.target_mask_filename,gdal.GA_ReadOnly)
                self.target_mask = gd.ReadAsArray().astype(bool)
            except:
                print('No target mask')

        if self.load_rdn:
            image = spectral.io.envi.open(self.rdn_filename_hdr)
            self.centers_nm = np.array([float(x) for x in image.metadata['wavelength']])
            self.fwhm_nm = np.array([float(x) for x in image.metadata['fwhm']])
            del image

        self.bandmask_filename = self._find_file('l1b', 'bandmask_b0106')
        if self.load_bandmask:
            b_hdr, self.bandmask = gis_utils.read_envi(self.bandmask_filename)
            self.bandmask_norm = self._make_bad_pixel_mask()
        self.l2a_mask_filename = self._find_file('l2a', 'mask_b0106')
        



    def ghg_process_filenames(self, output_path_name = 'OUT', baseline = False, slurm_log_filename = None, queue_str = '', verbose = False):
        #queue_str = '-p debug'
        slurm_log = '' if slurm_log_filename is None else f'-o {slurm_log_filename}.%j.out'
        l2a_mask_filename = '' if baseline else self.l2a_mask_filename
        #l2a_mask_filename = self.l2a_mask_filename

        env_stuff = 'export PATH=/beegfs/store/shared/anaconda3/envs/emit-isofit-ops/bin:/beegfs/store/shared/anaconda3/condabin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/beegfs/bin:/opt/phoenix-2.2.0/bin; ' + \
                    'export PYTHONPATH=/beegfs/store/emit/ops/repos/isofit/:/beegfs/store/emit/ops/repos/emit-utils'

        beg = f'sbatch -N 1 -c 40 {slurm_log} {queue_str} --mem=180G --wrap="{env_stuff}; python ghg_process.py '

        s = f'{self.rdn_filename} {self.obs_filename} {self.loc_filename} {self.glt_filename} {self.atm_filename} ' +\
            f'{self.bandmask_filename} {l2a_mask_filename} {output_path_name.rstrip("/")}/{self.name_string} --state_subs {self.state_subs_filename}'

        cmd = beg + s

        #if verbose:
           # print(cmd)
        return s
    
    def run_me(self, baseline, *, output_path_name='/store/xiang/test', slurm_log_filename=None):
      #  cmd = f'cd {src_folder}; {self.ghg_process_filenames(output_path_name=output_path_name, baseline = baseline, slurm_log_filename=slurm_log_filename)}'
        s=self.ghg_process_filenames(output_path_name=output_path_name, baseline = baseline, slurm_log_filename=slurm_log_filename)
        return s
        
    
    def _find_file(self, level, tag, ext = '.img'):
        f = glob.glob(self.run_folder + f'{level}/*{tag}*{ext}')
        if len(f) != 1:
            print('no file')
        return f[0]
    
    def _load_data(self, f, get_wavelengths = False):
        ret = gis_utils.read_envi(f, get_wavelenghts=get_wavelengths)
        if len(ret) == 2:
            h, d = ret
            return h, np.squeeze(d[:,:,:])
        else:
            h, d, w = ret
            return h, np.squeeze(d[:,:,:]), np.squeeze(w)

    def get_rgb_rfl(self, temp_filename = '/dev/shm/jays_temp.tif', r = 35, b = 23, g = 11):
        in_ds = gdal.Open(self.rfl_filename, gdal.gdalconst.GA_ReadOnly) 
        #cmd = ['gdal_translate', self.rfl_filename, temp, “-b”, “35”, “-b”, “23”, “-b”, “11”, “-ot”, “Byte”, “-scale”, “-exponent”, “0.6", “-of”, “PNG”, “-co”, “ZLEVEL=9"]
        out_ds = gdal.Translate(temp_filename, in_ds, options=f'-b {r} -b {b} -b {g} -ot Byte -scale -exponent 0.6')
        out_arr = out_ds.ReadAsArray()
        in_ds = None
        out_ds = None
        return out_arr.transpose([1,2,0])