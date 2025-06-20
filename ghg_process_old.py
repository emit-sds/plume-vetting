#! /usr/bin/env python
#
#  Copyright 2023 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov


import argparse
import subprocess
import target_generation_old
import logging
from spectral.io import envi
import numpy as np
import os
from utils import envi_header
from osgeo import gdal


def main(input_args=None, l=None, ind=None, average_over_pix_flag=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('radiance_file', type=str,  metavar='INPUT', help='path to input image')   
    parser.add_argument('obs_file', type=str,  help='path to observation image')   
    parser.add_argument('loc_file', type=str,  help='path to location image')   
    parser.add_argument('glt_file', type=str,  help='path to glt image')   
    parser.add_argument('atm_file', type=str,  help='path to atm image') 
    parser.add_argument('l1b_bandmask_file', type=str,  help='path to l1b bandmask image')   
    parser.add_argument('l2a_mask_file', type=str,  help='path to l2a mask image')   
    parser.add_argument('output_base', type=str,  help='output basepath for output image')    
    parser.add_argument('--state_subs', type=str, default=None,  help='state file from OE retrieval')    
    parser.add_argument('--overwrite', action='store_true',  help='state file from OE retrieval')    
    parser.add_argument('--ace_filter', action='store_true',  help='use an ACE filter during matched filter')    
    parser.add_argument('--loglevel', type=str, default='INFO', help='logging verbosity')    
    parser.add_argument('--logfile', type=str, default=None, help='output file to write log to')    
    parser.add_argument('--co2', action='store_true', help='flag to indicate whether to run co2')  
    #parser.add_argument('--l', type=float, default=None, help='additional argument')  
    
    args = parser.parse_args(input_args)

    radiance_file = args.radiance_file
    radiance_file_hdr = envi_header(radiance_file)
 
    obs_file = args.obs_file
    obs_file_hdr = envi_header(obs_file)

    loc_file = args.loc_file
    loc_file_hdr = envi_header(loc_file)

    logging.basicConfig(format='%(levelname)s:%(asctime)s ||| %(message)s', level=args.loglevel,
                        filename=args.logfile, datefmt='%Y-%m-%d,%H:%M:%S')

    # Target
    ch4_target_file = args.output_base + '_ch4_target'
    
    path = os.environ['PATH']
    path = path.replace(r'\Library\bin;', ':')
    os.environ['PATH'] = path
    args.overwrite=True


    if args.overwrite:
        solar_za = envi.open(obs_file_hdr).open_memmap(interleave='bip')[...,4]
        mean_solar_za = np.mean(solar_za[solar_za != -9999])

        sensor_za = envi.open(obs_file_hdr).open_memmap(interleave='bip')[...,2]
        mean_sensor_za = np.mean(sensor_za[sensor_za != -9999])

        elevation = envi.open(loc_file_hdr).open_memmap(interleave='bip')[...,2]
        mean_elevation = np.mean(elevation[elevation != -9999]) / 1000.
        mean_elevation = min(max(0, mean_elevation),3)

        if args.state_subs is not None:
            state_ds = envi.open(envi_header(args.state_subs))
            band_names = state_ds.metadata['band names']
            h2o = state_ds.open_memmap(interleave='bip')[...,band_names.index('H2OSTR')]
            mean_h2o = np.mean(h2o[h2o != -9999])
        else:
            # Just guess something...
            exit()
            mean_h2o = 1.3
    
       
        uas=target_generation_old.main(['--ch4', '-z', str(mean_sensor_za), '-g', str(mean_elevation), '-w', str(mean_h2o), '-s', str(mean_solar_za), '--output', ch4_target_file, '--hdr', radiance_file_hdr])
      

    return uas

    

if __name__ == '__main__':
    main()
