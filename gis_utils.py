import numpy as np
import pandas as pd
#import gdal
#import pyproj
#import osr
from osgeo import gdal
from matplotlib import pyplot as plt

#import spectral


'''
Some useful GDAL commands:

gdal_translate -projwin_srs "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs" -projwin -120.163759 39.255369 -119.952025 38.980077 
HYT_20180529t183556_MegaboxLine30CA__AV_tag_f180529t01p00r07_combined_avcl_hyt.tif lake_tahoe_all_bands.tif

'''

epsg_latlon = 'epsg:4326'
epsg_ca_UTM_11_North = 'epsg:32611'

dtype_map = [('1', np.uint8),                   # unsigned byte
             ('2', np.int16),                   # 16-bit int
             ('3', np.int32),                   # 32-bit int
             ('4', np.float32),                 # 32-bit float
             ('5', np.float64),                 # 64-bit float
             ('6', np.complex64),               # 2x32-bit complex
             ('9', np.complex128),              # 2x64-bit complex
             ('12', np.uint16),                 # 16-bit unsigned int
             ('13', np.uint32),                 # 32-bit unsigned int
             ('14', np.int64),                  # 64-bit int
             ('15', np.uint64)]                 # 64-bit unsigned int
envi_to_dtype = dict((k, np.dtype(v).char) for (k, v) in dtype_map)

def read_envi_header(file):
    '''
    I (Jay Fahlen) took this function directly from the spectral library with minimal changes. It is originally
    from spectral/spectral/io/envi.py

    USAGE: hdr = read_envi_header(file)

    Reads an ENVI ".hdr" file header and returns the parameters in a
    dictionary as strings.  Header field names are treated as case
    insensitive and all keys in the dictionary are lowercase.
    '''
    f = open(file, 'r')

    try:
        starts_with_ENVI = f.readline().strip().startswith('ENVI')
    except UnicodeDecodeError:
        msg = 'File does not appear to be an ENVI header (appears to be a ' \
          'binary file).'
        f.close()
        raise ValueError(msg)
    else:
        if not starts_with_ENVI:
            msg = 'File does not appear to be an ENVI header (missing "ENVI" \
              at beginning of first line).'
            f.close()
            raise ValueError(msg)

    lines = f.readlines()
    f.close()

    dict = {}
    have_nonlowercase_param = False
    support_nonlowercase_params = True
    try:
        while lines:
            line = lines.pop(0)
            if line.find('=') == -1: continue
            if line[0] == ';': continue

            (key, sep, val) = line.partition('=')
            key = key.strip()
            if not key.islower():
                have_nonlowercase_param = True
                if not support_nonlowercase_params:
                    key = key.lower()
            val = val.strip()
            if val and val[0] == '{':
                str = val.strip()
                while str[-1] != '}':
                    line = lines.pop(0)
                    if line[0] == ';': continue

                    str += '\n' + line.strip()
                if key == 'description':
                    dict[key] = str.strip('{}').strip()
                else:
                    vals = str[1:-1].split(',')
                    for j in range(len(vals)):
                        vals[j] = vals[j].strip()
                    dict[key] = vals
            else:
                dict[key] = val

        if have_nonlowercase_param and not support_nonlowercase_params:
            msg = 'Parameters with non-lowercase names encountered ' \
                  'and converted to lowercase. To retain source file ' \
                  'parameter name capitalization, set ' \
                  'spectral.settings.envi_support_nonlowercase_params to ' \
                  'True.'
            print(msg)
            print('ENVI header parameter names converted to lower case.')
        return dict
    except:
        raise ValueError()

def read_envi(filename, get_wavelenghts = False):

    if filename[-4:] == '.hdr':
        hdr_filename = filename
        envi_filename = filename[:-4] + '.img'
    
    elif filename[-4:] == '.img':
        hdr_filename = filename[:-4] + '.hdr'
        envi_filename = filename
    
    else:
        hdr_filename = filename + '.hdr'
        envi_filename = filename

    #print(filename)
    #print(hdr_filename)
    #print(envi_filename)

    data_hdr = read_envi_header(hdr_filename)
    nl, ns, nb = int(data_hdr['lines']), int(data_hdr['samples']), int(data_hdr['bands'])
    cnt = nl * ns * nb
    data = np.fromfile(envi_filename, count = cnt, dtype = envi_to_dtype[data_hdr['data type']]).reshape((nl, nb, ns))
    data = np.ascontiguousarray(data.transpose([0,2,1]))

    if get_wavelenghts:
        w_nm = np.array([float(x) for x in data_hdr['wavelength']])
        return data_hdr, data, w_nm

    return data_hdr, data

def load_tif_data(filename, band_indices_zero_based = None):

    gdata = gdal.Open(filename, gdal.GA_ReadOnly)
    geotransform = gdata.GetGeoTransform()

    if band_indices_zero_based is None:
        d = gdata.ReadAsArray()
    else:
        bands = gdata.GetRasterBand(band_indices_zero_based - 1)
        d = bands.ReadAsArray()
    return d


def plot_geotiff(filename, band = 30):
    gdata = gdal.Open(filename, gdal.GA_ReadOnly)
    x0, dx, _, y0, _, dy = gdata.GetGeoTransform()

    band = gdata.GetRasterBand(band - 1)
    d = band.ReadAsArray()

    xaxis = np.arange(gdata.RasterXSize)*dx + x0
    yaxis = np.arange(gdata.RasterYSize)*dy + y0

    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.imshow(d, extent = [xaxis[0], xaxis[-1], yaxis[-1], yaxis[0]])

    #plot_met_points = True
    #if plot_met_points:
    #    for key, (lat, lon) in met_locations_latlon_dict.items():
    #        x, y = convert_latlon_to_CA_UTM_11_N(lat, lon)
    #        ax.plot(x, y, 'o', label = key)
    
    #    ax.legend()

    return fig, ax

def get_array_index_from_native_coord(filename, x, y):
    gdata = gdal.Open(filename, gdal.GA_ReadOnly)
    x0, dx, _, y0, _, dy = gdata.GetGeoTransform()

    return (x - x0)/dx, (y - y0)/dy

def get_array_index_from_latlon(filename, lat, lon):
    gdata = gdal.Open(filename, gdal.GA_ReadOnly)
    proj = osr.SpatialReference(wkt = gdata.GetProjection())
    epsg_str = 'epsg:' + proj.GetAttrValue('AUTHORITY', 1)

    x, y = convert_latlon_to_CA_UTM_11_N(lat, lon, epsg_str=epsg_str)
    return get_array_index_from_native_coord(filename, x,y)


def convert_CA_UTM_11_N_to_latlon(x1, y1):
    inProj = pyproj.Proj(init=epsg_ca_UTM_11_North)
    outProj = pyproj.Proj(init=epsg_latlon)

    lon, lat = pyproj.transform(inProj,outProj,x1,y1)
    return lat, lon

def convert_latlon_to_CA_UTM_11_N(lat, lon, epsg_str = epsg_ca_UTM_11_North):
    inProj = pyproj.Proj(init=epsg_latlon)
    outProj = pyproj.Proj(init=epsg_str)
    return pyproj.transform(inProj,outProj,lon,lat)

def change_envi_header_lines(filename, output_filename):
    with open(filename,'r') as f:
        ls = f.readlines()
    
    for i, l in enumerate(ls):
        if l.startswith('bands'):
            l_idx = i
    
    ls[l_idx] = 'bands = 267\n'

    #s = '\n'.join(ls)

    with open(output_filename, 'w') as f:
        f.writelines(ls)

            
# Taken from EMIT's scrape_refine_upload.py
def rawspace_coordinate_conversion(glt, coordinates, trans, ortho=False):
    rawspace_coords = []
    for ind in coordinates:
        glt_ypx = int(round((ind[1] - trans[3])/ trans[5]))
        glt_xpx = int(round((ind[0] - trans[0])/ trans[1]))
        if ortho:
            rawspace_coords.append([glt_xpx,glt_ypx])
        else:
            lglt = glt[glt_ypx, glt_xpx,:]
            rawspace_coords.append(lglt.tolist())
    return rawspace_coords