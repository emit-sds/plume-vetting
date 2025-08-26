"""
EMIT plume class.

"""

import geojson
import geopandas as gpd
import numpy as np
import re
from shapely import Point, Polygon
import yaml

from . import emit_file
from .utils import geo_to_pix


class EMITPlume(object):
    """Class that encapsulates a number of common EMIT plume operations related
    to plume vetting.

    Args:

        plume_id (str): if plume_data points to a collection (see next), ID of
            plume of interest, e.g., 'CH4_PlumeComplex-1234'.

        plume_data: Plume data collection in any one of several formats:

            - (Path and) filename of GeoJSON FeatureCollection object data file,
              e.g., './data/manual_annotation.json'.  Note that, by definition,
              a GeoJSON FeatureCollection object is a single object containing
              (at least) the two fields, 'type' and 'features', where 'type' is
              equal to 'FeatureCollection' and 'features' is an array of GeoJSON
              'Feature' objects. In this case, the 'features' will be searched
              for one containing plume_id.

            - GeoPandas GeoDataFrame containing all plume collection data, the
              result of parsing a GeoJSON FeatureCollection using
              plume_data=geopandas.GeoDataFrame.from_features(plume_data['features']).
              As in the above, the collection will be searched for 'plume_id'.

            - GeoPandas GeoDataFrame of length one (i.e., a single object with
              'Plume ID'==plume_id. In this case, the 'plume_id' input is not
              required and will be assumed to be equal to
              plume_data['Plume ID'].iloc[0].

        cfg (str or dict): (Path and) filename of yaml configuration file, or
            file parsed as a dictionary (see examples).  cfg is used to identify
            lookup paths and filename types for supporting scene data, e.g., L1B
            GLT, radiance, and matched filter files

    Attributes:

        cfg (dict): Parsed cfg input.

        plume (GeoPandas GeoDataFrame): Data frame of length one containing
            plume_id data.  Note that this attribute enables access to all
            relevant plume data as described by plume.columns, using standard
            GeoPandas methods.

    """
    def __init__( self, plume_id=None, plume_data=None, cfg=None):
        """EMITPlume constructor using one of several approaches.

        """
        self._boundary_xy_pixels= None
        self._ch4_mf            = None
        self._fid_key           = None
        self._l1b_glt           = None
        self._random_variation  = None

        if isinstance(cfg,str):
            cf = open(cfg)
            self.cfg = yaml.safe_load(cf)
            cf.close()
        elif isinstance(cfg,dict):
            self.cfg = cfg
        else:
            raise RuntimeError(f"Required input 'cfg' must either be a yaml file or dict.")

        # initialize plume depending on one of several approaches:

        gpd_plume_data_to_search = []
        
        if isinstance(plume_data,gpd.geodataframe.GeoDataFrame):

            if len(plume_data) == 1:
                # simplest case:
                self.plume = plume_data
            else:
                gpd_plume_data_to_search = plume_data

        elif isinstance(plume_data,str) or isinstance(plume_data,geojson.feature.FeatureCollection):

            if isinstance(plume_data,str):
                pf = open(plume_data)
                plume_data_geojson = geojson.load(pf)
                pf.close()
            else:
                plume_data_geojson = plume_data

            gpd_plume_data_to_search = gpd.GeoDataFrame.from_features(plume_data_geojson['features'])

        if any(gpd_plume_data_to_search):

            # since plume collections might use slightly different names for
            # certain keys of interest, uniquely determine applicable key
            # values: (note that this is somewhat of an experiment. if, in
            # practice this turns out to be unreliable, we can always add field
            # names to the config file for particular collections such as
            # "'FileID':'fids'", "'PlumeID':'Plume ID'", etc.)

            # file id key:
            re_search_bool = [
                True if re.search('fids',key,re.IGNORECASE) else False
                for key in gpd_plume_data_to_search.keys()]
            _fid_key = [key for key,re_srch_bool in zip(gpd_plume_data_to_search.keys(),re_search_bool) if re_srch_bool]
            if len(_fid_key)==1:
                self._fid_key = _fid_key[0]
            else:
                raise RuntimeError("Could not find a plume data collection key containing the string 'fids'")

            # other possible key searches here...

            gpd_plume_data = gpd_plume_data_to_search[
                [pid==plume_id for pid in gpd_plume_data_to_search['Plume ID']]]
            if len(gpd_plume_data) == 1:
                self.plume = gpd_plume_data
            else:
                raise RuntimeError(f"Could not find instance of '{plume_id}' in '{plume_data}'")



    @property
    def boundary_pixels(self):
        """Apply affine transformation and L1B GLT lookup to compute plume
        boundary coordinates in acquisition file x/y pixel (sample/line) space.

        Returns:
            Plume boundary coordinates expressed as shapely polygon of x/y
            (sample/line) pixels.

        """
        if self._boundary_xy_pixels is None:

            if not self._l1b_glt:
                # get the referenced glt for the scene:
                self._l1b_glt = emit_file.EMITAcquisitionFile(
                    root=self.cfg['emit_acquistion_dataproducts_root'],
                    id=self.fid,
                    level='l1b', type=self.cfg['emit_l1b_glt_type'])

            # plume boundary, in acquisition pixel space (line/sample):
            _, line_sample_pix = geo_to_pix(
                self.plume['geometry'].get_coordinates(),
                self._l1b_glt)

            self._boundary_xy_pixels = gpd.GeoSeries(
                Polygon(line_sample_pix[:,[1,0]]))  # note line/sample -> sample/line

        return self._boundary_xy_pixels


    @property
    def ch4_mf(self):
        """Get the methane matched filter results for the plume's scene.

        Returns:
            EMITMatchedFilterFile object corresponding to the plume's methane
            matched filter results.

        """
        if not self._ch4_mf:
            self._ch4_mf = emit_file.EMITMatchedFilterFile(
                root=self.cfg['emit_matched_filter_dataproducts_root'],
                id=self.plume[self._fid_key].iloc[0][0],
                type='ch4_mf')
        return self._ch4_mf


    @property
    def fid(self):
        return self.plume[self._fid_key].iloc[0][0]


    def mask(self,random_variation=False):
        """Return a boolean mask in, and size of, acquisition xy space
        consisting of: True==pixels inside of plume boundary, False==pixels
        outside of plume boundary.

        Args:
            random_variation (bool): If False (default), mask will be computed
                with respect to original plume boundary. If True, mask will be
                computed for current random variation of plume boundary (ref.
                self.new_random_variation()).

        """
        if not random_variation:
            # compute original plume mask:
            mask = np.zeros(self.ch4_mf.data.shape,dtype=bool)
            (min_x, min_y, max_x, max_y) = tuple(round(self.boundary_pixels.bounds).astype(int).iloc[0])
            for x in range(min_x,max_x+1):
                for y in range(min_y,max_y+1):
                    if self.boundary_pixels.iloc[0].contains(Point(x,y)):
                        mask[y,x]=True  # note ch4_mf.data are line(y)/sample(x)
        elif random_variation and self._random_variation is not None:
            # compute mask for current random plume variation:
            mask = np.zeros(self.ch4_mf.data.shape,dtype=bool)
            (min_x, min_y, max_x, max_y) = tuple(round(self._random_variation.bounds).astype(int).iloc[0])
            for x in range(min_x,max_x+1):
                for y in range(min_y,max_y+1):
                    if self._random_variation.iloc[0].contains(Point(x,y)):
                        mask[y,x]=True  # note ch4_mf.data are line(y)/sample(x)
        else:
            raise RuntimeError(
                "A random variation is not available or has not been generated; cannot compute corresponding mask")

        return mask


    def new_random_variation(self):
        """Generate a rotation+translation random plume variation in acquisition
        pixel space which can be subsequently accessed via the random_variation
        property. Resulting varied plume boundary is guaranteed to not intersect
        with original plume boundary, and to be entirely contained within the
        acquisition frame.

        Returns:
            Updated attribute self._random_variation, generally accessed via
            random_variation property.

        Remarks:
            Implementation as an explicit 'new_random_variation' function
            ensures new variations are only generated upon request, and that all
            subsequent operations relating to access of the varied plume can be
            made to be self-consistent until the next update.

        """
        # rotation+translation variation:

        rotated_plume = None
        new_random_variation = None

        # random rotation within acquisition scene:
        lines, samples, _ = self.ch4_mf.data.shape
        min_x_scene, min_y_scene, max_x_scene, max_y_scene = 0, 0, samples-1, lines-1
        min_x = min_y = max_x = max_y = -1
        while (min_x < min_x_scene) or (min_y < min_y_scene) or (max_x > max_x_scene) or (max_y > max_y_scene):
            rotated_plume = self.boundary_pixels.rotate(
                np.random.uniform(0.,2*np.pi),use_radians=True)
            (min_x, min_y, max_x, max_y) = tuple(round(rotated_plume.bounds).astype(int).iloc[0])

        while new_random_variation is None:
            # random translation within acquisition scene:
            xoff = np.random.uniform(-min_x, max_x_scene-max_x)
            yoff = np.random.uniform(-min_y, max_y_scene-max_y)
            translated_rotated_plume = rotated_plume.translate(xoff,yoff)
            if translated_rotated_plume.overlaps(self.boundary_pixels).any():
                continue
            else:
                new_random_variation = translated_rotated_plume

        self._random_variation = new_random_variation
        #self._random_variation = new_random_variation.iloc[0]


    @property
    def plume_id(self):
        return self.plume['Plume ID'].iloc[0]


    @property
    def random_variation(self):
        """self._random_variation accessor (ref. self.new_random_variation()).

        """
        return self._random_variation

