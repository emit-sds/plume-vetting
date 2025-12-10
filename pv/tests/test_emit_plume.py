
import geojson
import geopandas as gpd
import unittest
import yaml

import pv

cfg_src = './config.yaml'
plume_data_src = './data/previous_manual_annotation_oneback.json'

class TestEMITPlume(unittest.TestCase):
    """Test plume data initialization using equivalent methods.

    """
    def testEMITPlumeConstructor(self):

        # initialize using raw GeoJSON FeatureCollection data file:
        emit_plume_1 = pv.emit_plume.EMITPlume(
            plume_data=plume_data_src, plume_id='CH4_PlumeComplex-3727', cfg=cfg_src)

        # initialize using parsed GeoJSON data:
        pf = open(plume_data_src)
        plume_data_all = geojson.load(pf)                   # geojson.feature.FeatureCollection
        pf.close()
        emit_plume_2 = pv.emit_plume.EMITPlume(
            plume_data=plume_data_all, plume_id='CH4_PlumeComplex-3727', cfg=cfg_src)

        # initialize using parsed geopandas data:
        gpd_plume_data_all = gpd.GeoDataFrame.from_features(plume_data_all['features'])
        emit_plume_3 = pv.emit_plume.EMITPlume(
            plume_data=gpd_plume_data_all, plume_id='CH4_PlumeComplex-3727', cfg=cfg_src)

        # initialize using parsed, pre-selected data:
        gpd_plume_data_selected = gpd_plume_data_all[       # geopandas.geodataframe.GeoDataFrame
            [plume_id=='CH4_PlumeComplex-3727' for plume_id in gpd_plume_data_all['Plume ID']]]
        cf = open(plume_data_src)
        emit_plume_4 = pv.emit_plume.EMITPlume(
            plume_data=gpd_plume_data_selected, cfg=yaml.safe_load(cf))
        cf.close()

        self.assertTrue(all(emit_plume_1.plume==emit_plume_2.plume))
        self.assertTrue(all(emit_plume_2.plume==emit_plume_3.plume))
        self.assertTrue(all(emit_plume_3.plume==emit_plume_4.plume))


if __name__=='__main__':
    unittest.main()

