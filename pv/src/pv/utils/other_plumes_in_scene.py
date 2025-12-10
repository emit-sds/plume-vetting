
def other_plumes_in_scene( fid, plume_id, plume_data_all):
    """Identify plumes other than 'plume_id' that are in the 'fid' scene of interest.

    Args:
        fid (str, or list): Scene of interest, in 'emityyyymmddthhmmss' format.
        plume_id (str, or list): Plume complex of interest (e.g., 'CH4_PlumeComplex-1234').
        plume_data_all (geopandas.geodataframe.GeoDataFrame): List of other plumes to search.

    Returns:
        List of plume_data_all 'Plume ID's other than 'plume_id' that are in the 'fid'
        scene of interest (may be None).

    """

    if isinstance(fid,str):
        fid_as_list = []
        fid_as_list.append(fid)
    elif isinstance(fid,list) and len(fid)==1:
        fid_as_list = fid
    else:
        raise RuntimeException('fid must be a string or single element list')

    if isinstance(plume_id,str):
        plume_id_as_list = []
        plume_id_as_list.append(plume_id)
    elif isinstance(plume_id,list) and len(plume_id)==1:
        plume_id_as_list = plume
    else:
        raise RuntimeException('plume_id must be a string or single element list')

    # all plumes in the scene (including plume_id plume of interest):
    plumes_in_scene = plume_data_all[
        [any(set(fid_as_list).intersection(set(fids))) for fids in gpd_plume_data_all['fids']] ]

    # ensure list only contains plumes other than plume_id (i.e. plume of interest):
    other_plumes_in_scene = plumes_in_scene[
        [plume_id_as_list[0]!=plume_id for plume_id in plumes_in_scene['Plume ID']] ]

    # return list of other plume_ids in the scene:
    try:
        return [plume_id for plume_id in other_plumes_in_scene['Plume ID']]
    except:
        return None

