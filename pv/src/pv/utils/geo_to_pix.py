
import numpy as np


def geo_to_pix( xy_geo, l1b_glt):
    """Use L1B GLT-provided affine transformation coefficients to map xy
    coordinates in geographic space (e.g., lon/lat) to pixel coordinates in L1B
    GLT and, from there, perform sample/line lookup to determine corresponding
    EMIT acquisition file pixel locations.

    Args:
        xy_geo (array like, float): List of x,y (lon,lat) geographical
            coordinate pairs.
        l1b_glt (EMITAcquisitionFile): L1B GLT file instance.

    Returns:
        l1b_glt_xy_pix (array like, int): L1B GLT xy (sample/line, column/row)
            pixel coordinates from affine transformation.
        emit_line_sample_pix (array like, int): Corresponding EMIT acquisition
            file line/sample lookup coordinates from L1B GLT.

    """
    # Implementation notes:
    #
    # Orthorectification (affine) transformation can be expressed as (ref.
    # https://github.com/nasa/EMIT-Data-Resources/blob/main/python/how-tos/How_to_Orthorectify.ipynb):
    #
    #   x_geo = GT[0] + x * GT[1] + y * GT[2]
    #   y_geo = GT[3] + x * GT[4] + y * GT[5]
    #
    # where (*):
    #
    #   x_geo, y_geo - coordinates in geographic space (e.g., lon/lat)
    #   x, y - coordinates in integer pixel space
    #   GT[0] - x coordinate of upper-left corner of upper-left pixel
    #   GT[1] - w-e pixel width
    #   GT[2] - row rotation (zero for north up images)
    #   GT[3] - y coordinate of upper-left corner of upper-left pixel
    #   GT[4] - column rotation (zero for north up images)
    #   GT[5] - n-s pixel height (negative for north-up image)
    #
    # The corresponding inverse relationship (pixel space from geographic space) is:
    #
    #   x = ((x_geo - GT[0])*GT[5] - (y_geo - GT[3])*GT[2]) / (GT[1]*GT[5] - GT[2]*GT[4])
    #   y = ((y_geo - GT[3])*GT[1] - (x_geo - GT[0])*GT[4]) / (GT[1]*GT[5] - GT[2]*GT[4])
    #
    # Which, for the case of EMIT with GT[2]=GT[4]=0 simplifies to:
    #
    #   x = (x_geo - GT[0]) / GT[1]
    #   y = (y_geo - GT[3]) / GT[5]
    #
    # Correlating GT[i] coefficients with L1B GLT 'map info' header contents (per
    # EMIT L1B Software Interface Specifications, JPL D-104187, and observation
    # with actual generated acquisitions):
    #
    #   -------------------------------------------------------------------------------
    #   map info field name  |  map info index (zeros-based)  |  GT index (zeros-based)
    #   -------------------------------------------------------------------------------
    #   proj name            |  0                             |  -
    #   pixel x loc          |  1                             |  -
    #   pixel y loc          |  2                             |  -
    #   pixel easting        |  3                             |  0
    #   pixel northing       |  4                             |  3
    #   x pixel size         |  5                             |  1
    #   y pixel size         |  6                             |  5
    #   datum                |  7                             |  -
    #
    #   The following conventions are used:
    #
    #               ^ +lat, +y_geo                     \
    #               |                                   | geographic
    #               |                                   | space
    #               o-----------  ---> +lon, +x_geo    /
    #   +line, +y | |           |                      \
    #             | |           |                       | pixel
    #             v |           |                       | (retrieval)
    #               |           |                       | space
    #               |           |                       |
    #                -----------                       /
    #                ---> +sample, +x
    #

    _xy_geo = np.array(xy_geo).reshape(-1, 2)
    _xy_pix = np.zeros(_xy_geo.shape, dtype=int)

    map_info = l1b_glt.map_info

    _xy_pix[:, 0] =  ((_xy_geo[:, 0]-float(map_info[3]))/float(map_info[5])).round().astype(int)
    _xy_pix[:, 1] = -((_xy_geo[:, 1]-float(map_info[4]))/float(map_info[6])).round().astype(int)

    # _xy_pix correspond to lon,lat. since lon indexes on columns and lat
    # indexes on rows, note the index interchange when addressing l1b glt matrix
    # terms:

    glt_data = l1b_glt.glt_data
    glt_sample_lookup = glt_data[_xy_pix[:, 1], _xy_pix[:, 0], 0] - 1
    glt_line_lookup   = glt_data[_xy_pix[:, 1], _xy_pix[:, 0], 1] - 1

    return \
        _xy_pix, \
        np.column_stack((glt_line_lookup,glt_sample_lookup)) # note: line, sample column order

