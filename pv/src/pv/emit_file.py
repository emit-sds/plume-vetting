"""
EMIT acquisition and matched filter file string and data access classes.

"""

import glob
import os
import re
import tempfile
import numpy as np
from osgeo import gdal
from spectral.io import envi


def EMITAcquisitionFile(filestr=None, **kwargs):
        fids = kwargs['ids']
        if fids is None:
            raise KeyError(f'ids must be provided')

        if len(fids) == 1:
            kwargs['id'] = fids[0]
            emit_file = EMITAcquisitionSingleFile(filestr, **kwargs)
        else:
            emit_file = EMITAcquisitionMultiFile(filestr, **kwargs)

        return emit_file


class EMITAcquisitionMultiFile(object):
    def __init__(self, filestr=None, **kwargs):
        fids = kwargs['ids']
        fids.sort()
        self.file_list = []

        for fid in fids:
            emit_file = EMITAcquisitionSingleFile(
                root=kwargs['root'],
                id=fid,
                level=kwargs['level'],
                type=kwargs['type'],
                ext=kwargs['ext']
            )

            self.file_list.append(emit_file)

    @property
    def data(self):
        combined_data = []
        for emit_file in self.file_list:
            combined_data.append(emit_file.data)

        combined_data = np.concatenate(combined_data, axis=0)

        return combined_data

    @property
    def glt_data(self):
        assert len(self.file_list) == 2

        glt_1 = self.file_list[0].data
        glt_2 = self.file_list[1].data
        glt_2 = glt_2.copy()

        # Update the second glt file 
        mask = glt_2[..., 1] != 0
        glt_2[mask, 1] += np.max(glt_1[..., 1])

        # Save the updated glt file in a tmp directory
        glt_2_filename = self.file_list[1].filename.replace('hdr', 'img')
        glt_2_ds = gdal.Open(glt_2_filename)
        glt_2_geotransform = glt_2_ds.GetGeoTransform()
        glt_2_projection = glt_2_ds.GetProjection()
        glt_2_metadata = glt_2_ds.GetMetadata()

        driver = gdal.GetDriverByName('GTiff')
        glt_2_tmp_file = os.path.join(tempfile.gettempdir(),
                                      f'{os.path.basename(glt_2_filename).replace(".img", "_tmp.tif")}')
        glt_2_tmp_ds = driver.Create(
            glt_2_tmp_file,
            glt_2.shape[1],
            glt_2.shape[0],
            2,
            np.int32
        )
        glt_2_tmp_ds.SetGeoTransform(glt_2_geotransform)
        glt_2_tmp_ds.SetProjection(glt_2_projection)
        glt_2_tmp_ds.SetMetadata(glt_2_metadata)

        glt_2_tmp_ds.GetRasterBand(1).WriteArray(glt_2[..., 0])
        glt_2_tmp_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_Undefined)
        glt_2_tmp_ds.GetRasterBand(1).SetDescription(glt_2_ds.GetRasterBand(1).GetDescription())

        glt_2_tmp_ds.GetRasterBand(2).WriteArray(glt_2[..., 1])
        glt_2_tmp_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_Undefined)
        glt_2_tmp_ds.GetRasterBand(2).SetDescription(glt_2_ds.GetRasterBand(2).GetDescription())
        glt_2_tmp_ds.FlushCache()
        glt_2_tmp_ds = None

        # It seems gdal has problems to set both bands' color interpretation to
        # undefined, so the glt2 tmp file can't be merged with the original glt1
        # file. As a workaround, we also generate a tmp glt1 file that has the
        # same color interpretation setting with the glt2 tmp file so that they
        # can be merged using gdalbuildvrt command.
        glt_1_filename = self.file_list[0].filename.replace('hdr', 'img')
        glt_1_tmp_file = os.path.join(tempfile.gettempdir(),
                                      f'{os.path.basename(glt_1_filename).replace(".img", "_tmp.tif")}')
        gdal.Translate(
            glt_1_tmp_file,
            glt_1_filename
        )
        glt_1_tmp_ds = gdal.Open(glt_1_tmp_file, gdal.GA_Update)
        glt_1_tmp_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_Undefined)
        glt_1_tmp_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_Undefined)
        glt_1_tmp_ds.FlushCache()
        glt_1_tmp_ds = None

        glt_vrt_ds = gdal.BuildVRT(
            '',
            [glt_1_tmp_file, glt_2_tmp_file],
            options=gdal.BuildVRTOptions(
                srcNodata=0,
                VRTNodata=0
            )
        )
        sample_data = glt_vrt_ds.GetRasterBand(1).ReadAsArray()
        line_data = glt_vrt_ds.GetRasterBand(2).ReadAsArray()

        combined_glt = np.stack([sample_data, line_data], axis=2)

        # Clean up the glt tmp files
        if os.path.exists(glt_1_tmp_file):
            os.remove(glt_1_tmp_file)
        if os.path.exists(glt_2_tmp_file):
            os.remove(glt_2_tmp_file)

        return combined_glt

    @property
    def date(self):
        return self.file_list[0].date

    @property
    def filename(self):
        return self.file_list[0].filename

    @property
    def filestr(self):
        return self.file_list[0].filestr

    @property
    def hdr(self):
        return self.file_list[0].hdr

    @property
    def map_info(self):
        combined_map_info = self.file_list[0].hdr['map info']
        for emit_file in self.file_list[1:]:
            individual_map_info = emit_file.hdr['map info']

            if float(individual_map_info[3]) < float(combined_map_info[3]):
                combined_map_info[3] = individual_map_info[3]

            if float(individual_map_info[4]) > float(combined_map_info[4]):
                combined_map_info[4] = individual_map_info[4]

            assert len(combined_map_info) == len(individual_map_info)
            assert combined_map_info[0] == individual_map_info[0]
            assert combined_map_info[1] == individual_map_info[1]
            assert combined_map_info[2] == individual_map_info[2]
            assert combined_map_info[5] == individual_map_info[5]
            assert combined_map_info[6] == individual_map_info[6]
            assert combined_map_info[7] == individual_map_info[7]

        return combined_map_info

    @property
    def id(self):
        return self.file_list[0].id

    @property
    def level(self):
        return self.file_list[0].level

    @property
    def orbit(self):
        return self.file_list[0].level

    @property
    def path(self):
        return self.file_list[0].path

    @property
    def path_and_filestr(self):
        return self.file_list[0].path_and_filestr

    @property
    def root(self):
        return self.file_list[0].root

    @property
    def scene(self):
        return self.file_list[0].scene

    @property
    def time(self):
        return self.file_list[0].time

    @property
    def type(self):
        return self.file_list[0].type

    @property
    def ver(self):
        return self.file_list[0].ver


class EMITAcquisitionSingleFile(object):
    """Gathers EMIT acquisition file, path, and high-level data access
    operations, where (path and) file name is assumed to be of the form
    <path>/emit<date>t<time>_<orbit>_<scene>_<level>_<type>_<ver>.<ext>.
    Further, and according to convention, the full <path> is assumed to be of
    the form <root>/<date>/emit<date>t<time>/<level>.

    Args:
        filestr (str): EMIT acquisition (path and) filename string.
            (default=None)
        **kwargs: Intead of filestr, individual file components may be provided
            and can be used as basis for constructing shell file-matching
            expressions.  Possible arguments include:
            root (str): EMIT acquisition file path root. (default=None)
            path (str): EMIT acquisition full file path (including root).
                (default=None)
            date (str or int): Acquistion date, in ISO yyyymmdd format.
                (default=None)
            time (str or int): Acquisition time, in ISO hhmmss format.
                (default=None)
            id (str):   EMIT acquisition path/file id consisting of, by
                convention, "emit<date>t<time>". (default=None)
            orbit (str or int): Five digit orbit number. (default=None)
            scene (str or int): Three digit scene number. (default=None)
            level (str): EMIT acquisition dataproduct level, e.g., 'l1a', 'l1b',
                'ghg', etc. (default=None)
            type (str): Acquisition type, e.g., 'ch4_mf', 'co2_mf', etc.
                (default=None)
            ver (str or int): Two digit version number. (default=None)
            ext (str):  File '.' + extension, e.g.,'.hdr', etc.  (default=None)

    Attributes/Properties:
        data (numpy.memmap): View of file's data.
        date (str): Date from either input filestr, or 'date' or 'id' kwarg.
        ext (str):  File extension, if applicable, from either input filestr
            or 'ext' kwarg.
        filename (str): Unique path and filename based on path_and_filestr
            directory lookup.
        filestr (str): File string from either input filestr, kwargs, or shell
            wildcard expression.
        path_and_filestr (str): (Path and) file string from either input
            filestr, kwargs, or shell wildcard expression.
        #file (str): File from input filestr, kwargs, or shell wildcard
        #    expression.
        #filestr (str): (Path and) filename from input, kwargs, or shell
        #    wildcard expression.
        hdr (dict): View of .hdr file data.
        id (str):   File id (emit<date>t<time> string) from input
            filestr, kwargs, or shell wildcard expression.
        level (str): Acquisition data processing level (e.g., 'l1a', 'l1b',
            etc.) from either input filestr or 'level' kwarg.
        orbit (str): Five digit orbit number from either input filestr or
            'orbit' kwarg.
        path (str): Path from either input filestr, kwargs, or shell wildcard
            expression.
        scene (str): Three digit scene number from either input filestr or
            'scene' kwarg.
        time (str): Time from either input filestr, or 'date' or 'id' kwarg.
        type (str): Type from either input filestr or 'type' kwarg.
        ver (str): Two digit acquisition processing version number from either
            input filestr or 'ver' kwarg.

    Remarks:
        If provided, 'id' will be used to set 'date' and 'time'.  If not, 'date'
        and 'time' will be used to set 'id'. If 'id' and 'date' and/or 'time'
        are provided, consistency checks will be performed.

    Raises:
        ValueError if both filestr and kwargs are provided, or if any argument
        conflicts are identified (e.g., inconsistent id and date strings, etc.)

    """
    file_fmt = '<path>/emit<date>t<time>_<orbit>_<scene>_<level>_<type>_<ver>.<ext>'
    path_fmt = '<root>/<date>/emit<date>t<time>/<level>/'

    def __init__( self, filestr=None, **kwargs):

        # directly-defined attributes:
        self.ext = None
        # attributes that may need to be, or could be, runtime evaluated:
        self._date  = self._file = self._filestr = self._id    = self._level = \
        self._orbit = self._path = self._root    = self._scene = self._time  = \
        self._type  = self._ver  = None

        if filestr and len(kwargs):
            raise ValueError(
                f"either 'filestr' or '**kwargs' may be provided, but not both")

        if filestr:

            self._filestr = filestr
            self._path  = os.path.dirname(filestr)
            self._file  = os.path.basename(filestr)
            _ext        = re.search('\\..*$',self._file).group()
            if _ext:
                self.ext = _ext
            else:
                self.ext = ''
            self._date  = re.search('\\d{8}',self._file).group()
            self._time  = re.search('\\d{6}_',self._file).group().rstrip('_')
            self._orbit = re.search('_o(\\d*)_',self._file).group(1)
            self._scene = re.search('_s(\\d*)_',self._file).group(1)
            level_type_ver_match_obj = re.search('_s\\d*_([a-zA-Z0-9]+)_(.*)_v(\\d*)',self._file)
            self._level = level_type_ver_match_obj.group(1)
            self._type  = level_type_ver_match_obj.group(2)
            self._ver   = level_type_ver_match_obj.group(3)

        elif len(kwargs):

            try:
                self._root  = kwargs.pop('root',None).rstrip('/')
            except:
                pass
            try:
                self._path  = kwargs.pop('path',None).rstrip('/')
            except:
                pass
            _date       = kwargs.pop('date',None)
            _date       = (str(_date) if _date else '')
            _time       = kwargs.pop('time',None)
            _time       = (str(_time) if _time else '')
            self._id    = kwargs.pop('id',None)
            if not self._id:
                self._date = _date
                self._time = _time
            else:
                if _date:
                    if _date == re.search('\\d{8}',self._id).group():
                        self._date = _date
                    else:
                        raise ValueError(
                            f"'date' argument ({_date}) does not match 'id' argument date ({id})")
                if _time:
                    if _time == re.search('\\d{6}$',self._id).group():
                        self._time = _time
                    else:
                        raise ValueError(
                            f"'time' argument ({_time}) does not match 'id' argument time ({id})")

            _orbit = kwargs.pop('orbit',None)
            if isinstance(_orbit,str):
                self._orbit = _orbit.lstrip('o')
            elif isinstance(_orbit,int):
                self._orbit = str(_orbit)

            _scene = kwargs.pop('scene',None)
            if isinstance(_scene,str):
                self._scene = _scene.lstrip('s')
            elif isinstance(_scene,int):
                self._scene = str(_scene)

            self._level = kwargs.pop('level',None)
            self._type  = kwargs.pop('type',None)

            _ver = kwargs.pop('ver',None)
            if isinstance(_ver,str):
                self._ver = _ver.lstrip('v')
            elif isinstance(_ver,int):
                self._ver = str(_ver)

            self.ext    = kwargs.pop('ext',None)
            if self.ext and not re.match('.',self.ext):
                    self.ext = '.' + self.ext

    @property
    def data(self):
        """Return a numpy.memmap view of the file's data.

        """
        ext_sav = self.ext
        if self.ext != '.hdr':
            self.ext = '.hdr'

        np_memmap_array = envi.open(self.filename).open_memmap()

        self.ext = ext_sav
        return np_memmap_array

    @property
    def glt_data(self):
        return self.data

    @property
    def date(self):
        # TODO: could also get date from path
        if self._date:
            return self._date
        elif self._file:
            return re.search('\\d{8}',self._file).group()
        elif self._id:
            return re.search('\\d{8}',self._id).group()
        else:
            return None

    @property
    def filename(self):
        """Return unique path and filename based on directory lookup.

        Raises:
            RuntimeError if nonunique filename found.

        """
        filestr_glob_results = glob.glob(self.path_and_filestr)
        if len(filestr_glob_results) == 1:
            return filestr_glob_results[0]
        else:
            e1 = f"search for file matching {self.path_and_filestr} returned non-unique result ({len(filestr_glob_results)} file(s) found)."
            e2 = f" Set 'ext' to obtain unique match."
            raise RuntimeError(e1+e2)

    @property
    def filestr(self):
        """Return emit<date>t<time>_<orbit>_<scene>_<level>_<type>_<ver>.<ext>
        string, or shell wildcard equivalent.

        """
        if self._filestr:
            return self._filestr
        else:
            return \
                self.id + '_' + \
                'o' + (self.orbit if self.orbit else '*') + '_' + \
                's' + (self.scene if self.scene else '*') + '_' + \
                (self.level if self.level else '*')       + '_' + \
                (self.type if self.type else '*')         + '_' + \
                'v' + (self.ver if self.ver else '*')           + \
                (self.ext if self.ext else '')

    @property
    def hdr(self):
        """Return dict view of .hdr file.

        """
        ext_sav = self.ext
        if self.ext != '.hdr':
            self.ext = '.hdr'

        hdr = envi.read_envi_header(self.filename)

        self.ext = ext_sav
        return hdr

    @property
    def map_info(self):
        return self.hdr['map info']

    @property
    def id(self):
        if self._id:
            return self._id
        else:
            return \
                'emit' + \
                (self.date if self.date else '*') + \
                't' + \
                (self.time if self.time else '*')

    @property
    def level(self):
        return self._level

    @property
    def orbit(self):
        return self._orbit

    @property
    def path(self):
        """Return '<root>/<date>/emit<date>t<time>/<level>/' expression from
        either input filestr or path, or expansion using available components or
        their wildcard equivalents.

        """
        if self._path:
            return self._path
        else:
            return \
                (self.root if self.root else '*') + '/' + \
                (self.date if self.date else '*') + '/' + \
                self.id                           + '/' + \
                (self.level if self.level else '*')

    @property
    def path_and_filestr(self):
        """Return '<path>/emit<date>t<time>_<orbit>_<scene>_<level>_<type>_<ver>.<ext>'
        expression from either input filestr, or expansion using available
        components or their wildcard equivalents.

        """
        if self._filestr:
            return self._filestr
        else:
            return ((self.path + '/') if self.path else '') + self.filestr

    @property
    def root(self):
        return self._root

    @property
    def scene(self):
        return self._scene

    @property
    def time(self):
        # TODO: could also get time from path
        if self._time:
            return self._time
        elif self._file:
            return re.search('\\d{6}_',self._file).group().rstrip('_')
        elif self._id:
            return re.search('\\d{6}$',self._id).group()
        else:
            return None

    @property
    def type(self):
        return self._type

    @property
    def ver(self):
        return self._ver


def EMITMatchedFilterFile(filestr=None, **kwargs):
    fids = kwargs['ids']
    if fids is None:
        raise KeyError(f'ids must be provided')

    if len(fids) == 1:
        mf_file = EMITMatchedFilterSingleFile(filestr, **kwargs)
    else:
        mf_file = EMITMatchedFilterMultiFile(filestr, **kwargs)

    return mf_file


class EMITMatchedFilterMultiFile(object):
    def __init__(self, filestr=None, **kwargs):
        fids = kwargs['ids']
        self.file_list = []

        for fid in fids:
            mf_file = EMITMatchedFilterSingleFile(
                root=kwargs['root'],
                id=fid,
                type=kwargs['type'],
                ext=kwargs['ext']
            )

            self.file_list.append(mf_file)

    @property
    def data(self):
        combined_data = []
        for mf_file in self.file_list:
            combined_data.append(mf_file.data)

        combined_data = np.concatenate(combined_data, axis=0)

        return combined_data

    @property
    def date(self):
        return self.file_list[0].date

    @property
    def filename(self):
        return self.file_list[0].filename

    @property
    def filestr(self):
        return self.file_list[0].filestr

    @property
    def hdr(self):
        return self.file_list[0].hdr

    @property
    def id(self):
        return self.file_list[0].id

    @property
    def path(self):
        return self.file_list[0].path

    @property
    def path_and_filestr(self):
        return self.file_list[0].path_and_filestr

    @property
    def root(self):
        return self.file_list[0].root

    @property
    def time(self):
        return self.file_list[0].time

    @property
    def type(self):
        return self.file_list[0].type


class EMITMatchedFilterSingleFile(object):
    """Gathers path, file, and high-level data access operations for EMIT
    matched filter, and associated matched filter, files, where (path and) file
    name is assumed to be of the form <path>/emit<date>t<time>_<type>.<ext>.
    Further, and according to convention, the full <path> is assumed to be of
    the form <root>/<date>/.

    Args:
        filestr (str): EMIT matched filter (path and) filename string.
            (default=None)
        **kwargs: Intead of filestr, individual file components may be provided
            and can be used as basis for constructing shell file-matching
            expressions.  Possible arguments include:
            root (str): EMIT matched filter file path root. (default=None)
            path (str): EMIT matched filter full file path (including root).
                (default=None)
            date (str or int): File date, in ISO yyyymmdd format. (default=None)
            time (str or int): File time, in ISO hhmmss format. (default=None)
            id (str):   EMIT matched filter file id consisting of, by
                convention, "emit<date>t<time>". (default=None)
            type (str): Matched filter type, e.g., 'ch4_mf', 'co2_mf', etc.
                (default=None)
            ext (str):  Matched filter file '.' + extension, e.g.,'.hdr', etc.
                (default=None)

    Attributes/Properties:
        data (numpy.memmap): View of file's data.
        date (str): Date from either input filestr, or date or id kwarg.
        ext (str):  File extension, if applicable, from either input filestr
            or kwarg.
        filename (str): Unique path and filename based on path_and_filestr
            directory lookup.
        filestr (str): File string from either input filestr, kwargs, or shell
            wildcard expression.
        path_and_filestr (str): (Path and) file string from either input
            filestr, kwargs, or shell wildcard expression.
        #file (str): File from input filestr, kwargs, or shell wildcard
        #    expression, whichever applies.
        #filestr (str): (Path and) filename either from input, kwargs, or shell
        #    wildcard expression, whichever applies.
        hdr (dict): View of .hdr file data.
        id (str):   File id (emit<date>t<time> string) from either input
            filestr, kwargs, or shell wildcard expression, whichever applies.
        path (str): Path from either input filestr, kwargs, or shell wildcard
            expression.
        time (str): Time from either input filestr, or date or id kwarg.
        type (type): Type from either input filestr or type kwarg.

    Remarks:
        If provided, 'id' will be used to set 'date' and 'time'.  If not, 'date'
        and 'time' will be used to set 'id'. If 'id' and 'date' and/or 'time'
        are provided, consistency checks will be performed.

    Raises:
        ValueError if both filestr and kwargs are provided, or if any argument
        conflicts are identified (e.g., inconsistent id and date strings, etc.)

    """
    file_fmt = '<path>/emit<date>t<time>_<type>.<ext>'
    path_fmt = '<root>/<date>'

    def __init__(self, filestr=None, **kwargs):

        # directly-defined attributes:
        self.ext = None
        # attributes that may need to be, or could be, runtime evaluated:
        self._date = self._file = self._filestr = self._id = self._path = \
        self._root = self._time = self._type = None

        if filestr and len(kwargs):
            raise ValueError(
                f"either 'filestr' or '**kwargs' may be provided, but not both")

        if filestr:

            self._filestr = filestr
            self._path  = os.path.dirname(filestr)
            self._file  = os.path.basename(filestr)
            self.ext    = os.path.splitext(filestr)[-1]
            self._date  = re.search('\\d{8}',self._file).group()
            self._time  = re.search('\\d{6}_',self._file).group().rstrip('_')
            self._type  = re.search('_.*',self._file).group().lstrip('_')
            if self.ext:
                self._type = self._type.replace(self.ext,'')

        elif len(kwargs):

            try:
                self._root  = kwargs.pop('root',None).rstrip('/')
            except:
                pass
            try:
                self._path  = kwargs.pop('path',None).rstrip('/')
            except:
                pass

            _date       = kwargs.pop('date',None)
            _date       = (str(_date) if _date else '')

            _time       = kwargs.pop('time',None)
            _time       = (str(_time) if _time else '')

            self._id    = kwargs.pop('id',None)
            if not self._id:
                self._date = _date
                self._time = _time
            else:
                if _date:
                    if _date == re.search('\\d{8}',self._id).group():
                        self._date = _date
                    else:
                        raise ValueError(
                            f"'date' argument ({_date}) does not match 'id' argument date ({id})")
                if _time:
                    if _time == re.search('\\d{6}$',self._id).group():
                        self._time = _time
                    else:
                        raise ValueError(
                            f"'time' argument ({_time}) does not match 'id' argument time ({id})")

            self._type  = kwargs.pop('type',None)

            self.ext    = kwargs.pop('ext',None)
            if self.ext and not re.match('.',self.ext):
                    self.ext = '.' + self.ext


    @property
    def data(self):
        """Return a numpy.memmap view of the file's data.

        """
        ext_sav = self.ext
        if self.ext != '.hdr':
            self.ext = '.hdr'

        np_memmap_array = envi.open(self.filename).open_memmap()

        self.ext = ext_sav
        return np_memmap_array


    @property
    def date(self):
        if self._date:
            return self._date
        elif self._file:
            return re.search('\\d{8}',self._file).group()
        elif self._id:
            return re.search('\\d{8}',self._id).group()
        else:
            return None


    @property
    def filename(self):
        """Return unique path and filename based on directory lookup.

        Raises:
            RuntimeError if nonunique filename found.

        """
        filestr_glob_results = glob.glob(self.path_and_filestr)
        if len(filestr_glob_results) == 1:
            return filestr_glob_results[0]
        else:
            e1 = f"search for file matching {self.path_and_filestr} returned non-unique result ({len(filestr_glob_results)} file(s) found)."
            e2 = f" Set 'ext' to obtain unique match."
            raise RuntimeError(e1+e2)


    @property
    def filestr(self):
        """Return 'emit<date>t<time>_<type>.<ext>' string, or shell wildcard
        equivalent.

        """
        if self._filestr:
            return self._filestr
        else:
            return \
                self.id + '_' + \
                (self.type if self.type else '*') + \
                (self.ext if self.ext else '')


    @property
    def hdr(self):
        """Return dict view of .hdr file.

        """
        ext_sav = self.ext
        if self.ext != '.hdr':
            self.ext = '.hdr'

        hdr = envi.read_envi_header(self.filename)

        self.ext = ext_sav
        return hdr


    @property
    def id(self):
        if self._id:
            return self._id
        else:
            return \
                'emit' + \
                (self.date if self.date else '*') + \
                't' + \
                (self.time if self.time else '*')


    @property
    def path(self):
        """Return '<root>/<date>' expression from either input filestr or path,
        or expansion using available components or their wildcard equivalents.

        """
        if self._path:
            return self._path
        else:
            return \
                (self.root if self.root else '*') + '/' + \
                (self.date if self.date else '*')


    @property
    def path_and_filestr(self):
        """Return '<path>/emit<date>t<time>_<type>.<ext>'
        expression from either input filestr, or expansion using available
        components or their wildcard equivalents.

        """
        if self._filestr:
            return self._filestr
        else:
            return ((self.path + '/') if self.path else '') + self.filestr


    @property
    def root(self):
        return self._root


    @property
    def time(self):
        if self._time:
            return self._time
        elif self._file:
            return re.search('\\d{6}_',self._file).group().rstrip('_')
        elif self._id:
            return re.search('\\d{6}$',self._id).group()
        else:
            return None


    @property
    def type(self):
        return self._type
