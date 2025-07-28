"""
EMIT acquisition and matched filter file string and data access classes.

"""

import glob
import os
import re
from spectral.io import envi

#TODO: possible EMITFile base class for shared data, methods


class EMITAcquisitionFile(object):
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
        file (str): File from input filestr, kwargs, or shell wildcard
            expression.
        filestr (str): (Path and) filename from input, kwargs, or shell
            wildcard expression.
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

        filestr_glob_results = glob.glob(self.filestr)

        if len(filestr_glob_results) == 1:
            # unique file; read:
            np_memmap_array = envi.open(filestr_glob_results[0]).open_memmap()
        else:
            raise RuntimeError(
                f'search for file matching {self.filestr} returned non-unique result ({len(filestr_glob_results)} file(s) found)')

        self.ext = ext_sav
        return np_memmap_array


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
    def file(self):
        """Return emit<date>t<time>_<orbit>_<scene>_<level>_<type>_<ver>.<ext>
        string, or shell wildcard equivalent.

        """
        if self._file:
            return self._file
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
    def filestr(self):
        """Return
        '<path>/emit<date>t<time>_<orbit>_<scene>_<level>_<type>_<ver>.<ext>'
        expression from either input filestr, or expansion using available
        components or their wildcard equivalents.

        """
        if self._filestr:
            return self._filestr
        else:
            return ((self.path + '/') if self.path else '') + self.file


    @property
    def hdr(self):
        """Return dict view of .hdr file.

        """
        ext_sav = self.ext
        if self.ext != '.hdr':
            self.ext = '.hdr'

        filestr_glob_results = glob.glob(self.filestr)

        if len(filestr_glob_results) == 1:
            # unique file; read:
            hdr = envi.read_envi_header(filestr_glob_results[0])
        else:
            raise RuntimeError(
                f'search for file matching {self.filestr} returned non-unique result ({len(filestr_glob_results)} file(s) found)')

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


class EMITMatchedFilterFile(object):
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
        file (str): File from input filestr, kwargs, or shell wildcard
            expression, whichever applies.
        filestr (str): (Path and) filename either from input, kwargs, or shell
            wildcard expression, whichever applies.
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

    def __init__( self, filestr=None, **kwargs):

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

        filestr_glob_results = glob.glob(self.filestr)

        if len(filestr_glob_results) == 1:
            # unique file; read:
            np_memmap_array = envi.open(filestr_glob_results[0]).open_memmap()
        else:
            raise RuntimeError(
                f'search for file matching {self.filestr} returned non-unique result ({len(filestr_glob_results)} file(s) found)')

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
    def file(self):
        """Return 'emit<date>t<time>_<type>.<ext>' expression from input
        filestr, or expansion using available components or their shell wildcard
        equivalents.

        """
        if self._file:
            return self._file
        else:
            return self.id + \
                '_' + \
                (self.type if self.type else '*') + \
                (self.ext if self.ext else '')


    @property
    def filestr(self):
        """Return '<root>/<date>' expresssion from input filestr, or expansion
        using available components or their wildcard equivalents.

        """
        if self._filestr:
            return self._filestr
        else:
            return ((self.path + '/') if self.path else '') + self.file


    @property
    def hdr(self):
        """Return dict view of .hdr file.

        """
        ext_sav = self.ext
        if self.ext != '.hdr':
            self.ext = '.hdr'

        filestr_glob_results = glob.glob(self.filestr)

        if len(filestr_glob_results) == 1:
            # unique file; read:
            hdr = envi.read_envi_header(filestr_glob_results[0])
        else:
            raise RuntimeError(
                f'search for file matching {self.filestr} returned non-unique result ({len(filestr_glob_results)} file(s) found)')

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

