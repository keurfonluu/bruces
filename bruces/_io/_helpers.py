import os

__all__ = [
    "register",
    "read",
    "write",
]


_extension_to_filetype = {}
_reader_map = {}
_writer_map = {}


def register(file_format, extensions, reader, writer=None):
    """Register new catalog format."""
    for ext in extensions:
        _extension_to_filetype[ext] = file_format

    if reader is not None:
        _reader_map[file_format] = reader

    if writer is not None:
        _writer_map[file_format] = writer


def read(filename, file_format=None, **kwargs):
    """
    Read catalog.

    Parameters
    ----------
    filename : str, pathlike or buffer
        Input file name or buffer.
    file_format : str ('csv') or None, optional, default None
        Input file format.

    Returns
    -------
    :class:`bruces.Catalog`
        Earthquake catalog.
    
    """
    if file_format is None:
        file_format = get_file_format(filename)

    if file_format not in _reader_map:
        raise ValueError()
        
    return _reader_map[file_format](filename, **kwargs)


def write(filename, catalog, file_format=None, **kwargs):
    """
    Write catalog to CSV file.

    Parameters
    ----------
    filename : str, pathlike or buffer
        Output file name or buffer.
    catalog : :class:`bruces.Catalog`
        Earthquake catalog to export.
    file_format : str ('csv') or None, optional, default None
        Output file format.
    
    """
    if file_format is None:
        file_format = get_file_format(filename)

    if file_format not in _writer_map:
        raise ValueError()

    _writer_map[file_format](filename, catalog, **kwargs)


def get_file_format(filename):
    """Get file format."""
    ext = os.path.splitext(filename)[-1]

    return (
        _extension_to_filetype[ext]
        if ext in _extension_to_filetype
        else None
    )
