from .refVOS import RefVOSDataset

__all__ = ['RefVOSDataset', 'RESDataset']


def __getattr__(name):
    # RefVOS evaluation should not require the optional RefCOCO visualization
    # stack (matplotlib, pycocotools helpers). Load it only for image RES jobs.
    if name == 'RESDataset':
        from .RES import RESDataset
        return RESDataset
    raise AttributeError(name)
