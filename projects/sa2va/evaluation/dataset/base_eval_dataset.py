from torch.utils.data import Dataset
import copy
from collections.abc import Mapping
from typing import Union
import logging
from abc import abstractmethod


class BaseEvalDataset(Dataset):

    METAINFO: dict = dict(name='default')

    def __init__(self, metainfo: Union[Mapping, None] = None):
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

    @classmethod
    def _load_metainfo(cls,
                       metainfo: Union[Mapping, None] = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (Mapping, optional): Meta information dict.
                If ``metainfo`` contains existed filename, it will be
                parsed by ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, Mapping):
            raise TypeError('metainfo should be a Mapping or Config, '
                            f'but got {type(metainfo)}')

        for k, v in metainfo.items():
            if isinstance(v, str):
                # If type of value is string, and can be loaded from
                # corresponding backend. it means the file name of meta file.
                try:
                    with open(v, 'r') as f:
                        cls_metainfo[k] = [line.rstrip('\n') for line in f]
                except (TypeError, FileNotFoundError):
                    logging.getLogger(__name__).warning(
                        f'{v} is not a meta file, simply parsed as meta '
                        'information')
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo

    @property
    def metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return copy.deepcopy(self._metainfo)

    @abstractmethod
    def evaluate(self, results, work_dir):
        pass
