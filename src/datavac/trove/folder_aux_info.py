from enum import Enum
from typing import NamedTuple
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from datavac.trove.trove_util import get_cached_glob

class MissingFolderInfoException(Exception): pass

class ReadAction(Enum):
    """Enum-like class for read actions."""
    STORE_PATH = 'store_path'
    STORE_BASENAME = 'store_basename'
    #APPLY_MATERIAL_LUT = 'apply_material_lut'

AuxInfoItem = NamedTuple('AuxInfoItem', [('name', str),
                                        ('read_action', ReadAction),
                                        ('filter', str),
                                        ('count', str)])
class FolderAuxInfoReader():

    def __init__(self, items: Sequence[AuxInfoItem]=(),
                 folder_name_reader: Callable[[Path, dict[str,Any]], dict[str,Any]]=lambda folder, info_already_known: {}):
        self._items = items
        self._folder_name_reader = folder_name_reader

    def read(self, folder: Path,
             cached_glob: Optional[Callable[[Path,str],list[Path]]] = None,
             info_already_known: Optional[dict[str,Any]] = None,
             super_folder: Optional[Path]=None) -> dict:
        
        read_from_folder:dict[str,Any]=info_already_known.copy() if info_already_known else {}
        read_from_folder=self._folder_name_reader(folder, read_from_folder)
        
        for i in self._items:
            cached_glob = cached_glob or get_cached_glob()
            assert i.count in ['required','optional']
            if len(potential_finds:=list(cached_glob(folder,(i.filter))))==1:
                match i.read_action:
                    case ReadAction.STORE_PATH:
                        read_from_folder[i.name]=potential_finds[0]
                    case ReadAction.STORE_BASENAME:
                        read_from_folder[i.name]=potential_finds[0].name.split(".")[0]
                    #case 'apply_material_lut':
                    #    read_from_folder['material_lut']=\
                    #        pd.read_csv(potential_finds[0]).set_index(FULLNAME_COL).to_dict(orient='index')
            else:
                if i.count=='required':
                    raise MissingFolderInfoException(
                        f"Found {len(potential_finds)} options for {i.name} "\
                            f"(filter '{i.filter}') in \"{str(folder.relative_to(super_folder) if super_folder else folder)}'\"")

        return read_from_folder
    