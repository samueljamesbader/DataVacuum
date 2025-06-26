from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from datavac.config.layout_params import LayoutParameters

if TYPE_CHECKING:
    from pathlib import Path
    from sqlalchemy import Connection


_layout_params:Optional[LayoutParameters]=None
def get_folder_layout_params(layout_params_dir: Path, layout_params_yaml: Path,
                             force_regenerate=False, conn:Optional[Connection]=None,) -> LayoutParameters:
    from datavac.io.layout_params import LayoutParameters as FolderLayoutParameters
    from datavac.util.caching import pickle_db_cached
    global _layout_params
    if force_regenerate or (_layout_params is None):
        from datavac.util.caching import pickle_db_cached
        _layout_params,_layout_params_timestamp=\
            pickle_db_cached('LayoutParams',namespace='vac',conn=conn)(FolderLayoutParameters)(
                params_dir=layout_params_dir,yaml_path=layout_params_yaml,force=force_regenerate)
    return _layout_params # type: ignore