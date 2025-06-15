from typing import TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from datavac.config.data_definition import DVColumn


class ReaderCard(): pass

@dataclass
class Trove(): 
    """A source of data to read into the database, e.g. a folder structure or other source database."""

    load_info_columns: list[DVColumn] = field(default_factory=list)
    """List of columns associated to a load from this trove."""

