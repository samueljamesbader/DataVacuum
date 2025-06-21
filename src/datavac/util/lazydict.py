from collections.abc import Mapping
from typing import Any, Callable

def _default_splitter(key: str) -> tuple[str, str]:
    """
    Default splitter function that splits the key into category and subkey.
    It assumes the key is in the format 'category subkey'.
    """
    parts = key.split(maxsplit=1)
    assert len(parts) == 2, f"Key '{key}' does not contain a category and subkey separated by a space."
    return parts[0], parts[1]

def _default_joiner(category: str, subkey: str) -> str:
    """ Default joiner function that joins category and subkey into a single key.
    It assumes the key is in the format 'category subkey'.
    """
    return f"{category} {subkey}"

class CategorizedLazyDict(Mapping):
    def __init__(self,
                 category_getters: dict[str, Callable[[], dict[str, Any]]],
                 category_splitter: Callable[[str], tuple[str,str]] = _default_splitter,
                 category_joiner: Callable[[str, str], str] = _default_joiner
                 ):
        self._category_getters = category_getters
        self._category_splitter = category_splitter
        self._category_joiner = category_joiner
        self._data = {}

    def _load_category(self, category: str) -> None:
        """Load the data for a specific category if it hasn't been loaded yet."""
        if category not in self._data:
            self._data[category] = self._category_getters[category]()

    def __getitem__(self, key: str) -> Any:
        category, subkey = self._category_splitter(key)
        self._load_category(category)
        return self._data[category][subkey]
    
    def __iter__(self):
        for category in self._category_getters:
            self._load_category(category)
            yield from (self._category_joiner(category, subkey) for subkey in self._data[category])

    def __len__(self) -> int:
        return sum(len(self._data[category]) for category in self._category_getters if category in self._data)
    
    def __contains__(self, key: str) -> bool:
        category, subkey = self._category_splitter(key)
        if category not in self._category_getters: return False
        else:
            self._load_category(category)
            return subkey in self._data[category]
 
class FunctionLazyDict(Mapping):
    def __init__(self, getter: Callable[[str], Any], keylister: Callable[[], list[str]]):
        """
        A lazy dictionary that retrieves values using a function.
        
        Args:
            getter: A function that takes a key and returns the corresponding value.
            keylister: A function that returns a list of keys.
        """
        self._getter = getter
        self._keylister = keylister
        self._data = {}

    def _load_key(self, key: str) -> None:
        """Load the value for a specific key if it hasn't been loaded yet."""
        if key not in self._data:
            self._data[key] = self._getter(key)
    def __getitem__(self, key: str) -> Any:
        self._load_key(key)
        return self._data[key]
    def __iter__(self):
        yield from self._keylister()
    def __len__(self) -> int:
        return len(self._keylister())
    def __contains__(self, key: str) -> bool:
        return key in self._keylister() 