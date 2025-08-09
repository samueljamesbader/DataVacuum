from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import panel.theme
    from tornado.web import RequestHandler


@dataclass
class ScheduledFunction:
    name: str; func: Callable; period: str; delay: str

class ServerConfig:
    port: int = 3000

    def get_theme(self) -> type[panel.theme.Theme]:
        from panel.theme.material import MaterialDefaultTheme
        return MaterialDefaultTheme
    
    def get_scheduled_functions(self) -> list[ScheduledFunction]:
        return []

    def pre_serve_setup(self): pass

    def get_additional_handlers(self) -> dict[str, type[RequestHandler]]:
        return {}