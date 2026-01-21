from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, cast


if TYPE_CHECKING:
    import panel.theme
    from tornado.web import RequestHandler
    from datavac.appserve.api import RoutedCallable


@dataclass
class ScheduledFunction:
    name: str; func: Callable; period: str; delay: str

@dataclass
class ServerConfig:
    port: int = 3000
    read_role: str = 'read'
    api_roles: dict[str,str|None]= field(default_factory=lambda:{
        '/api/get_data': 'read',
        '/api/get_factors': 'read',
        '/api/get_sweeps_for_jmp': 'read',
        '/api/get_user': None,
        '/api/get_flow_names': 'read',
        '/api/get_split_table': 'read',
        '/api/get_mgoa_names': 'read',
        '/api/get_available_columns': 'read',
        '/api/read_only_sql': 'read',
        })
    access_key_expiry_days:float = 14

    def get_theme(self) -> type[panel.theme.Theme]:
        from panel.theme.material import MaterialDefaultTheme
        return MaterialDefaultTheme
    
    def get_scheduled_functions(self) -> list[ScheduledFunction]:
        return []

    def pre_serve_setup(self): pass

    @property
    def potential_api_funcs(self) -> list[RoutedCallable]:
        from datavac.database.db_get import get_data, get_factors, get_sweeps_for_jmp, get_mgoa_names, get_available_columns
        from datavac.appserve.dvsecrets.ak_client_side import get_user
        from datavac.config.sample_splits import get_flow_names, get_split_table
        from datavac.database.db_util import read_only_sql
        return [get_data, get_factors, get_user, get_sweeps_for_jmp, get_flow_names, get_split_table,
                get_mgoa_names, get_available_columns, read_only_sql]

    @property
    def api_funcs_and_roles(self) -> dict[RoutedCallable,str|None]:
        role_map:dict[str|None,str|None]={'read': self.read_role}
        return {f:role_map.get(self.api_roles[f.route],self.api_roles[f.route])
                               for f in self.potential_api_funcs if f.route in self.api_roles}

    def get_additional_handlers(self) -> dict[str, type[RequestHandler]]:
        from datavac.util.util import import_modfunc
        from datavac.config.contexts import get_context_download_request_handler
        from tornado.web import RequestHandler
        ah={route:cast(type[RequestHandler],import_modfunc(handler)) for route,handler in self.get_yaml().get("additional_handlers",{}).items()}
        ah['/context']=get_context_download_request_handler()
        return ah
    
    def get_all_handlers(self) -> dict[str, type[RequestHandler]]:
        from datavac.config.project_config import PCONF
        assert PCONF().direct_db_access, "Something's wrong if we're launching a server that does not have direct DB access..."
        from datavac.appserve.api import as_api_handler_pd
        main_handlers = {f.route: as_api_handler_pd(f, role) for f,role in self.api_funcs_and_roles.items()} 
        return dict(**main_handlers,**self.get_additional_handlers())
    
    def get_yaml(self) -> dict:
        return {}