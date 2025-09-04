from __future__ import annotations

import ast
import io
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol
from datavac.util.dvlogging import logger



if TYPE_CHECKING:
    from typing import Callable, Generic, TypeVar
    from tornado.web import RequestHandler
    import pandas as pd
    T=TypeVar('T',covariant=True)
    class RoutedCallable(Generic[T],Protocol):
        route: str
        return_type: str
        def __call__(self, *args, **kwargs) -> T: ...
        def __name__(self) -> str: ...


def as_api_handler_pd(func:RoutedCallable, role: str|None) -> type[RequestHandler]:
    from datavac.appserve.auth import UsePanelAuthMixin
    serverside_return_handler = {'pd': _serverside_return_handler_pd, 'ast': _serverside_return_handler_ast}[func.return_type]
    class Handler(UsePanelAuthMixin):
        require_token_method: bool = True
        async def post_authorized(self, user_info: dict[str, Any]):
            import pandas as pd
            import inspect
            try:
                payload = ast.literal_eval(self.request.body.decode('utf-8'))
                args= payload.get('args', []); kwargs= payload.get('kwargs', {})
                sig=inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                assert 'user_info' not in bound.arguments, "user_info is a reserved argument name"
                assert 'conn' not in bound.arguments, "conn is a reserved argument name"
                if 'user_info' in sig.parameters: kwargs['user_info'] = user_info
                result=func(*args,**kwargs)
                self.set_status(200)
                self.set_header('Content-Type', 'application/octet-stream')
                try:
                    bts=serverside_return_handler(result)
                except Exception as e:
                    raise ValueError(f"Error converting return value of {func.__name__}") from e
                self.write(bts)
            except Exception as e:
                import traceback
                self.set_status(400)
                self.write(f"Error processing request: {traceback.format_exception(e)}".encode('utf-8'))
        @property
        def required_role(self) -> str|None: return role
    return Handler


def client_server_split(method_name:str,return_type:str='pd',split_on:str='direct_db_access') -> Callable[[Callable[...,T]],RoutedCallable[T]]:
    clientside_return_handler = {'pd': _clientside_return_handler_pd, 'ast': _clientside_return_handler_ast}[return_type]
    from datavac.config.project_config import PCONF
    split_func={'direct_db_access': lambda: PCONF().direct_db_access,
                'is_server': lambda: PCONF().is_server
                }[split_on]
    def as_client_function(func, route)->Callable[... , pd.DataFrame]:
        @wraps(func)
        def wrapper(*args, client_retries=1, **kwargs):
            import requests
            from datavac.config.project_config import PCONF
            from datavac.appserve.dvsecrets.ak_client_side import get_access_key
            resp= requests.post(PCONF().deployment_uri+'/api/'+route, data=str({'args':args, 'kwargs':kwargs}),
                         headers={'Authorization':f'Bearer {get_access_key()}'},
                         verify=(PCONF().cert_depo.get_ssl_rootcert_path_for_deployment() or False))
            if resp.status_code!=200:
                if (resp.status_code in [401,403]) and (client_retries>0):
                    print(resp.text)
                    logger.debug("Refreshing access key to retrying API call due to error...")
                    from datavac.appserve.dvsecrets.ak_client_side import refresh_access_key
                    refresh_access_key(logout_first=True)
                    return wrapper(*args, client_retries=client_retries-1, **kwargs)
                else:
                    print(resp.text)
                    if resp.status_code==401:
                        if 'expired' in resp.text.lower():
                            from datavac.appserve.dvsecrets.ak_client_side import AccessKeyExpiredError
                            raise AccessKeyExpiredError(resp.text)
                        else:
                            from datavac.appserve.dvsecrets.ak_client_side import AccessKeyInvalidError
                            raise AccessKeyInvalidError(resp.text)
                    elif resp.status_code==403:
                        from datavac.appserve.dvsecrets.ak_client_side import AccessKeyPermissionError
                        raise AccessKeyPermissionError(resp.text)
                    else:
                        raise Exception(f"Error calling {func.__name__}: {resp.status_code} {resp.text}")
            return clientside_return_handler(resp.content)
        return wrapper
    def factory(func:Callable)->RoutedCallable:
        @wraps(func)
        def wrapper(*args, client_retries=1, **kwargs):
            if split_func():
                from pydantic import validate_call
                return validate_call(func)(*args, **kwargs)
            else: return as_client_function(func, method_name)(*args, client_retries=client_retries, **kwargs)
        wrapper.route='/api/'+method_name # type: ignore
        wrapper.return_type = return_type # type: ignore
        return wrapper # type: ignore
    return factory

def _serverside_return_handler_pd(result)->bytes:
    import pandas as pd
    if not isinstance(result,pd.DataFrame):
        raise ValueError(f"Function did not return a pandas DataFrame")
    outbuf=io.BytesIO()
    result.to_parquet(outbuf, index=False)
    outbuf.seek(0)
    return outbuf.getvalue()
def _clientside_return_handler_pd(bts:bytes)->pd.DataFrame:
    import pandas as pd
    return pd.read_parquet(io.BytesIO(bts))

def _serverside_return_handler_ast(result)->bytes:
    return str(result).encode('utf-8')
def _clientside_return_handler_ast(bts:bytes)->Any:
    return ast.literal_eval(bts.decode('utf-8'))