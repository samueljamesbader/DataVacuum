from datavac.database.db_create import ensure_clear_database
import pytest


@pytest.fixture(scope='package',autouse=True)
def mock_env_dem1():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("DATAVACUUM_CONTEXT", "builtin:demo1")
        yield
    #assert False


@pytest.fixture(scope='package',autouse=True)
def example_data(mock_env_dem1):
    from datavac.examples.demo1.example_data import make_example_data
    from datavac.util.caching import cli_clear_local_cache
    from datavac.database.db_create import create_all
    cli_clear_local_cache()
    ensure_clear_database()
    create_all()
    make_example_data()
