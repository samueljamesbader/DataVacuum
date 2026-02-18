import pytest

@pytest.fixture(scope='package',autouse=True)
def mock_env_demo3():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("DATAVACUUM_CONTEXT", "builtin:demo3")
        from datavac import unload_my_imports; unload_my_imports()
        print("set up mock environment for demo3")
        yield
        from datavac import unload_my_imports; unload_my_imports()

@pytest.fixture(scope='package',autouse=True)
def prep_db(mock_env_demo3): _prep_db()
def _prep_db():
    from datavac.database.db_create import ensure_clear_database
    from datavac.util.caching import cli_clear_local_cache
    from datavac.database.db_create import create_all
    from datavac.database.db_semidev import upload_mask_info
    cli_clear_local_cache()
    ensure_clear_database()
    create_all()
    upload_mask_info({'Mask1': [None,None]})
