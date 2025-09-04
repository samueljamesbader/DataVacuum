import pytest

@pytest.fixture(scope='package',autouse=True)
def mock_env_demo2():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("DATAVACUUM_CONTEXT", "builtin:demo2")
        from datavac import unload_my_imports; unload_my_imports()
        print("set up mock environment for demo2")
        yield
        from datavac import unload_my_imports; unload_my_imports()