import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from addon.config import Config  # noqa: E402
from addon.fixtures import catalog, trakt_history  # noqa: E402
from addon.memory_store import MemoryStore, load_records_from_json  # noqa: E402
from addon.rows import build_user_state  # noqa: E402


@pytest.fixture(scope="session")
def fixture_catalog():
    return catalog(dim=384)


@pytest.fixture()
def store(fixture_catalog):
    st = MemoryStore()
    load_records_from_json(st, fixture_catalog)
    return st


@pytest.fixture()
def history():
    return trakt_history()


@pytest.fixture(scope="session")
def secret():
    return "unit-test-secret"


@pytest.fixture(autouse=True)
def install_secret(secret, monkeypatch):
    monkeypatch.setenv("INSTALL_SECRET", secret)


@pytest.fixture()
def config(secret):
    return Config(secret=secret, mode="blockbuster", langs=("en",))


@pytest.fixture()
def state(store, history):
    return build_user_state(store, history)
