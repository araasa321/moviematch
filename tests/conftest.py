import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from addon.config import Config  # noqa: E402
from addon.fixtures import catalog, trakt_history  # noqa: E402
from addon.memory_store import MemoryStore, load_records_from_json  # noqa: E402
from addon.rows import build_user_state  # noqa: E402


def project_payload(payload: dict, with_payload) -> dict:
    """Emulate Qdrant's `with_payload` selector for the fake clients.

    Every fake `search()` used to return the WHOLE payload whatever was asked
    for, so no existing test could notice a field being dropped from the #2b
    allowlist. Honouring a list here means all ~290 row/rank tests now run
    against a projected payload: if a ranking term silently depends on a dropped
    field, the test that exercises it fails immediately instead of degrading in
    production only.

    `True` / `False` keep their old meaning here. `False` still returns the
    payload because that is what the existing scroll paths expect — do not
    change those.
    """
    if isinstance(with_payload, (list, tuple)):
        return {k: v for k, v in payload.items() if k in with_payload}
    return payload


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
