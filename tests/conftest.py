import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from waveforge import db as dbmod, ingest, scenario, store  # noqa: E402

PKG = os.path.join(os.path.dirname(__file__), "..", "casepkg", "wave-overlay-demo")
CASE = "wave-overlay-demo"


@pytest.fixture
def conn(tmp_path):
    c = dbmod.open_db(str(tmp_path / "t.db"))
    yield c
    c.close()


@pytest.fixture
def loaded(conn):
    ingest.load_case_package(conn, PKG)
    return conn


@pytest.fixture
def scenariod(loaded):
    for _ in range(7):
        r = scenario.apply_next_step(loaded, PKG, CASE)
        loaded.commit()
    return loaded


@pytest.fixture
def client(tmp_path):
    app = create_app(str(tmp_path / "api.db"))
    return TestClient(app)
