"""Pytest configuration: apply emle-mace data-layer patches for the test suite."""

import pytest
import emle_mace.data as _emle_data


@pytest.fixture(autouse=True, scope="session")
def patch_mace_data():
    """Patch mace's AtomicData so EMLE fields are populated in all tests."""
    _emle_data.patch()
    yield
    _emle_data.restore()