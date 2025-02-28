"""Main config file for tests"""

import pytest
import numpy as np


@pytest.fixture(scope="session", name="seed")
def seed_fixture() -> int:
    """Seed used for reproducibility"""
    return 12345


@pytest.fixture(scope="session", name="rng")
def rng_fixture(seed: int) -> np.random.Generator:
    """Seeded Random number generator"""
    return np.random.default_rng(seed=seed)
