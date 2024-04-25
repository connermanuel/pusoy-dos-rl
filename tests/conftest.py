import pytest

from pusoy.models import DenseA2C


@pytest.fixture
def base_model_a2c():
    """Basic nondescript version of a model."""
    return DenseA2C()
