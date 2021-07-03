"""Test main `Cosa` class functionality.
"""
import pytest
from cosa import Cosa


def test_read():
    """Test reading functionality."""
    cosa = Cosa()
    # Existing file
    cosa.read("tests/test_files/miso.jpg")
    assert cosa.image is not None
    # Non-existing file
    assert cosa.image is not None
    with pytest.raises(ValueError):
        cosa.read("tests/test_files/fake.jpg")
