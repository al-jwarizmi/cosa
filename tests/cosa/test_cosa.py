"""Test main `Cosa` class functionality.
"""
import pytest
from cosa import Cosa
from pathlib import Path


def test_read():
    """Test reading functionality."""
    input_path = "tests/test_files/miso.jpg"
    fake_path = "tests/test_files/fake.jpg"
    cosa = Cosa()
    # Existing file
    cosa.read(input_path)
    # Non-existing file
    assert cosa.image is not None
    with pytest.raises(ValueError):
        cosa.read(fake_path)


def test_write():
    """Test saving functionality."""
    input_path = "tests/test_files/miso.jpg"
    output_path = "tests/test_files/miso_write.jpg"
    cosa = Cosa()
    # Existing file
    cosa.read(input_path)
    # Non-existing file
    assert cosa.image is not None
    cosa.transform("elastic")
    assert cosa.transformed is not None
    cosa.write(output_path)
    assert Path(output_path).is_file()


def test_k_rep():
    """Test `k_rep` transformation."""
    input_path = "tests/test_files/miso.jpg"
    cosa = Cosa()
    # Existing file
    cosa.read(input_path)
    # Non-existing file
    assert cosa.image is not None
    cosa.transform("k_rep")
    assert cosa.transformed is not None


def test_elastic():
    """Test `elastic` transformation."""
    input_path = "tests/test_files/miso.jpg"
    cosa = Cosa()
    # Existing file
    cosa.read(input_path)
    # Non-existing file
    assert cosa.image is not None
    cosa.transform("elastic")
    assert cosa.transformed is not None
