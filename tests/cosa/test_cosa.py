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
    cosa.transform("intarsia", height=50, width=50, num_colours=4)
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


def test_jpeg():
    """Test `jpeg` transformation."""
    input_path = "tests/test_files/miso.jpg"
    cosa = Cosa()
    # Existing file
    cosa.read(input_path)
    # Non-existing file
    assert cosa.image is not None
    cosa.transform("jpeg")
    assert cosa.transformed is not None


def test_voronoi():
    """Test `voronoi` transformation."""
    input_path = "tests/test_files/miso.jpg"
    cosa = Cosa()
    # Existing file
    cosa.read(input_path)
    # Non-existing file
    assert cosa.image is not None
    cosa.transform("voronoi")
    assert cosa.transformed is not None


def test_dither():
    """Test `dither` transformation."""
    input_path = "tests/test_files/miso.jpg"
    cosa = Cosa()
    # Existing file
    cosa.read(input_path)
    # Non-existing file
    assert cosa.image is not None
    cosa.transform("dither")
    assert cosa.transformed is not None


def test_intarsia():
    """Test `intarsia` transformation."""
    input_path = "tests/test_files/miso.jpg"
    cosa = Cosa()
    # Existing file
    cosa.read(input_path)
    # Non-existing file
    assert cosa.image is not None
    cosa.transform("intarsia", height=50, width=50, num_colours=4)
    assert cosa.transformed is not None
