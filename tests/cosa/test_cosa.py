"""Test main `Cosa` class functionality.
"""
from cosa import Cosa


def test_read():
    """ Test reading functionality.
    """
    cosa = Cosa()
    # Existing file
    cosa.read("tests/test_files/miso.jpg")
    assert cosa.image is not None
