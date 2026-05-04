"""
Polygon Regularization Package

A package for regularizing polygons by aligning edges to principal directions.
"""

from importlib.metadata import PackageNotFoundError, version

from .coordinator import regularize_geodataframe

try:
    __version__ = version("buildingregulariser")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Package-wide exports
__all__ = [
    "regularize_geodataframe",
    "__version__",
]
