try:
    from ._version import version as __version__
except ImportError:
    import importlib.metadata
    __version__ = importlib.metadata.version("napari-biopb")
except:
    __version__ = "unknown"

from ._widget import BiopbImageWidget

__all__ = ("BiopbImageWidget",)
