from importlib.metadata import PackageNotFoundError, version

from .lsss import LiStephensSurface

__all__ = ["LiStephensSurface"]
__version__ = version(__name__)
