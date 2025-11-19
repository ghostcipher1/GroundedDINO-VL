from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("groundeddino-vl")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without installation)
    __version__ = "0.0.0"
