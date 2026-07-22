"""eolas-data — Python client for the eolas.fyi statistical data API."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .cdc import merge_changes
from .client import Client, SyncResult
from .dataset import Dataset
from .exceptions import (
    APIError,
    AuthenticationError,
    BulkLicenceRestricted,
    BulkNotYetAvailable,
    BulkUpgradeRequired,
    ChangesLicenceRestricted,
    ChangesUpgradeRequired,
    EolasError,
    NotFoundError,
    RateLimitError,
    WatermarkExpired,
)

# Single source of truth: read the installed distribution's version so
# __version__, `eolas version`, and the User-Agent can never drift from
# pyproject again (REL-1). Falls back to the pyproject value only for an
# editable tree that was never `pip install`-ed.
try:
    __version__ = _pkg_version("eolas-data")
except PackageNotFoundError:  # pragma: no cover - uninstalled source checkout
    __version__ = "1.9.0"

__all__ = [
    "Client",
    "Dataset",
    "SyncResult",
    "merge_changes",
    "EolasError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "APIError",
    "BulkUpgradeRequired",
    "BulkLicenceRestricted",
    "BulkNotYetAvailable",
    "ChangesUpgradeRequired",
    "ChangesLicenceRestricted",
    "WatermarkExpired",
]
