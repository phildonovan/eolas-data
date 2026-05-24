"""eolas-data — Python client for the eolas.fyi statistical data API."""
from .client import Client, SyncResult
from .dataset import Dataset
from .exceptions import (
    APIError,
    AuthenticationError,
    BulkLicenceRestricted,
    BulkNotYetAvailable,
    BulkUpgradeRequired,
    EolasError,
    NotFoundError,
    RateLimitError,
)

__version__ = "1.0.0"

__all__ = [
    "Client",
    "Dataset",
    "SyncResult",
    "EolasError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "APIError",
    "BulkUpgradeRequired",
    "BulkLicenceRestricted",
    "BulkNotYetAvailable",
]
