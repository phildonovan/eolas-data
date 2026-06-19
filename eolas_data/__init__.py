"""eolas-data — Python client for the eolas.fyi statistical data API."""
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

__version__ = "1.3.17"

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
    "ChangesUpgradeRequired",
    "ChangesLicenceRestricted",
    "WatermarkExpired",
]
