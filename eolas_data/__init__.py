"""eolas-data — Python client for the eolas.fyi statistical data API."""
from .client import Client
from .dataset import Dataset
from .exceptions import APIError, AuthenticationError, EolasError, NotFoundError, RateLimitError

__version__ = "1.7.0"

__all__ = [
    "Client",
    "Dataset",
    "EolasError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "APIError",
]
