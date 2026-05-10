class EolasError(Exception):
    """Base exception for the eolas-data client."""


class AuthenticationError(EolasError):
    pass


class RateLimitError(EolasError):
    pass


class NotFoundError(EolasError):
    pass


class APIError(EolasError):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")
