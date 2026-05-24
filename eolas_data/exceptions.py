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


# ---------------------------------------------------------------------------
# Bulk-download-specific exceptions (subclass APIError so callers that
# catch APIError still catch these; callers that want to handle bulk refusals
# specifically can match the narrower type).
# ---------------------------------------------------------------------------

class BulkUpgradeRequired(APIError):
    """Raised on HTTP 402: the requested freshness level requires a Pro plan."""

    def __init__(self, message: str = (
        "Fresh bulk downloads are a Pro feature. Free accounts get the latest "
        "monthly snapshot — see https://eolas.fyi/pricing."
    )):
        super().__init__(402, message)


class BulkLicenceRestricted(APIError):
    """Raised on HTTP 403 with a licence-restriction body from the bulk endpoint.

    The server detail (e.g. 'licence: OECD') is surfaced verbatim so the
    caller knows which dataset and why.
    """

    def __init__(self, message: str):
        super().__init__(403, message)


class BulkNotYetAvailable(APIError):
    """Raised on HTTP 503: the monthly snapshot for this dataset does not exist yet."""

    def __init__(self, message: str = (
        "Monthly bulk snapshots are still rolling out for this dataset. "
        "Try again after the 1st of next month, or upgrade to Pro for "
        "on-demand current snapshots — see https://eolas.fyi/pricing."
    )):
        super().__init__(503, message)
