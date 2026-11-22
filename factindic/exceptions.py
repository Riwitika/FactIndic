"""Standard exception hierarchy for FactIndic research pipeline.

Distinguishes between transient rate limits, permanent quota exhaustion,
authentication failures, timeouts, and malformed model outputs.
"""

from typing import Optional


class FactIndicError(Exception):
    """Base class for all FactIndic errors."""
    pass


class AuthenticationError(FactIndicError):
    """Raised when API credentials are missing, rejected, or invalid (HTTP 401/403)."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.provider = provider


class RateLimitError(FactIndicError):
    """Raised when a temporary/burst rate limit is encountered (HTTP 429 with Retry-After).

    This error is transient and retryable after the backoff duration.
    """

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        status_code: Optional[int] = 429,
        response_body: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = status_code
        self.response_body = response_body
        self.provider = provider


class QuotaExhaustedError(FactIndicError):
    """Raised when an account-level or monthly quota has been exhausted.

    This error is NON-RETRYABLE within the current billing/quota period (e.g. Cohere
    trial key limited to 1,000 API calls / month). Execution must stop cleanly.
    """

    def __init__(
        self,
        message: str,
        response_body: Optional[str] = None,
        provider: Optional[str] = None,
        quota_type: str = "monthly_limit",
    ):
        super().__init__(message)
        self.response_body = response_body
        self.provider = provider
        self.quota_type = quota_type


class InferenceTimeoutError(FactIndicError):
    """Raised when an inference API call times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        provider: Optional[str] = None,
    ):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.provider = provider


class MalformedResponseError(FactIndicError):
    """Raised when model response is not valid JSON or fails schema constraints."""

    def __init__(
        self,
        message: str,
        raw_output: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        super().__init__(message)
        self.raw_output = raw_output
        self.provider = provider
