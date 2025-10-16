class GatewayError(Exception):
    """Base exception for gateway-level errors."""


class QueueTimeoutError(GatewayError):
    """Raised when a request response is not received within the wait window."""


class RequestRejectedError(GatewayError):
    """Raised when the downstream service reports a failure for the request."""


class ServiceNotConfiguredError(GatewayError):
    """Raised when a requested service does not have a configuration block."""
