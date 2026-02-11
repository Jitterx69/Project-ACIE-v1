"""
Custom exceptions for ACIE SDK
"""


class ACIEError(Exception):
    """Base exception for all ACIE SDK errors."""
    pass


class ACIEConnectionError(ACIEError):
    """Raised when connection to ACIE server fails."""
    pass


class ACIEAuthenticationError(ACIEError):
    """Raised when authentication fails."""
    pass


class ACIENotFoundError(ACIEError):
    """Raised when requested resource is not found."""
    pass


class ACIEServerError(ACIEError):
    """Raised when server returns an error."""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class ACIEValidationError(ACIEError):
    """Raised when request validation fails."""
    pass


class ACIETimeoutError(ACIEError):
    """Raised when request times out."""
    pass
