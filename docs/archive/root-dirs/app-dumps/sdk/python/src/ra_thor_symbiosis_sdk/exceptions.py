"""Custom exceptions for Ra-Thor Symbiosis SDK"""

class RaThorError(Exception):
    """Base exception for all Ra-Thor SDK errors"""
    pass

class HandshakeError(RaThorError):
    """Raised when a handshake fails or cannot be completed"""
    pass

class ValenceError(RaThorError):
    """Raised when valence operations fail"""
    pass

class OntologyError(RaThorError):
    """Raised when ontology mapping fails"""
    pass

class ConnectionError(RaThorError):
    """Raised when connection to Ra-Thor fails"""
    pass

class AuthenticationError(RaThorError):
    """Raised when authentication fails"""
    pass