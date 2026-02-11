"""
Structured logging for ACIE with request tracing
Provides JSON-formatted logs with correlation IDs and contextual information
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
import uuid

# Context variable for request ID tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_data['request_id'] = request_id
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        
        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger with contextual information"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with JSON formatter"""
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method"""
        extra_data = kwargs
        self.logger._log(level, message, (), extra={'extra_data': extra_data})
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, **kwargs)


# Global logger instance
logger = StructuredLogger('acie')


def set_request_id(request_id: Optional[str] = None):
    """Set request ID in context"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def get_request_id() -> Optional[str]:
    """Get current request ID from context"""
    return request_id_var.get()


def log_request(method: str, path: str, status_code: int, latency_ms: float, **kwargs):
    """Log HTTP request"""
    logger.info(
        f"{method} {path} {status_code}",
        method=method,
        path=path,
        status_code=status_code,
        latency_ms=latency_ms,
        **kwargs
    )


def log_inference(
    model_version: str,
    observation_shape: tuple,
    intervention: Dict[str, Any],
    latency_ms: float,
    success: bool = True,
    **kwargs
):
    """Log inference event"""
    logger.info(
        "Inference completed",
        model_version=model_version,
        observation_shape=list(observation_shape),
        intervention=intervention,
        latency_ms=latency_ms,
        success=success,
        **kwargs
    )


def log_error(error_type: str, error_message: str, **kwargs):
    """Log error event"""
    logger.error(
        f"Error occurred: {error_type}",
        error_type=error_type,
        error_message=error_message,
        **kwargs
    )


# FastAPI middleware for request logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate and set request ID
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        set_request_id(request_id)
        
        # Log request start
        start_time = time.time()
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_host=request.client.host if request.client else None,
            user_agent=request.headers.get('user-agent')
        )
        
        # Process request
        try:
            response = await call_next(request)
            latency_ms = (time.time() - start_time) * 1000
            
            # Log request completion
            log_request(
                method=request.method,
                path=str(request.url.path),
                status_code=response.status_code,
                latency_ms=latency_ms
            )
            
            # Add request ID to response headers
            response.headers['X-Request-ID'] = request_id
            
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                method=request.method,
                path=str(request.url.path),
                latency_ms=latency_ms
            )
            raise
