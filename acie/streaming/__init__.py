"""ACIE streaming module"""

try:
    from .kafka_client import (
        ACIEEventProducer,
        ACIEEventConsumer,
        get_producer
    )
except ImportError:
    pass
