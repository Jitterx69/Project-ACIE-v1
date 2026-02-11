"""ACIE gRPC module"""

try:
    from . import acie_pb2
    from . import acie_pb2_grpc
    from .server import serve
except ImportError:
    # Allow import even if protobufs aren't generated yet
    pass
