from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch
from acie.logging.structured_logger import logger

class RetrievalStrategy(ABC):
    """
    Abstract Base Class for Context Retrieval Strategies.
    """
    @abstractmethod
    def retrieve(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Retrieve context based on metadata.
        
        Args:
            metadata: Query metadata.
            
        Returns:
            torch.Tensor: Retrieved context.
            
        Raises:
            KeyError: If context not found.
        """
        pass

    @abstractmethod
    def add_context(self, key: str, tensor: torch.Tensor):
        """Add context to the store."""
        pass


class DictContextStore(RetrievalStrategy):
    """
    In-memory dictionary-based context store.
    """
    def __init__(self):
        self.context_store: Dict[str, torch.Tensor] = {}
        logger.info("Initialized DictContextStore")

    def add_context(self, key: str, tensor: torch.Tensor):
        """
        Add context to the store.
        
        Args:
            key: Unique identifier for the context.
            tensor: The context tensor (e.g., weights).
        """
        if key in self.context_store:
            logger.warning(f"Overwriting context for key: {key}")
        
        self.context_store[key] = tensor
        logger.debug(f"Added context for key: {key}, shape: {tensor.shape}")

    def retrieve(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Retrieve context based on metadata['query_key'].
        """
        key = metadata.get("query_key")
        if not key:
            logger.error("Metadata missing 'query_key'")
            raise ValueError("Metadata must contain 'query_key'")
            
        if key not in self.context_store:
            logger.warning(f"Context not found for key: {key}")
            raise KeyError(f"Context not found for key: {key}")
            
        logger.info(f"Retrieved context for key: {key}")
        return self.context_store[key]
