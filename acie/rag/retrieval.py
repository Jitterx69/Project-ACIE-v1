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


class PGVectorRetriever(RetrievalStrategy):
    """
    Retrieval strategy using PostgreSQL/pgvector.
    """
    def __init__(self, connection_string: str = None):
        from acie.db.vector_store import VectorStore
        self.vector_store = VectorStore(connection_string)
        logger.info("Initialized PGVectorRetriever connected to Postgres.")

    def add_context(self, key: str, tensor: torch.Tensor):
        """
        Add context to the vector store.
        converts tensor to list[float] and stores with key as metadata.
        """
        # Flatten tensor to list[float]
        embedding = tensor.flatten().tolist()
        
        # Store
        self.vector_store.add_documents(
            texts=[f"Context for key: {key}"],
            embeddings=[embedding],
            metadatas=[{"query_key": key}]
        )
        logger.debug(f"Added context to PGVector for key: {key}")

    def retrieve(self, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Retrieve context based on embedding similarity or key lookup.
        If 'query_embedding' is in metadata, does vector search.
        If 'query_key' is in metadata, we might need a direct lookup method in VectorStore (not yet implemented efficiently there, so we warn).
        
        For RAG, we typically use query_embedding.
        """
        query_embedding = metadata.get("query_embedding")
        
        if query_embedding:
            # Vector Search
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.flatten().tolist()
                
            results = self.vector_store.search(query_embedding, limit=1, include_embedding=True)
            if not results:
                raise KeyError("No context found for query_embedding")
            
            # Use the retrieved embedding as the context tensor
            # In a real RAG, we might retrieve text and embed it again, or retrieve a separate weight tensor.
            # Here, we assume the embedding ITSELF is the modulation signal.
            retrieved_embedding = results[0].get("embedding")
            if not retrieved_embedding:
                raise ValueError("Retrieved document missing embedding")
                
            return torch.tensor(retrieved_embedding, dtype=torch.float32)
            
        raise NotImplementedError("Metadata must contain 'query_embedding' for PGVector retrieval.")
