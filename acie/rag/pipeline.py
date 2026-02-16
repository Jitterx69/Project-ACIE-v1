from typing import Dict, Any, Union, Optional
import torch
from .config import RAGConfig
from .ingestion import ImageIngestion
from .retrieval import RetrievalStrategy, DictContextStore
from .generation import SecureGenerationModel
from acie.cipher_embeddings.tensor import CipherTensor
from acie.logging.structured_logger import logger

class HEImageRAGPipeline:
    """
    RAG Pipeline for Encrypted Image Processing using Homomorphic Encryption (ACIE-H).
    
    Flow:
    1. Image -> Ingestion -> Encrypted CipherTensor
    2. Metadata -> Retrieval -> Plaintext Context (Weights/Vectors)
    3. (EncryptedImage, PlaintextContext) -> SecureGeneration -> EncryptedResult
    4. EncryptedResult -> Decryption (Client-side simulation) -> Result
    """
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration object. If None, uses default.
        """
        self.config = config or RAGConfig.default()
        logger.info(f"Initializing HEImageRAGPipeline with config: {self.config}")
        
        self.ingestion = ImageIngestion(self.config)
        
        # We currently default to DictContextStore, but this could be configurable
        self.retrieval = DictContextStore()
        
        self.generation = SecureGenerationModel(self.config)

    def add_context(self, key: str, context_tensor: torch.Tensor):
        """Add context to the retrieval engine."""
        self.retrieval.add_context(key, context_tensor)

    def run(self, image_path: str, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Run the full RAG pipeline.
        
        Args:
            image_path: Path to the input image.
            metadata: Metadata for context retrieval.
            
        Returns:
            torch.Tensor: The decrypted result.
            
        Raises:
            Exception: If any step fails.
        """
        logger.info(f"Starting RAG pipeline run for image: {image_path}")
        try:
            # 1. Ingestion (Encryption)
            # Logs handled inside ingestion
            encrypted_image: CipherTensor = self.ingestion.load_and_encrypt(image_path)
            
            # 2. Retrieval
            # Logs handled inside retrieval
            context_context = self.retrieval.retrieve(metadata)
            
            # 3. Generation (Secure Processing)
            # Logs handled inside generation
            encrypted_result = self.generation(encrypted_image, context_weights=context_context)
            
            # 4. Decryption
            logger.info("Decrypting result...")
            decrypted_result = self.ingestion.decrypt_result(encrypted_result)
            
            logger.info("Pipeline run completed successfully.")
            return decrypted_result
            
        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            raise e
