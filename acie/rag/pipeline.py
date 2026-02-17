from typing import Dict, Any, Union, Optional, Generator
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

    def run_stream(self, image_path: str, metadata: Dict[str, Any]) -> Generator[torch.Tensor, None, None]:
        """
        Run the RAG pipeline in streaming mode for large images.
        
        Args:
            image_path: Path to the large input image.
            metadata: Metadata for context retrieval.
            
        Yields:
            torch.Tensor: Decrypted result for each tile.
        """
        logger.info(f"Starting Streaming RAG pipeline run for: {image_path}")
        try:
            # 1. Retrieve Context (Assume same context for all tiles for now)
            context_context = self.retrieval.retrieve(metadata)
            
            # 2. Ingestion Stream
            # Initialize Physics Aggregator
            from .physics_aggregator import GlobalPhysicsAggregator
            self.physics_aggregator = GlobalPhysicsAggregator()
            
            tile_generator = self.ingestion.process_large_image_stream(image_path)
            
            for i, result_data in enumerate(tile_generator):
                # Unpack tuple from generator or handle legacy
                if isinstance(result_data, tuple):
                    encrypted_tile, tile_coords = result_data
                else:
                     encrypted_tile, tile_coords = result_data, (0,0)
                     
                # 3. Secure Generation per tile
                # Note: Generation model usually expects full input or specific structure
                # Here we assume the model can handle independent tiles or we adapt the model
                # For RAG, we might need tiled context as well, but using global context for simplicity
                
                encrypted_result = self.generation(encrypted_tile, context_weights=context_context)
                
                # 4. Decryption per tile
                decrypted_result = self.ingestion.decrypt_result(encrypted_result)
                
                # 5. Physics Update (Global Constraint Enforcement)
                self.physics_aggregator.update(decrypted_result, tile_box=tile_coords)
                
                yield decrypted_result
                
            # 6. Global Validation
            global_violations = self.physics_aggregator.validate_global_constraints()
            if global_violations:
                logger.warning(f"Global Physics Violations detected: {global_violations}")
                
            logger.info("Streaming pipeline run completed successfully.")
            
        except Exception as e:
            logger.error(f"Streaming pipeline failed: {e}")
            raise e
