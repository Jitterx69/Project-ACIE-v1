import torch
import torch.nn as nn
from typing import Optional
from acie.cipher_embeddings.tensor import CipherTensor
from acie.models.secure_layers import SecureLinear
from acie.rag.config import RAGConfig
from acie.logging.structured_logger import logger

class SecureGenerationModel(nn.Module):
    """
    A Secure Model that processes encrypted inputs using retrieved context.
    
    Architecture:
    Input (Encrypted) -> SecureLinear (using retrieved context as weights) -> Output (Encrypted)
    """
    def __init__(self, config: RAGConfig):
        super().__init__()
        self.config = config
        logger.info(
            f"Initializing SecureGenerationModel: {config.input_dim} -> {config.output_dim}"
        )
        self.layer = SecureLinear(config.input_dim, config.output_dim)

    def forward(self, x: CipherTensor, context_weights: Optional[torch.Tensor] = None) -> CipherTensor:
        """
        Forward pass with optional context injection.
        
        Args:
            x: Encrypted input tensor.
            context_weights: Optional context to *replace* the model weights.
            
        Returns:
            CipherTensor: Encrypted output.
        """
        if context_weights is not None:
            # Validate shape
            expected_shape = self.layer.weight.shape
            if context_weights.shape != expected_shape:
                logger.error(
                    f"Context shape mismatch. Expected {expected_shape}, got {context_weights.shape}"
                )
                raise ValueError(
                    f"Context shape mismatch. Expected {expected_shape}, got {context_weights.shape}"
                )

            logger.debug("Injecting context weights into SecureLinear layer.")
            # For this RAG pipeline, we use the context_weights to *modulate*
            # or *augment* the input, or use them as the weights themselves.
            with torch.no_grad():
                self.layer.weight.copy_(context_weights)
        
        logger.debug("Executing secure forward pass.")
        return self.layer(x)
