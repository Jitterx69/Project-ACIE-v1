from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class RAGConfig:
    """
    Configuration for the Homomorphic Encryption RAG Pipeline.
    """
    # Encryption
    key_size: int = 1024

    # Image Processing
    image_width: int = 1024
    image_height: int = 1024
    grayscale: bool = True
    
    # Large Image Support (Tiling)
    tile_size: int = 1024
    
    # Model Architecture
    input_dim: int = 1048576  # 1024*1024
    output_dim: int = 10
    
    # System
    log_level: str = "INFO"
    log_file: Optional[str] = "rag_pipeline.log"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.image_width * self.image_height != self.input_dim:
            raise ValueError(
                f"Input dimension ({self.input_dim}) must match image dimensions ({self.image_width}x{self.image_height})"
            )
        if self.key_size < 128:
            raise ValueError("Key size must be at least 128 bits.")

    @classmethod
    def default(cls) -> 'RAGConfig':
        """Return default configuration."""
        return cls()
