import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Generator
from acie.cipher_embeddings.acie_h import ACIEHomomorphicCipher
from acie.cipher_embeddings.tensor import CipherTensor
from acie.rag.config import RAGConfig
from acie.logging.structured_logger import logger

class ImageIngestion:
    """
    Handles loading, preprocessing, and encrypting images for the HE RAG Pipeline.
    Supports both standard loading and streaming tiled processing for large datasets.
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        logger.info(f"Initializing ImageIngestion with key_size={config.key_size}, tile_size={config.tile_size}")
        
        self.cipher = ACIEHomomorphicCipher(key_size=config.key_size)
        
        # Transform for standard pipeline
        self.transform = transforms.Compose([
            transforms.Resize((config.image_height, config.image_width)),
            transforms.Grayscale() if config.grayscale else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
        
        # Transform for tiles (no resize, just tensor conversion)
        self.tile_transform = transforms.Compose([
            transforms.Grayscale() if config.grayscale else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])

    def load_and_encrypt(self, image_path: str) -> CipherTensor:
        """
        Legacy method: Loads generic image, preprocesses it, and encrypts it into a CipherTensor.
        Keep for backward compatibility with small images.
        """
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        logger.info(f"Loading and encrypting image: {image_path}")
        
        try:
            img = Image.open(path)
            tensor_img = self.transform(img)
            flat_img = tensor_img.view(-1)
            return self._encrypt_tensor(flat_img)
        except Exception as e:
            logger.error(f"Legacy processing failed: {e}")
            raise ValueError(f"Processing failed: {e}")

    def process_large_image_stream(self, image_path: str) -> Generator[CipherTensor, None, None]:
        """
        Streams a large image file in tiles, encrypting each tile independently to reduce memory usage.
        
        Args:
            image_path: Path to the large input image.
            
        Yields:
            CipherTensor: Encrypted tile data.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        logger.info(f"Starting tiled stream processing for: {image_path}")
        
        try:
            with Image.open(path) as img:
                width, height = img.size
                tile_size = self.config.tile_size
                
                logger.info(f"Image Size: {width}x{height}, Tile Size: {tile_size}")
                
                for i in range(0, width, tile_size):
                    for j in range(0, height, tile_size):
                        # Calculate actual tile box (handling edges)
                        box = (i, j, min(i + tile_size, width), min(j + tile_size, height))
                        
                        # Lazy load/crop the tile
                        tile = img.crop(box)
                        
                        # Handle padding for edge tiles to ensure consistent tensor size
                        if tile.size != (tile_size, tile_size):
                            # Create a black background of the full tile size
                            new_tile = Image.new(tile.mode, (tile_size, tile_size))
                            new_tile.paste(tile, (0, 0))
                            tile = new_tile
                        
                        # Apply local transforms (convert to tensor)
                        tensor_tile = self.tile_transform(tile)
                        flat_tile = tensor_tile.view(-1)
                        
                        # Verify dimension if rigorous checking is needed, otherwise pad or process as is
                        # For RAG, we might need consistent dimensions, but assuming inputs align with tile config
                        
                       # Save/Stream encrypted tile to disk or DB immediately to free RAM
                # Yield tuple (encrypted_tile, (i, j, w, h))
                yield self._encrypt_tensor(flat_tile), (i, j, width, height)
                        
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise RuntimeError(f"Streaming failed: {e}")

    def _encrypt_tensor(self, flat_tensor: torch.Tensor) -> CipherTensor:
        """Helper to encrypt a flattened tensor."""
        # Scale to 0-255 range (assuming input is 0-1)
        scaled_img = (flat_tensor * 255).long().tolist()
        
        try:
            encrypted_values = [self.cipher.encrypt(x) for x in scaled_img]
            return CipherTensor(encrypted_values, self.cipher)
        except Exception as e:
            logger.error(f"Encryption primitive failed: {e}")
            raise

    def decrypt_result(self, result: CipherTensor) -> torch.Tensor:
        """Helper to decrypt a result using the internal cipher instance."""
        logger.debug("Decrypting result tensor...")
        return result.decrypt()
