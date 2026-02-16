import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from acie.cipher_embeddings.acie_h import ACIEHomomorphicCipher
from acie.cipher_embeddings.tensor import CipherTensor
from acie.rag.config import RAGConfig
from acie.logging.structured_logger import logger

class ImageIngestion:
    """
    Handles loading, preprocessing, and encrypting images for the HE RAG Pipeline.
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        logger.info(f"Initializing ImageIngestion with key_size={config.key_size}")
        
        self.cipher = ACIEHomomorphicCipher(key_size=config.key_size)
        
        self.transform = transforms.Compose([
            transforms.Resize((config.image_height, config.image_width)),
            transforms.Grayscale() if config.grayscale else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])

    def load_and_encrypt(self, image_path: str) -> CipherTensor:
        """
        Loads an image, preprocesses it, and encrypts it into a CipherTensor.
        
        Args:
            image_path: Path to the input image file.
            
        Returns:
            CipherTensor: The encrypted flattened image vector.
            
        Raises:
            FileNotFoundError: If image file does not exist.
            ValueError: If image processing fails.
        """
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        logger.info(f"Loading and encrypting image: {image_path}")
        
        # 1. Load Image
        try:
            img = Image.open(path)
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            raise ValueError(f"Failed to load image: {e}")

        # 2. Preprocess
        try:
            tensor_img = self.transform(img)
            # Flatten: [C, H, W] -> [N]
            flat_img = tensor_img.view(-1)
            
            if flat_img.numel() != self.config.input_dim:
                logger.warning(
                    f"Dimension mismatch: Expected {self.config.input_dim}, got {flat_img.numel()}"
                )
        except Exception as e:
             logger.error(f"Preprocessing failed: {e}")
             raise ValueError(f"Preprocessing failed: {e}")
        
        # 3. Quantize (Float -> Int)
        # Scale to 0-255 range (assuming input is 0-1)
        scaled_img = (flat_img * 255).long().tolist()
        
        # 4. Encrypt
        logger.debug(f"Encrypting {len(scaled_img)} values...")
        try:
            encrypted_values = [self.cipher.encrypt(x) for x in scaled_img]
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise RuntimeError(f"Encryption failed: {e}")
            
        logger.info("Image successfully encrypted.")
        return CipherTensor(encrypted_values, self.cipher)

    def decrypt_result(self, result: CipherTensor) -> torch.Tensor:
        """Helper to decrypt a result using the internal cipher instance."""
        logger.debug("Decrypting result tensor...")
        return result.decrypt()
