# ACIE Homomorphic Encryption RAG Pipeline

**Module**: `acie.rag`

This module implements a production-grade Retrieval-Augmented Generation (RAG) pipeline for processing images using Homomorphic Encryption (HE). It allows for secure inference where the input data remains encrypted during processing, augmented by retrieved plaintext context.

## Architecture

The pipeline consists of four main stages:

1.  **Ingestion (`ingestion.py`)**: 
    - Loads input images.
    - Preprocesses them (Resize, Grayscale, Tensor conversion).
    - Encrypts the pixel data using the Paillier cryptosystem (`ACIEHomomorphicCipher`).
    - Produces a `CipherTensor`.

2.  **Retrieval (`retrieval.py`)**:
    - Uses a `RetrievalStrategy` to fetch context based on query metadata.
    - Currently supports `DictContextStore` for in-memory retrieval of model weights or prototype vectors.
    - **Note**: Retrieval is based on *plaintext metadata* as FHE search is not supported.

3.  **Generation (`generation.py`)**:
    - Executes a `SecureGenerationModel`.
    - Uses `SecureLinear` layers to process the encrypted input.
    - Injects retrieved context (weights) dynamically into the secure layer to modulate processing.
    - Output is an encrypted result tensor.

4.  **Decryption**:
    - The final encrypted result is decrypted (simulating client-side decryption) to produce plaintext logits or features.

## Components

### `RAGConfig` (`config.py`)
Central configuration management using dataclasses.
- `key_size`: Size of the Paillier key (e.g., 1024, 2048).
- `input_dim`: Expected input dimension (flattened image size).
- `image_width`, `image_height`: Dimensions for resizing.

### `HEImageRAGPipeline` (`pipeline.py`)
Main orchestration class.
```python
from acie.rag import HEImageRAGPipeline, RAGConfig

config = RAGConfig(key_size=1024)
pipeline = HEImageRAGPipeline(config)
```

## Usage

See `examples/rag_demo.py` for a complete runnable example.

```python
# 1. Initialize
pipeline = HEImageRAGPipeline()

# 2. Add Context
pipeline.add_context("model_v1", weights_tensor)

# 3. Run
result = pipeline.run("image.png", {"query_key": "model_v1"})
```

## Logging
The module uses `acie.logging.structured_logger` to emit detailed JSON-formatted logs for observability.
