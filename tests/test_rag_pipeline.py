import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock pgvector dependencies which might not be installed in test env
sys.modules["pgvector"] = MagicMock()
sys.modules["pgvector.sqlalchemy"] = MagicMock()
sys.modules["sqlalchemy"] = MagicMock()
sys.modules["sqlalchemy.orm"] = MagicMock()
sys.modules["sqlalchemy.dialects.postgresql"] = MagicMock()

import torch
from acie.rag.pipeline import HEImageRAGPipeline
from acie.rag.config import RAGConfig

class TestRAGPipeline(unittest.TestCase):
    @patch('acie.db.vector_store.VectorStore')
    def test_pgvector_retrieval(self, MockVectorStore):
        """Test RAG pipeline with PGVector retriever."""
        # Setup environment to use pgvector
        os.environ["RAG_RETRIEVER_TYPE"] = "pgvector"
        
        # Mock VectorStore instance
        mock_store = MockVectorStore.return_value
        # Mock search return: list of dicts with 'embedding'
        # Embedding should match SecureLinear weight shape: (out_dim, in_dim) or similar?
        # SecureLinear(1048576, 10) weight is (10, 1048576)
        # That's huge. 
        # For test config, let's use small dims.
        
        config = RAGConfig(
            image_width=10, 
            image_height=10, 
            input_dim=100, 
            output_dim=2, 
            key_size=128
        )
        
        # Expected embedding size = 2 * 100 = 200 floats
        expected_embedding = [0.1] * 200
        mock_store.search.return_value = [{
            "id": 1,
            "embedding": expected_embedding,
            "metadata": {"key": "test"}
        }]
        
        # Initialize Pipeline
        pipeline = HEImageRAGPipeline(config)
        
        # Verify Retriever Initialization
        from acie.rag.retrieval import PGVectorRetriever
        self.assertIsInstance(pipeline.retrieval, PGVectorRetriever)
        
        # Run Pipeline
        # We need to mock ingestion load_and_encrypt to avoid reading real file
        pipeline.ingestion.load_and_encrypt = MagicMock()
        # Mock input tensor (CipherTensor mostly acts like tensor in mock if not strict)
        # But SecureGenerationModel expects CipherTensor. 
        # Let's mock the generation model output directly to skip complex crypto in unit test
        pipeline.generation = MagicMock()
        pipeline.generation.return_value = "encrypted_result_mock"
        
        pipeline.ingestion.decrypt_result = MagicMock(return_value=torch.tensor([1.0, 2.0]))
        
        metadata = {"query_embedding": [0.5] * 10} # Dummy query
        result = pipeline.run("dummy.jpg", metadata)
        
        # Assertions
        mock_store.search.assert_called_once()
        pipeline.generation.assert_called_once()
        
        # Verify context injection
        # The generation mock was called. In real run, it gets 'context_weights'.
        # We can check call args
        args, kwargs = pipeline.generation.call_args
        self.assertIn('context_weights', kwargs)
        context_tensor = kwargs['context_weights']
        self.assertEqual(context_tensor.numel(), 200)
        
        print("\nRAG Pipeline with PGVector verified successfully.")

if __name__ == '__main__':
    unittest.main()
