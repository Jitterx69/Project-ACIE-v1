import os
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, select
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    metadata_ = Column("metadata", JSONB, default={})
    # 768 dimensions for standard BERT/HuggingFace embeddings
    embedding = Column(Vector(768))

class VectorStore:
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize connection to Postgres with pgvector
        """
        if connection_string is None:
            # Construct from env vars
            user = os.getenv("POSTGRES_USER", "acie")
            password = os.getenv("POSTGRES_PASSWORD", "acie_secure_password")
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            db = os.getenv("POSTGRES_DB", "acie_db")
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db}"
            
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        
    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict]] = None):
        """Store documents and their embeddings"""
        session = self.Session()
        try:
            if metadatas is None:
                metadatas = [{}] * len(texts)
                
            docs = []
            for text, emb, meta in zip(texts, embeddings, metadatas):
                docs.append(Document(content=text, embedding=emb, metadata_=meta))
            
            session.add_all(docs)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Similarity search using Cosine Distance (via <=> operator or l2_distance <->)
        pgvector uses:
        <-> : Euclidean distance (L2)
        <=> : Cosine distance
        <#> : Inner product
        """
        session = self.Session()
        try:
            # Use cosine distance for similarity
            results = session.scalars(
                select(Document)
                .order_by(Document.embedding.cosine_distance(query_embedding))
                .limit(limit)
            ).all()
            
            return [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata_
                }
                for doc in results
            ]
        finally:
            session.close()
