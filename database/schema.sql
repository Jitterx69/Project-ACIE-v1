-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
-- Users Table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
-- Inference Audit Logs (Time-series)
CREATE TABLE IF NOT EXISTS inference_logs (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(36) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    input_hash VARCHAR(64),
    latency_ms FLOAT,
    status VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
-- Vector Embeddings for RAG
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(768) -- Assuming 768 dim for BERT/similar
);
-- Create HNSW Index for fast similarity search
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
-- Seed initial admin user (password: admin)
-- In production, use a migration script or external seeding
INSERT INTO users (username, password_hash, role)
VALUES (
        'admin',
        '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW',
        'admin'
    ) ON CONFLICT (username) DO NOTHING;