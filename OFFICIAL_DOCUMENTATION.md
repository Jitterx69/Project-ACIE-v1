# Astrophysical Counterfactual Inference Engine (ACIE)
## Official System Documentation v1.0

---

## 1. Executive Summary

### 1.1 Project Name & Vision
The **Astrophysical Counterfactual Inference Engine (ACIE)** is a pioneering artificial intelligence system designed to revolutionize scientific discovery in astronomy and physics. It bridges the gap between high-dimensional observational data and theoretical physical models by enabling **counterfactual reasoning**: the ability to ask "What if?" questions about physical systems (e.g., "How would this galaxy's spectrum change if its metallicity were doubled?").

### 1.2 Problem Statement
Traditional deep learning models operate as "black boxes," mapping inputs to outputs based on correlation rather than causation. In scientific domains, this is insufficient because:
1.  **Correlations are not Causal**: A model might learn that red galaxies are old, but changing the color doesn't make a galaxy older.
2.  **Physical Violations**: Generative models often produce outputs that violate conservation of energy, mass, or momentum.
3.  **Data Sensitivity**: Proprietary or sensitive observational data cannot be safely processed by third-party AI services.

### 1.3 Objectives
-   **Causal Inference**: Implement Structural Causal Models (SCM) to explicitly model cause-and-effect relationships.
-   **Physical Fidelity**: Enforce fundamental laws of physics (Conservation, Stability) via differentiable constraint layers.
-   **Secure Computation**: Enable fully encrypted inference using Homomorphic Encryption (HE), ensuring zero data leakage.
-   **High-Performance Computing**: Leverage Rust-based acceleration for critical matrix operations and cryptographic primitives.

### 1.4 Success Metrics
-   **Physical Consistency**: The system must achieve a <0.1% violation rate for conservation laws in generated outputs.
-   **Inference Latency**: Sub-100ms response time for standard queries, even with encryption overhead.
-   **Scalability**: The architecture must support distributed deployment via Kubernetes, scaling to handling terabytes of astronomical survey data.

---

## 2. System Architecture

The ACIE system employs a modern, microservices-based architecture designed for modularity, scalability, and security.

### 2.1 High-Level Components
1.  **Secure Gateway (Java)**: The fortress guarding the system. Handles all incoming traffic, authentication, and request routing.
2.  **Inference Engine (Python)**: The "Brain." Orchestrates the RAG pipeline, causal reasoning, and physics validation.
3.  **Accelerator (Rust)**: The "Muscle." Provides raw computational power for sparse matrix algebra and cryptographic operations.
4.  **Analyst (R)**: The "Statistician." Performs rigorous post-hoc analysis, hierarchical modeling, and reporting.
5.  **Dashboard (Frontend)**: The "Face." A React-based interface for scientists to interact with the system.

### 2.2 Data Flow Diagram
1.  **Ingestion**: User uploads observational data (FITS/Images) via the Dashboard.
2.  **Authentication**: Java Gateway validates the request using OAuth2/JWT and logs the access.
3.  **Queuing**: The request is pushed to a Redis-backed Job Queue (managed by Kafka for streaming).
4.  **Processing**: A Python Worker picks up the job.
    -   **Encryption**: Data is encrypted using the Rust-backed Homomorphic Encryption module.
    -   **Retrieval**: The RAG pipeline fetches similar historical contexts from the PostgreSQL Vector Store.
    -   **Inference**: The SCM + PINN generates a counterfactual result.
5.  **Validation**: Physics constraints are checked. If violated, the generation is rejected or corrected.
6.  **Storage**: The final result and its vector embedding are stored in PostgreSQL.
7.  **Analysis**: The R service is triggered to update population-level statistics.

---

## 3. Deep Dive: Core Technologies

### 3.1 Structural Causal Models (SCM)
The SCM (`acie.core.scm`) is not just a graph but a fully executable causal engine.
-   **Topology**: A Directed Acyclic Graph (DAG) representing the physical system. Nodes are variables (Latent $P$, Observable $O$, Noise $N$), and edges represent causal mechanisms.
-   **Interventions**: The engine implements the `do`-operator (`do(X=x)`), mathematically severing the natural causes of a variable $X$ and forcing it to value $x$. This allows for true counterfactual generation, unlike conditional probability $P(Y|X)$.
-   **Mechanisms**: Each relationship is modeled as a learnable function $X_i = f_i(Pa(X_i), U_i)$, typically parameterized by a neural network.

### 3.2 Physics-Informed Neural Networks (PINNs)
ACIE integrates physics directly into the learning process via specialized layers (`acie.models.physics_layers`).
-   **ConservationLayer**: Differentiable layer that computes the violation of conservation laws (Mass, Energy, Momentum). It adds a penalty term to the loss function: $L_{total} = L_{task} + \lambda L_{physics}$.
-   **StabilityLayer**: Checks for dynamical stability conditions, such as the Virial Theorem ($2K + U = 0$) for self-gravitating systems.
-   **ObservationalBoundaryLayer**: Enforces hard constraints on observable quantities (e.g., flux must be positive, redshift must be non-negative).

### 3.3 Homomorphic Encryption (HE)
To enable secure collaboration, ACIE processes data in an encrypted state (`acie.cipher_embeddings`).
-   **Scheme**: A custom HE scheme supporting linear operations (Addition, Multiplication by scalar) on ciphertext.
-   **Implementation**: Critical primitives (`encrypt_batch`, `dot_product`) are implemented in Rust for maximum performance, utilizing AVX-512 SIMD instructions.
-   **Security**: Ensures that the server never sees the raw input data, only the encrypted vector.

### 3.4 Rust Accelerator
The `acie_core` module is a high-performance extension written in Rust.
-   **Sparse Matrix Algebra**: Implements Compressed Sparse Row (CSR) format for memory-efficient storage of large interaction matrices.
-   **Parallelism**: Uses `rayon` for multi-threaded execution of matrix multiplications (SpMM) and element-wise operations.
-   **FFI**: Exposed to Python via `pyo3` and `numpy` C-API, providing a zero-copy interface for maximum throughput.

---

## 4. Module Specifications

### 4.1 Java Gateway (`java/`)
-   **Framework**: Spring Boot 3.2.0.
-   **Security**: `SecurityConfig` implements robust JWT validation and role-based access control (RBAC).
-   **Communication**: gRPC integration (`io.grpc`) for low-latency inter-service communication.
-   **Logging**: Comprehensive `RequestLogging` aspect that persists audit trails to the database via JPA.
-   **Concurrency**: `AsyncService` with `CompletableFuture` for non-blocking request handling.

### 4.2 R Analysis Package (`ACIEr`)
-   **Purpose**: Advanced statistical inference and visualization.
-   **Capabilities**:
    -   `plot_latent_space`: Dimensionality reduction (PCA, t-SNE) to visualize the manifold of physical states.
    -   `fit_hierarchical_model`: Uses `lme4` to fit mixed-effects models, capturing population-level trends and individual variations.
    -   `calculate_spatial_correlation`: Computes the 2-Point Correlation Function (2PCF) for spatial clustering analysis.
-   **Reporting**: Automated generation of RMarkdown analysis reports.

### 4.3 RAG Pipeline & Database
-   **Vector Store**: PostgreSQL with `pgvector`.
    -   *Schema*: Stores 1536-dimensional embeddings for efficient semantic search.
    -   *Retrieval*: `PGVectorRetriever` executes cosine similarity searches to find relevant historical instances (context) for the current inference task.
-   **Ingestion**: Automated pipeline (`ImageIngestion`) to process, embed, and index new astronomical images.

---

## 5. Deployment Guide

### 5.1 Infrastructure Requirements
-   **Containerization**: All services are Dockerized.
    -   `Dockerfile.production` for the Python Inference Engine.
    -   `java/Dockerfile` for the Gateway.
    -   `database/Dockerfile` for the customized Postgres.
-   **Orchestration**: Comprehensive Kubernetes manifests (`k8s/`) provided for production deployment.
    -   StatefulSets for Redis and Postgres to ensure data persistence.
    -   Deployments for stateless services (API, Java, Frontend).
    -   ConfigMaps and Secrets for configuration management.

### 5.2 Redis Operations
Redis is critical for the asynchronous job queue and caching.
-   **Persistence**: Configured with Append Only File (AOF) via `appendfsync everysec` for durability.
-   **Security**: Hardened configuration disabling dangerous commands (`FLUSHALL`, `CONFIG`) and mandating password authentication.
-   **Memory Management**: `maxmemory-policy allkeys-lru` ensures the cache never causes an OOM crash.

### 5.3 Monitoring & Observability
-   **Metrics**: Prometheus scrapes endpoints from Spring Boot Actuator and Python clients.
-   **Visualization**: Grafana dashboards provide real-time views of:
    -   Request throughput and latency.
    -   Physics violation rates.
    -   GPU utilization (via DCGM exporter).
    -   Audit logs and security events.

---

## 6. Future Directions

### 6.1 Quantum-Classical Hybrid
Future versions will integrate Quantum Kernels ($Q(x, x')$) into the SCM to model highly complex, non-classical correlations in quantum physical systems.

### 6.2 Federated Learning
Extending the secure infrastructure to support Federated Learning, allowing multiple research institutions to train a shared ACIE model without sharing raw proprietary datasets.

### 6.3 Automated Causal Discovery
Implementing algorithms (e.g., PC, FCI) to automatically discover the structure of the SCM DAG directly from observational data, rather than relying on expert-defined graphs.

---

**© 2026 ACIE Project Team. All Rights Reserved.**
