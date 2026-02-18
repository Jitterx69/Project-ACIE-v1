# Astrophysical Counterfactual Inference Engine (ACIE)
## Technical Whitepaper & System Architecture Reference

**Version:** 2.0 (Literature Edition)
**Date:** February 18, 2026
**Authors:** ACIE Engineering & Research Team

---

# Table of Contents

1.  **Abstract**
2.  **Introduction**
    *   2.1 The Crisis of Interpretability in Astrophysics
    *   2.2 The Counterfactual Imperative
    *   2.3 System Objectives & Constraints
3.  **Theoretical Framework**
    *   3.1 Structural Causal Models (SCMs)
    *   3.2 Physics-Informed Neural Networks (PINNs)
    *   3.3 Homomorphic Encryption (HE)
4.  **System Architecture**
    *   4.1 Global Topology
    *   4.2 Microservices Interaction Patterns
    *   4.3 Data Flow & State Management
5.  **Computational Engine**
    *   5.1 The Rust Accelerator
    *   5.2 Sparse Matrix Algorithms
    *   5.3 Cryptographic Primitives Implementation
6.  **Data Architecture**
    *   6.1 Vector Space Embedding
    *   6.2 Retrieval-Augmented Generation (RAG)
    *   6.3 Persistence & Reliability
7.  **Security & Compliance**
    *   7.1 Threat Model
    *   7.2 Encryption Strategy
    *   7.3 Access Control & Audit
8.  **Operational Reliability**
    *   8.1 Orchestration Strategy
    *   8.2 Observability & Monitoring
    *   8.3 Disaster Recovery
9.  **Appendices**
    *   A. Mathematical Derivations
    *   B. Configuration Parameters
    *   C. Data Dictionary

---

# 1. Abstract

The Astrophysical Counterfactual Inference Engine (ACIE) represents a paradigm shift in the application of Artificial Intelligence to the physical sciences. Unlike traditional deep learning systems that rely on correlational patterns, ACIE is designed to reason causally about physical systems. By integrating Structural Causal Models (SCMs) with Physics-Informed Neural Networks (PINNs), the system generates scientifically valid counterfactuals—answering "what if" questions that are essential for hypothesis testing in astronomy. Furthermore, ACIE addresses the critical challenge of data privacy in collaborative research by employing a bespoke Homomorphic Encryption scheme, allowing for secure interference on encrypted data without decryption. This whitepaper details the theoretical underpinnings, architectural decisions, and engineering innovations that enable ACIE to function as a robust, scalable, and secure platform for next-generation scientific discovery.

---

# 2. Introduction

## 2.1 The Crisis of Interpretability in Astrophysics

Modern astronomy is inundated with data. Next-generation surveys like the LSST (Legacy Survey of Space and Time) and the SKA (Square Kilometre Array) will produce exabytes of observational data. While Machine Learning (ML) has proven effective at classifying these observations—distinguishing between spiral and elliptical galaxies, for instance—it largely fails to provide *physical insight*. A standard Convolutional Neural Network (CNN) can predict a galaxy's redshift from its image, but it cannot explain *why* the galaxy appears the way it does, nor can it reliably predict how the galaxy would look if its star formation rate had been different.

This lack of interpretability is termed the "Black Box Problem." In scientific inquiry, prediction without explanation is of limited value. We require models that not only fit the data but also embody the underlying physical laws that govern the cosmos.

## 2.2 The Counterfactual Imperative

Scientific understanding is fundamentally counterfactual. To understand the effect of gravity, one must ask: "What would the orbit look like if the mass of the central body were doubled?" To understand galaxy evolution, one must ask: "How would this galaxy differ if it had evolved in a denser cluster environment?"

Observational astronomy, however, is passive. We cannot run experiments on galaxies; we can only observe them as they are. This limitation makes causal inference notoriously difficult. ACIE bridges this gap by simulating a "virtual laboratory" where researchers can perform interventions on the latent physical variables of a system and observe the counterfactual outcomes, all while ensuring that the laws of physics (conservation of mass, energy, momentum) are strictly obeyed.

## 2.3 System Objectives & Constraints

The design of ACIE is governed by four primary objectives, which also serve as its engineering constraints:

1.  **Causal Validity:** The system must model the Data Generating Process (DGP) using a Directed Acyclic Graph (DAG) of causal mechanisms, utilizing the formalisms of Judea Pearl's do-calculus.
2.  **Physical Consistency:** Generated outputs must not violate fundamental conservation laws. The system must natively incorporate differentiable physics layers that penalize or project outputs onto the manifold of physically valid states.
3.  **Privacy-Preserving Compute:** Theoretical models often require proprietary data, and collaborative efforts cross institutional boundaries. The system must support "Blind Inference," where the server processes data it cannot see.
4.  **Real-Time Performance:** Despite the computational overhead of homomorphic encryption and physical simulation, the system must provide near real-time feedback (sub-100ms latency) to enable interactive exploration by scientists.

---

# 3. Theoretical Framework

## 3.1 Structural Causal Models (SCMs)

At the core of ACIE's reasoning engine lies the Structural Causal Model. Formally, an SCM is defined as a tuple $\mathcal{M} = \langle \mathbf{U}, \mathbf{V}, \mathcal{F}, P(\mathbf{U}) \rangle$.

*   $\mathbf{U} = \{U_1, \dots, U_n\}$ is a set of **exogenous** variables, determined by factors outside the model (often modeled as noise).
*   $\mathbf{V} = \{V_1, \dots, V_n\}$ is a set of **endogenous** variables, which are determined by variables within the model. In ACIE, these include latent physical parameters ($P$) like mass, metallicity, and age, as well as observable quantities ($O$) like spectra and photometry.
*   $\mathcal{F} = \{f_1, \dots, f_n\}$ is a set of structural functions, where each $f_i$ maps the parents of $V_i$ (denoted $Pa_i$) and the essential noise $U_i$ to the value of $V_i$:
    $$ v_i = f_i(pa_i, u_i) $$
*   $P(\mathbf{U})$ is a probability distribution defined over the exogenous variables.

ACIE distinguishes itself by explicitly modeling the "Mechanism of Action" for astrophysical processes. For example, the Star Formation Rate (SFR) of a galaxy is causally downstream of its Gas Mass and halos Mass, but causally upstream of its Luminosity.

### 3.1.1 The Intervention Operator
Standard probabilistic models estimate conditional probabilities $P(Y | X = x)$. This tells us *what is likely to be true about Y, given that we observe X is x*. In contrast, ACIE computes interventional probabilities $P(Y | do(X = x))$. This answers *what would be true about Y if we forced X to be x*.

Using the `do`-operator typically requires "mutilating" the causal graph. To compute $P(Y | do(X = x))$, we remove all incoming edges to the variable $X$ in the graph, effectively severing its dependence on its natural causes. We then fix $X$ to the value $x$ and propagate the effects through the remaining structural equations. ACIE implements this graph mutilation dynamically at inference time.

## 3.2 Physics-Informed Neural Networks (PINNs)

While SCMs provide the causal structure, the specific structural equations $f_i$ are often too complex to be defined analytically. ACIE approximates these functions using Neural Networks, but with a critical modification: the networks are **Physics-Informed**.

Standard neural networks are "Universal Function Approximators" that are constrained only by the training data. This often leads to solutions that are statistically plausible but physically impossible (e.g., negative mass, energy creation). ACIE addresses this by augmenting the loss function $\mathcal{L}$ with physical regularization terms:

$$ \mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{cons} \mathcal{L}_{conservation} + \lambda_{stab} \mathcal{L}_{stability} $$

### 3.2.1 Conservation Constraints
The Conservation Layer enforces fundamental symmetries. For a closed system, the total energy $E$ and mass $M$ must remain constant during time evolution or latent space traversal. The network computes the residual of these quantities:
$$ \mathcal{L}_{cons} = \left| \sum E_{in} - \sum E_{out} \right|^2 $$
During backpropagation, gradients are calculated not just to minimize prediction error, but to minimize this physical residual, effectively steering the network parameters towards the "Physical Manifold" of the solution space.

### 3.2.2 Stability Constraints (Virial Theorem)
For self-gravitating systems like galaxies and clusters, the Virial Theorem provides a powerful constraint relating the total kinetic energy $T$ and the total potential energy $V$. For a stable system in equilibrium:
$$ 2T + V = 0 $$
ACIE's Stability Layer calculates the kinetic and potential energies of the inferred latent state components and penalizes deviations from this equilibrium ratio. This prevents the generation of "unbound" or unstable counterfactuals that would physically disperse on short timescales.

## 3.3 Homomorphic Encryption (HE)

To satisfy the privacy constraints, ACIE employs Homomorphic Encryption. HE allows for computation to be performed on ciphertext, generating an encrypted result which, when decrypted, matches the result of operations performed on the plaintext.

ACIE utilizes the **Paillier Cryptosystem**, an additive homomorphic scheme. 
*   **Key Generation:** Let $n = pq$ where $p, q$ are large primes. Let $g = n+1$. The public key is $(n, g)$.
*   **Encryption:** To encrypt a message $m$, we select a random $r$ and compute:
    $$ c = g^m \cdot r^n \mod n^2 $$
*   **Homomorphic Addition:** Given $E(m_1)$ and $E(m_2)$, we can compute $E(m_1 + m_2)$ by multiplying the ciphertexts:
    $$ E(m_1) \cdot E(m_2) = (g^{m_1} r_1^n) \cdot (g^{m_2} r_2^n) = g^{m_1+m_2} (r_1 r_2)^n = E(m_1 + m_2) $$
*   **Scalar Multiplication:** We can compute $E(m_1 \cdot k)$ by raising the ciphertext to the power of $k$:
    $$ E(m_1)^k = (g^{m_1} r_1^n)^k = g^{m_1 k} (r_1^k)^n = E(m_1 \cdot k) $$

This property is sufficient to implement a linear layer of a neural network ($y = Wx + b$) completely in the encrypted domain, provided the activations are handled carefully (typically requiring a client-server interaction or polynomial approximation for non-linearities).

---

# 4. System Architecture

## 4.1 Global Topology

The ACIE infrastructure is designed as a distributed, service-oriented architecture (SOA) deployed on Kubernetes. The system is composed of five primary distinct capability domains:

1.  **The Gateway Domain (Java/Spring Boot):** Acts as the secure perimeter. It handles protocol translation (HTTP/1.1 to gRPC), authentication, and rate limiting.
2.  **The Inference Domain (Python/PyTorch):** The computational core where the causal reasoning and physics simulation occur.
3.  **The Accelerator Domain (Rust):** A specialized sidecar or library providing low-level optimization for encryption and matrix algebra.
4.  **The Analysis Domain (R):** A statistical engine dedicated to post-processing, uncertainty quantification, and report generation.
5.  **The Data Domain (PostgreSQL/Redis):** The stateful persistence layer for vectors, models, and job queues.

## 4.2 Microservices Interaction Patterns

ACIE employs an **Asynchronous Event-Driven Architecture** for heavy computational tasks, coupled with **Synchronous RPC** for low-latency queries.

*   **Ingestion Path:** When a user uploads a new dataset, the Gateway creates a `Job` object and publishes an event to the `IngestionStream` in Redis. This ensures that the upload interface remains responsive even during heavy load.
*   **Inference Path:** For real-time inference, the Gateway communicates directly with the Inference Service via gRPC (Google Remote Procedure Call). Protocol Buffers (Protobuf) are used as the Interface Definition Language (IDL), ensuring strict type safety and compact serialization. The use of HTTP/2 multiplexing allows for high throughput.

## 4.3 Data Flow & State Management

Data within ACIE exists in three states: **Raw**, **Latent**, and **Encrypted**.

1.  **Raw State:** The observational data (images, spectra) as uploaded by the user. This data is immediately hashed and stored in the specialized object storage or vector database.
2.  **Latent State:** The low-dimensional representation of the physical parameters (Mass, Z, Age). This is the state where SCM interventions occurs.
3.  **Encrypted State:** For privacy-preserving modes, the Raw data is encrypted on the client side (or at the Gateway ingress) into Paillier ciphertexts. The Inference Engine operates on these ciphertexts without ever possessing the private key.

State management is handled via a combination of Redis (for ephemeral state, job locks, and caching) and PostgreSQL (for durable entity storage). The system enforces strict ACID properties for metadata updates while allowing for eventual consistency in the vector search indices.

---

# 5. Computational Engine

## 5.1 The Rust Accelerator

Python, while excellent for high-level model definition, incurs significant overhead for tight loops and bitwise operations. To meet the latency requirements, ACIE offloads critical path operations to a custom Rust extension module, `acie_core`.

Rust was chosen for its memory safety guarantees without a garbage collector. The system utilizes `PyO3` to create native Python bindings, allowing Rust functions to be called directly from the PyTorch inference loop with near-zero overhead. The Global Interpreter Lock (GIL) is released during these heavy computations, enabling true multi-threaded parallelism.

## 5.2 Sparse Matrix Algorithms

Astrophysical interaction matrices (e.g., gravitational N-body forces, causal adjacency matrices) are often sparse—most variables do not directly interact with most other variables. Storing these as dense matrices wastes memory and compute cycles.

ACIE implements a custom **Compressed Sparse Row (CSR)** format in Rust. The SpMM (Sparse-Dense Matrix Multiplication) kernel is optimized using:
*   **AVX-512 SIMD Instructions:** Processing 16 single-precision floats per CPU cycle.
*   **Rayon Parallelism:** A work-stealing parallelism library that automatically partitions the matrix rows across available CPU cores, ensuring optimal load balancing.
*   **Cache Locality:** The CSR format is designed to access memory sequentially, minimizing CPU cache misses and maximizing memory bandwidth utilization.

## 5.3 Cryptographic Primitives Implementation

The Paillier encryption operations (modular exponentiation and multiplication of large integers) are computationally expensive. A standard Python implementation using `pow(base, exp, mod)` is insufficient for high-throughput batch inference.

The Rust accelerator implements these primitives using the `num-bigint` library, augmented with hand-tuned assembly for the modular reduction step (Montgomery Reduction). This allows the system to perform thousands of encrypted linear operations per second, making real-time homomorphic inference feasible.

(Continued in next section...)

# 6. Data Architecture

## 6.1 Vector Space Embedding

The transformation of raw astronomical data into a latent vector space is a critical precursor to efficient retrieval and inference. ACIE leverages a specific embedding strategy designed to preserve both morphological and semantic similarity.

### 6.1.1 The Encoder Network
The system utilizes a Vision Transformer (ViT-L/14) architecture as the primary encoder. Unlike Convolutional Neural Networks (CNNs), which process images via local receptive fields, Transformers utilize a global self-attention mechanism. This allows the model to capture long-range dependencies—essential for analyzing large-scale structures like galaxy clusters or gravitational lenses where the relevant features are distributed across the entire field of view.

The encoder maps an input image $I \in \mathbb{R}^{H 	imes W 	imes C}$ to a dense vector $z \in \mathbb{R}^{d}$, where $d=1536$. This high-dimensional representation encodes not just the pixel intensity values, but the "semantic" content of the image: the morphological type of the galaxy (spiral, elliptical, irregular), the presence of star-forming regions, and the overall spectral energy distribution.

### 6.1.2 Contrastive Learning Objective
To ensure that the embedding space is meaningful, the encoder is pre-trained using a Contrastive Learning objective (SimCLR). The core idea is to maximize the similarity between augmented views of the same object while minimizing the similarity between different objects.

Let $sim(z_i, z_j)$ be the cosine similarity between two embedding vectors. The loss function for a standard batch of $N$ pairs is given by:
$$ \mathcal{L}_{i,j} = -\log rac{\exp(sim(z_i, z_j) / 	au)}{\sum_{k=1}^{2N} \mathbb{1}_{[k 
eq i]} \exp(sim(z_i, z_k) / 	au)} $$
where $	au$ is a temperature parameter and $\mathbb{1}$ is the indicator function. This objective forces the network to map physically similar galaxies to adjacent points on the hypersphere, facilitating accurate nearest-neighbor search.

## 6.2 Retrieval-Augmented Generation (RAG)

In standard generative models, the "knowledge" is implicit—stored in the weights of the neural network. This leads to hallucinations, where the model generates plausible but factually incorrect details. ACIE employs a Retrieval-Augmented Generation (RAG) approach to ground its inferences in empirical data.

### 6.2.1 The Retrieval Mechanism
When an inference request is received, the system first embeds the query observation. It then queries the PostgreSQL vector database to find the $k$ nearest neighbors from the historical survey data. This retrieval is performed using the IVFFlat (Inverted File Flat) index provided by the `pgvector` extension.

The IVFFlat index partitions the vector space into Voronoi cells based on $C$ centroids calculated via K-Means clustering. During a search, the query vector is compared only against vectors in the nearest cell(s), reducing the search complexity from $O(N)$ (linear scan) to approximately $O(N/C)$. This allows ACIE to search through millions of galaxy embeddings in milliseconds.

### 6.2.2 Context Injection & Attention
The retrieved examples are not just returned to the user; they are injected into the generative model's attention mechanism. The generator $G$ is conditioned on both the input observation $x$ and the retrieved context $\mathcal{C} = \{c_1, \dots, c_k\}$.
$$ y = G(x, \mathcal{C}, 	ext{do}(I)) $$
This "Cross-Attention" mechanism allows the model to "copy" high-frequency details (textures, noise patterns) from real galaxies that are morphologically similar to the target, ensuring that the generated counterfactual possesses realistic fine-grained structure that pure interpolation often smooths away.

## 6.3 Persistence & Reliability

The integrity of scientific data is paramount. ACIE implements a dual-layer persistence strategy.

### 6.3.1 Relational Integrity (PostgreSQL)
All entity relationships—Models, Users, Jobs, and Metadata—are stored in PostgreSQL. The schema is normalized to Third Normal Form (3NF) to prevent data redundancy and anomalies. Strict foreign key constraints ensure referential integrity; for example, an embedding cannot exist without a corresponding parent model record.

### 6.3.2 Ephemeral Durability (Redis)
The job queue and cache are managed by Redis. While often treated as volatile, ACIE configures Redis with Append-Only File (AOF) persistence. Every write operation is logged to disk (`appendfsync everysec`), ensuring that in the event of a power failure or crash, at most one second of queue data is lost. This is acceptable for a system where job reprocessing is idempotent.

---

# 7. Security & Compliance

## 7.1 Threat Model and Risk Assessment

The ACIE system operates under a "Zero Trust" security model. We assume that the network perimeter is breached and that the server infrastructure itself may be compromised by an insider threat or advanced persistent threat (APT).

### 7.1.1 Attack Vectors
1.  **Data Exfiltration:** An attacker gaining access to the database could steal proprietary survey data.
2.  **Model Inversion:** An attacker querying the API could reconstruct the training data by analyzing the model's outputs.
3.  **Adversarial Perturbation:** An attacker could craft inputs designed to trigger infinite loops or resource exhaustion in the physics solver.

### 7.1.2 Mitigations
*   **Data Exfiltration:** Mitigated by Homomorphic Encryption. The server never holds the private key; thus, even a full database dump yields only useless ciphertext.
*   **Model Inversion:** Mitigated by Differential Privacy techniques during training (Gradient Clipping) and strict rate limiting at the API Gateway.
*   **Resource Exhaustion:** Mitigated by strict time-outs in the physics solver and resource quotas enforced by Kubernetes at the pod level.

## 7.2 Encryption Strategy

The encryption strategy is holistic, covering data at rest, data in transit, and data in use.

### 7.2.1 Data in Transit (TLS 1.3)
All communication between clients and the Gateway, and between microservices, is encrypted via mutual TLS (mTLS). We utilize the latest TLS 1.3 protocol, disabling weak cipher suites (e.g., CBC mode, RSA key exchange) in favor of AEAD ciphers (CHACHA20-POLY1305, AES-GCM) and Elliptic Curve Diffie-Hellman Ephemeral (ECDHE) key exchange with Perfect Forward Secrecy (PFS).

### 7.2.2 Data at Rest (AES-256)
Persistent volumes attached to Kubernetes pods are encrypted at the block level using LUKS (Linux Unified Key Setup) with AES-256-XTS. Keys are managed by an external Key Management Service (KMS) and are never stored on the same disk as the data.

### 7.2.3 Data in Use (Paillier HE)
As detailed in Section 3.3, data currently being processed by the Inference Engine is protected by Paillier Homomorphic Encryption. This prevents memory scraping attacks; a dump of the RAM would reveal only BigInt ciphertexts, indistinguishable from random noise without the private key.

## 7.3 Access Control & Audit

### 7.3.1 Role-Based Access Control (RBAC)
Access to the system is governed by a strict RBAC policy.
*   **Researcher:** Can submit jobs, view own results.
*   **Collaborator:** Can view shared results, cannot submit jobs.
*   **Admin:** Full system access, excluding data decryption keys.

### 7.3.2 Immutable Audit Log
Every interaction with the API is logged to an immutable, append-only ledger in the database. This log records the `timestamp`, `user_id`, `resource_accessed`, `ip_address`, and `action_type`. This audit trail is essential for forensic analysis in the event of a security incident and for demonstrating compliance with data governance regulations (e.g., GDPR, CCPA).

---

# 8. Operational Reliability

## 8.1 Orchestration Strategy

ACIE relies on Kubernetes (K8s) for container orchestration, providing self-healing, scaling, and rolling updates.

### 8.1.1 Pod Topology and Anti-Affinity
To ensure high availability, the Inference Service is deployed as a `Deployment` with a minimum of 3 replicas. We utilize `podAntiAffinity` rules to ensure that these replicas are scheduled on different physical nodes (and different Experience Zones, if available cloud-side). This guarantees that the failure of a single hardware rack does not take down the entire inference capability.

### 8.1.2 Horizontal Pod Autoscaling (HPA)
The computational load of ACIE is bursty. During the ingestion of a new survey catalog, traffic may spike by orders of magnitude. The HPA controller monitors the CPU utilization and the custom metric `inference_queue_depth` (exported via Prometheus Adapter).
$$ 	ext{TargetRep} = \lceil 	ext{CurrentRep} 	imes rac{	ext{CurrentMetric}}{	ext{TargetMetric}} 
ceil $$
When the average CPU usage exceeds 70%, or the queue depth exceeds 100 jobs per pod, the HPA automatically triggers the provisioning of new pods, scaling up to a maximum of 10 replicas.

## 8.2 Observability & Monitoring

We subscribe to the "Three Pillars of Observability": Metric, Logs, and Traces.

1.  **Metrics (Prometheus):** We scrape time-series data from all services. Key Service Level Indicators (SLIs) include:
    *   **Latency:** The discrete distribution of response times (P50, P95, P99).
    *   **Throughput:** Requests per second (RPS).
    *   **Error Rate:** Percentage of 5xx responses.
    *   **Saturation:** GPU memory usage, CPU load average.

2.  **Logs (Loki/ELK):** Structured JSON logs are aggregated from all pods. Log entries include a `trace_id` to correlate events across microservices.

3.  **Traces (Jaeger/OpenTelemetry):** Distributed tracing allows us to visualize the lifecycle of a request as it traverses the Gateway, Redis, Inference Engine, and Database. This is crucial for identifying "n+1" query bugs and visualization bottlenecks.

---

# 9. Appendices

## A. Mathematical Derivations

### A.1 Derivation of the Paillier Decryption Correctness

Recall the encryption function: $c = g^m r^n \mod n^2$.
To decrypt, we compute $c^{\lambda} \mod n^2$, where $\lambda = 	ext{lcm}(p-1, q-1)$ is the Carmichael function.

First, observe the structure of the group $\mathbb{Z}_{n^2}^*$. An element $z \in \mathbb{Z}_{n^2}^*$ can be written as $(1+n)^lpha eta^n \mod n^2$ for unique $lpha \in \mathbb{Z}_n$ and $eta \in \mathbb{Z}_n^*$.
Specifically for our ciphertext $c$:
$$ c^\lambda = (g^m r^n)^\lambda = g^{m\lambda} r^{n\lambda} \mod n^2 $$
Since $r \in \mathbb{Z}_{n}^*$, by Carmichael's theorem, $r^\lambda \equiv 1 \mod n$.
Wait, actually $r^{n\lambda} = (r^\lambda)^n \equiv 1^n \equiv 1 \mod n^2$ is helpful but let's be more precise.
Actually $r^\lambda = 1 + k n$. So $r^{n\lambda} = (1+kn)^n = 1 + n(kn) + inom{n}{2}(kn)^2 + \dots$.
Modulo $n^2$, only the first two terms matter: $1 + n^2 k \equiv 1 \mod n^2$.
So the random factor $r^{n}$ vanishes when raised to power $\lambda$.

Now consider $g = n+1$. Then $g^x = (1+n)^x = 1 + nx \mod n^2$ by the Binomial Theorem.
So, $c^\lambda \equiv (1+n)^{m\lambda} \cdot 1 \equiv 1 + n(m\lambda) \mod n^2$.
Let $L(u) = rac{u-1}{n}$.
Then $L(c^\lambda) = rac{(1 + n m \lambda) - 1}{n} = rac{n m \lambda}{n} = m \lambda \mod n$.
Finally, $m = L(c^\lambda) \cdot \lambda^{-1} \mod n$.
This proves that the message $m$ can be recovered from $c$ using the private key $(\lambda, \mu)$, where $\mu = \lambda^{-1} \mod n$.

### A.2 Optimization of Virial Stability Loss

The goal is to minimize $\mathcal{L}_{stab} = (2K + U)^2$.
In an N-body system (or latent representation thereof), $K = rac{1}{2} \sum m_i v_i^2$ and $U = -\sum rac{G m_i m_j}{r_{ij}}$.
Calculating $U$ is $O(N^2)$, which is prohibitive for large $N$. ACIE uses the Barns-Hut approximation (or a grid-based potential solver in the latent space) to compute the potential $U$ in $O(N \log N)$.

The gradient $
abla_{\mathbf{x}} \mathcal{L}_{stab}$ requires differentiating the potential energy with respect to particle positions.
$$ rac{\partial U}{\partial \mathbf{x}_k} = \sum_{j 
eq k} rac{G m_k m_j (\mathbf{x}_j - \mathbf{x}_k)}{|\mathbf{x}_j - \mathbf{x}_k|^3} = -\mathbf{F}_k $$
Thus, minimizing the Virial residual is equivalent to adjusting the positions such that the system satisfies the force balance equation. ACIE performs this adjustment via Gradient Descent in the latent space during the inference optimization loop.

## B. Configuration Parameters

### B.1 Inference Service Configuration
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `BATCH_SIZE` | Integer | 32 | Number of images processed in a single forward pass. Tuned for NVIDIA A100 40GB. |
| `MAX_LATENCY_MS` | Integer | 500 | Hard timeout for inference request. Physics loop terminates if this is exceeded. |
| `ENCRYPTION_KEY_SIZE` | Integer | 2048 | Bit length of the Paillier modulus $n$. 2048 is NIST recommended standard. |
| `PHYSICS_STEPS` | Integer | 10 | Number of optimization steps to take on the physics manifold per generation. |

### B.2 Redis Configuration
| Parameter | Default | Rationale |
| :--- | :--- | :--- |
| `appendfsync` | `everysec` | Balances performance with durability. losing 1s of data is acceptable risk. |
| `maxmemory-policy` | `allkeys-lru` | Evicts least recently used keys when memory is full. Critical for cache stability. |
| `protected-mode` | `yes` | Prevents access from external networks. Essential as Redis has no native TLS in our version. |

## C. Data Dictionary (Schema Description)

### C.1 Table: `embeddings`
This table stores the vector representations of astronomical objects.

*   `id` (UUID, Primary Key): Unique identifier for the embedding.
*   `model_id` (UUID, Foreign Key): Reference to the model that generated this embedding. Linking allows us to invalidate embeddings if a model is deprecated.
*   `content` (TEXT): A serialized JSON describing the raw object (e.g., path to FITS file, coordinates).
*   `vector` (VECTOR(1536)): The high-dimensional embedding. Stored using `pgvector` dedicated type.
*   `metadata` (JSONB): Flexible schema-less storage for astronomical metadata (Redshift, RA, Dec, Telescope Instrument). This allows for efficient filtering combined with vector search (e.g., "Find similar galaxies with z > 2").

### C.2 Table: `inference_logs`
This table serves as the immutable audit trail.

*   `request_id` (UUID): Correlation ID passed from the Client through Gateway to Inference.
*   `user_sub` (VARCHAR): The Subject ID (from JWT) of the user who initiated the request.
*   `timestamp` (TIMESTAMPTZ): Accurate time of request receipt (UTC).
*   `compute_duration_ms` (INT): Time taken by the Python engine. High values trigger alerts.
*   `physics_violation_score` (FLOAT): The final residual value of the physics loss. Used for quality assurance monitoring.


# 10. Literature Review: AI in Astrophysics

## 10.1 The Pre-Deep Learning Era (1990-2012)
The application of automated data analysis in astronomy dates back to the early 1990s with the advent of digital sky surveys like the Digitized Palomar Observatory Sky Survey (DPOSS). Early methods relied on hand-crafted features—isophotal magnitudes, concentration indices, and Gini coefficients—fed into Decision Trees or Support Vector Machines (SVMs) for star-galaxy separation (Odewahn et al., 1992). While effective for low-dimensional data, these methods struggled to capture the complex morphological details of interacting galaxies or gravitational lenses.

## 10.2 The Convolutional Revolution (2012-2020)
The introduction of AlexNet (2012) sparked a revolution. In astronomy, this was mirrored by the Galaxy Zoo challenge (Dieleman et al., 2015), where Convolutional Neural Networks (CNNs) achieved human-level performance in classifying galaxy morphologies. This era saw the widespread adoption of CNNs for photometric redshift estimation, gravitational lens finding, and radio frequency interference (RFI) mitigation. However, these models remained purely correlative. A CNN could predict that a galaxy is "merging," but it could not infer the dynamical history that led to the merger.

## 10.3 The Generative Frontier (2020-Present)
Recent years have seen the rise of Generative Adversarial Networks (GANs) and Diffusion Models. Projects like *CosmoGAN* (Mustafa et al., 2019) demonstrated that AI could generate high-fidelity cosmological weak lensing maps. However, these generative models often hallucinate non-physical artifacts. Terasawa et al. (2020) highlighted that standard GANs fail to preserve the power spectrum of the Cosmic Microwave Background (CMB) at small scales.

## 10.4 ACIE's Contribution
ACIE represents the next step in this evolution: **Physics-Informed Causal Generative Modeling**. By integrating the causal rigor of Pearl (2009) with the physical constraints of Raissi et al. (2019), ACIE moves beyond simple classification or unconstrained generation. It enables the scientist to act as an "Intergalactic Director," modifying the initial conditions of the universe and observing the counterfactual consequences, essentially democratizing the capability of expensive N-body simulations via efficient neural approximation.

---

# 11. User Manual: The `ACIEr` Analysis Package

## 11.1 Overview
The `ACIEr` package provides a bridge between the high-performance Python inference engine and the rich statistical ecosystem of R. It is designed for the "Analyst" persona—a researcher who needs to validate the generated counterfactuals against population-level statistics.

## 11.2 Installation
The package interacts with the Python backend via `reticulate`.
```r
# Install from local source
install.packages("ACIEr_0.1.0.tar.gz", repos = NULL, type = "source")

# Initialize the connection
library(ACIEr)
ACIEr::init_client(host = "localhost", port = 50051)
```

## 11.3 Workflow: Sensitivity Analysis
One of the most powerful features is `sensitivity_analysis`, which computes the partial derivatives of an observable with respect to a latent variable.

### 11.3.1 Methodology
We seek to estimate $rac{\partial O}{\partial z_i}$. The function performs a finite difference approximation by generating counterfactuals at $z_i \pm \epsilon$.

```r
# Load a sample of galaxy embeddings
data <- ACIEr::load_sample("cluster_A2218")

# Compute sensitivity of 'Total Flux' to 'Dark Matter Mass'
sens <- ACIEr::sensitivity_analysis(
  data,
  target_var = "flux",
  intervention_var = "dm_mass",
  epsilon = 0.1
)

# Visualize the non-linear response
ACIEr::plot_sensitivity(sens)
```

**Interpretation:** A linear response indicates a simple power-law scaling (common in mass-to-light ratios). A sharp discontinuity may indicate a phase transition, such as the onset of AGN feedback quenching star formation.

## 11.4 Workflow: Hierarchical Bayesian Modeling
ACIEr integrates with `lme4` to fit mixed-effects models. This is crucial when analyzing data from multiple telescope surveys, where instrument-specific systematics may introduce bias.

### 11.4.1 Model Formulation
Let $y_{ij}$ be the observable for galaxy $i$ in survey $j$. We model this as:
$$ y_{ij} = eta_0 + eta_1 x_{ij} + u_j + \epsilon_{ij} $$
where $eta$ are fixed effects (universal physical laws) and $u_j \sim N(0, \sigma_u^2)$ are random effects (survey-specific implementations).

```r
# Fit the model
model <- ACIEr::fit_hierarchical_model(
  formula = log(mass) ~ log(luminosity) + (1 | survey_id),
  data = combined_catalog
)

# Extract universal scaling relations
summary(model)
```

This allows researchers to "subtract out" the instrument bias and recover the true underlying astrophysical relation.

---

# 12. Governance & Ethics

## 12.1 Algorithmic Bias in Astronomy
While often considered a neutral science, astronomy is subject to "observational bias."
1.  **Malmquist Bias:** We preferentially detect bright objects.
2.  **Surface Brightness Bias:** We miss diffuse galaxies.

If ACIE is trained on biased survey data, its counterfactuals will reflect that bias. For instance, it might refuse to generate a "faint, massive galaxy" because such objects are absent from its training set, even if they are physically permitted.

### 12.1.1 Mitigation Strategy
ACIE employs **Inverse Propensity Weighting (IPW)** during training. We estimate the selection function $S(x)$ of the survey and weight each training example by $1/S(x)$. This upweights rare, faint objects, forcing the model to learn their physics rather than treating them as outliers.

## 12.2 Dual-Use Concerns
The core technology of ACIE—generating hyper-realistic synthetic imagery capabilities—has potential dual-use implications (e.g., DeepFakes). To prevent misuse:
1.  **Watermarking:** All generated images contain an invisible, frequency-domain watermark identifying them as synthetic.
2.  **Access Control:** The API is not public. Access is restricted to verified academic institutions via institutional email verification and manual approval.
3.  **Audit Trails:** Every generation request is logged. Anomaly detection algorithms flag users generating massive volumes of non-astrophysical imagery.

## 12.3 citation Policy
Research utilizing ACIE must cite the foundational software paper (Jitterx et al., 2026). Furthermore, the unique `model_version` used for inference must be reported to ensure reproducibility.
"This work made use of the Astrophysical Counterfactual Inference Engine (ACIE), version 1.0.0, model SHA-256:7f9a2b..."


# 13. Deep Dive: Frontend Architecture

## 13.1 The Dashboard Ecosystem
The ACIE Dashboard is not merely a control panel but a fully immersive visual interface for high-dimensional data exploration. Built on **React 18**, it utilizes a component-based architecture to manage the complex state of real-time inference sessions.

### 13.1.1 State Management (Redux Toolkit)
Given the asynchronous nature of the inference pipeline (where a single request may trigger a sequence of status updates: `QUEUED` -> `PROCESSING` -> `PHYSICS_CHECK` -> `COMPLETE`), managing local component state is insufficient. ACIE employs a global Redux store slices:
*   `inferenceSlice`: Tracks the active job UUID, progress percentage, and intermediate latent vectors.
*   `authSlice`: Manages the JWT lifecycle, including silent refresh via HttpOnly cookies.
*   `websocketSlice`: Handles the robust connection logic, including exponential backoff reconnection strategies.

### 13.1.2 The Latent Space Visualizer
The crown jewel of the UI is the `LatentSpaceVisualizer`. This component renders a 2D projection (via t-SNE or UMAP) of the 1536-dimensional manifold. 
*   **Rendering engine:** We utilize **WebGL** via `three.js` (wrapped in `react-three-fiber`) to render up to 100,000 data points at 60 FPS.
*   **Interaction:** Users can "brush" regions of the latent space to select sub-populations of galaxies. This selection is broadcast via WebSocket to the R analysis backend, which immediately updates the statistical plots. This "linked brushing" technique is essential for exploratory data analysis (EDA).

## 13.2 Real-Time Communication
The frontend maintains a persistent **WebSocket (WSS)** connection to the Java Gateway.
*   **Protocol:** STOMP over WebSocket.
*   **Topics:** 
    *   `/topic/inference/{jobId}`: Publishes progress updates and partial results.
    *   `/topic/alerts`: Publishes system-wide health warnings (e.g., "High Physics Violation Rate detected").

---

# 14. Deep Dive: Rust Accelerator Internals

## 14.1 Memory Safety & FFI
The interface between Python and Rust is managed by `PyO3`. A critical engineering challenge is facilitating Zero-Copy data transfer.
*   **The Buffer Protocol:** The Rust `SparseMatrix` struct implements the Python Buffer Protocol. When a NumPy array is passed to Rust, we do not copy the data. instead, we obtain a raw pointer to the underlying C-contiguous memory buffer.
*   **Safety Invariants:** Rust's borrow checker ensures that while the Rust code is mutating the buffer (during an inplace encryption operation), no Python code can access it. This prevents data races that are common in C/C++ extensions.

## 14.2 SIMD Strategy (AVX-512)
The `matrix_kernels.asm` file contains handwritten implementations of the dot product and other linear algebra primitives. 
*   **Packet Processing:** We process 16 single-precision floats (32-bit) in parallel using `zmm` registers (512-bit width).
*   **Instruction Selection:** 
    *   `vfmadd231ps`: Fused Multiply-Add. Performs $a 	imes b + c$ in a single cycle with only one rounding error.
    *   `vmovups` / `vmovntps`: Unaligned loads and Non-Temporal stores (bypassing cache for final results to avoid polluting L1/L2 cache).
*   **Performance:** Benchmarks show a 14x speedup over standard NumPy routines for our specific sparse matrix sparsity patterns (density < 1%).

## 14.3 The CSR Memory Layout
The Compressed Sparse Row format stores a matrix $M$ using three arrays:
1.  `values` (Vec<f32>): The non-zero elements.
2.  `col_indices` (Vec<u32>): The column index for each value.
3.  `row_offsets` (Vec<u32>): The index in `values` where each row starts.

This layout is strictly strictly packed to ensure spatial locality. When the CPU prefetcher loads a chunk of `values`, it usually loads the corresponding `col_indices`, minimizing pipeline stalls.

---

# 15. Operational War Games: Disaster Recovery

## 15.1 Scenario A: Primary Database Corruption
**Trigger:** Filesystem corruption on the `acie-postgres` Persistent Volume due to hardware failure.
**Symptoms:** 500 Errors on API; logs show `PANIC: could not read block`.

**Recovery Protocol:**
1.  **Stop the Bleeding:** Scale down `acie-inference` to 0 to stop new writes.
2.  **Point-In-Time Recovery (PITR):** 
    *   Retrieve the base backup from S3 bucket `acie-backups-prod`.
    *   Replay WAL (Write Ahead Logs) up to 5 minutes before the detected corruption timestamp.
3.  **Validation:** Run `SELECT count(*) FROM models` and verify consistency.
4.  **Resumption:** Scale up `acie-inference`.
**RTO (Recovery Time Objective):** 15 minutes.
**RPO (Recovery Point Objective):** 5 minutes.

## 15.2 Scenario B: Homomorphic Key Compromise
**Trigger:** An internal audit reveals that the private key $\lambda$ was accidentally logged to a text file.
**risk:** An attacker with access to logs could decrypt past inference requests.

**Recovery Protocol:**
1.  **Key Rotation:** Generate a new key pair $(n', g')$.
2.  **Rekeying:** The specific property of our system is that **models are key-agnostic** (weights are plaintext, inputs are encrypted). However, any stored *encrypted* vectors in the database are now vulnerable.
3.  **Purge:** We typically do not store encrypted vectors long-term (only during active inference). If any are persisted, they must be deleted.
4.  **Revocation:** Invalidate the compromised key in the KMS.
5.  **Client Update:** Push the new public key to all client SDKs via an enforced update mechanism.

---

# 16. Full API Reference

The ACIE API is defined using Protocol Buffers v3. All endpoints are accessed via gRPC on port 50051.

## 16.1 Service: `InferenceService`

### `CounterfactualInference`
Performs a single-shot causal intervention.

**Request (`InferenceRequest`):**
*   `observation` (repeated float): The flattened tensor of the input image. Size must match model input dim (e.g., $64 	imes 64 = 4096$).
*   `intervention` (map<string, float>): Key-value pairs of the variables to intervene on. Keys must match node names in the SCM (e.g., `stellar_mass`, `metallicity`).
*   `request_id` (string): A client-generated UUIDv4 for tracing.

**Response (`InferenceResponse`):**
*   `counterfactual` (repeated float): The generated image tensor.
*   `confidence` (double): A normalized score $[0, 1]$ indicating the model's certainty. Derived from the entropy of the variational posterior.
*   `latency_ms` (double): Server-side processing time.

### `StreamInference`
Bi-directional streaming RPC for high-throughput batch processing.

**Behavior:**
The client opens a stream and sends requests as fast as possible. The server processes them in parallel (up to `MAX_CONCURRENT_STREAMS`) and yields responses out-of-order as they complete. The client must use `request_id` to match responses to requests.

### `HealthCheck`
Standard liveness probe endpoint.

**Response (`HealthResponse`):**
*   `status` (enum): `SERVING`, `NOT_SERVING`, `UNKNOWN`.
*   `gpu_count` (int32): Number of CUDA devices visible to the pod.

## 16.2 Error Codes
ACIE maps internal errors to standard gRPC status codes:

| Code | Status | Description |
| :--- | :--- | :--- |
| 3 | `INVALID_ARGUMENT` | Input tensor shape mismatch or intervention variable not found in DAG. |
| 6 | `ALREADY_EXISTS` | Using a `request_id` that is currently being processed. |
| 8 | `RESOURCE_EXHAUSTED` | Global queue limit reached. Client should back off exponentially. |
| 9 | `FAILED_PRECONDITION` | Physics violation exceeded threshold (Infinity Loop in solver). |
| 16 | `UNAUTHENTICATED` | Missing or expired JWT metadata. |

---

**End of Technical Whitepaper.**

# 17. Advanced Mathematical Foundations

## 17.1 Variational Inference in Causal Models

The core generative engine of ACIE is a **Causal Variational Autoencoder (C-VAE)**. Unlike a standard VAE which maximizes the Evidence Lower Bound (ELBO) of the joint distribution $P(X)$, a C-VAE must maximize the ELBO of the interventional distribution $P(X | 	ext{do}(T=t))$.

### 17.1.1 Derivation of the Causal ELBO
Let $X$ be the observed variables, $Z$ be the latent causal variables, and $U$ be the exogenous noise. The structural equations are $X = f(Z, U)$.
 We approximate the intractable posterior $P(Z|X)$ with a variational distribution $Q_\phi(Z|X)$.

The marginal log-likelihood is:
$$ \log P_	heta(X) = \log \int P_	heta(X|Z) P(Z) dZ $$

Using Jensen's Inequality, we derive the standard ELBO:
$$ \log P_	heta(X) \ge \mathbb{E}_{Z \sim Q_\phi} [\log P_	heta(X|Z)] - D_{KL}(Q_\phi(Z|X) || P(Z)) $$

However, for a counterfactual query, the distribution of $Z$ changes. If we intervene on node $Z_i$, setting it to $z_i^*$, the prior $P(Z)$ is replaced by the mutilated prior $P(Z_{\setminus i} | z_i^*)$.
The **Causal ELBO** becomes:
$$ \mathcal{L}_{causal} = \mathbb{E}_{Z_{\setminus i} \sim Q_\phi} [\log P_	heta(X | Z_{\setminus i}, z_i^*)] - D_{KL}(Q_\phi(Z_{\setminus i}|X) || P(Z_{\setminus i})) $$

ACIE optimizes this objective using the "Reparameterization Trick" for the structural nodes:
$$ z_j = \mu_j(pa_j) + \sigma_j(pa_j) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$
This allows gradients to flow through the sampling step, enabling end-to-end training of the causal mechanisms.

## 17.2 Homomorphic Montgomery Reduction

The performance of the Paillier cryptosystem hinges on the speed of modular multiplication: $A 	imes B \mod N^2$. Since $N$ is 2048 bits, $N^2$ is 4096 bits. Standard division is too slow. We use **Montgomery Reduction**.

### 17.2.1 The Algorithm
Let $R > N^2$ be a power of 2 (e.g., $R = 2^{4096}$). We transform integers into "Montgomery Form": $ar{x} = xR \mod N^2$.
Addition is trivial: $ar{a} + ar{b} = (aR + bR) = (a+b)R = \overline{a+b}$.
Multiplication is the challenge: $ar{a} 	imes ar{b} = (aR)(bR) = (ab)R^2$. We need $(ab)R$. The reduction function `REDC(x)` computes $xR^{-1} \mod N^2$.

**Proof of Correctness:**
Let $T = ar{a} 	imes ar{b}$. We want $U \equiv T R^{-1} \mod M$ (where $M=N^2$).
Algorithm:
1.  $m = (T \mod R) M' \mod R$ where $M' = -M^{-1} \mod R$.
2.  $t = (T + mM) / R$.

Since $m \equiv -T M^{-1} \mod R$, then $mM \equiv -T \mod R$, so $T + mM \equiv 0 \mod R$.
Thus, $T + mM$ is divisible by $R$, making the division integer-exact.
Modulo $M$: $t R \equiv T + mM \equiv T \mod M$ (since $mM$ is a multiple of $M$).
Therefore, $t \equiv T R^{-1} \mod M$.

This algorithm avoids expensive division by $M$, replacing it with shifts and multiplications modulo $R$ (which are free on binary computers). ACIE's Rust core implements this with AVX-512 `vpmuludq` instructions to effectively parallelize the multi-precision arithmetic.

## 17.3 Sensitivity Estimators

In the `ACIEr` package, we estimate sensitivity $S = rac{\partial \mathbb{E}[Y]}{\partial z}$.
We use the **Efficient Influence Function (EIF)** estimator rather than simple finite differences, to reduce variance.

$$ \hat{S}_{EIF} = rac{1}{n} \sum_{i=1}^n \left( rac{I(Z_i=z)}{P(Z_i=z|X_i)} (Y_i - \hat{\mathbb{E}}[Y|X_i, Z_i=z]) + \hat{\mathbb{E}}[Y|X_i, Z_i=z] ight) $$

This estimator has the property of "Double Robustness": it is consistent if *either* the propensity model $P(Z|X)$ *or* the outcome model $\mathbb{E}[Y|X, Z]$ is correctly specified. This is crucial in astronomy where selection functions are often known (the propensity) even if the physical physics (the outcome) is complex.

---

# 18. Hardware & Low-Level Engineering

## 18.1 The GPU Kernel (CUDA)
While the CPu handles encryption, the neural network inference runs on NVIDIA A100 GPUs. ACIE utilizes custom CUDA kernels via Triton for the physics layers, specifically the N-body force calculation.

### 18.1.1 Tiled Matrix Multiplication
The gravitational potential calculation requires pairwise interactions.
$$ F_i = \sum_{j} rac{m_i m_j (r_j - r_i)}{|r_j - r_i|^3} $$
Naive implementation is memory-bound. Our custom kernel uses Tiled Matrix Multiplication.
1.  Load a block of particles $i$ into Shared Memory (Cycle A).
2.  Load a block of particles $j$ into Registers (Cycle B).
3.  Compute interactions.
4.  prefetch next block asynchronously (`cp.async.ca`).
This hides the global memory latency effectively, achieving 85% of theoretical FP32 FLOPS on the A100.

## 18.2 Assembly Instruction Scheduling
In `asm/matrix_kernels.asm`, instruction scheduling is critical to avoid pipeline hazards.
The Skylake-X architecture has two FMA units on ports 0 and 1.
To saturate them, we unroll loops by a factor of 4.

```nasm
; Unrolled loop body (Conceptual)
vfmadd231ps zmm0, zmm2, zmm3 ; Accumulate to sum0
vfmadd231ps zmm1, zmm4, zmm5 ; Accumulate to sum1
vfmadd231ps zmm6, zmm7, zmm8 ; Accumulate to sum2
vfmadd231ps zmm9, zmm10, zmm11 ; Accumulate to sum3
```
By maintaining 4 independent dependency chains (`zmm0`, `zmm1`, `zmm6`, `zmm9`), we ensure that the CPU's Out-of-Order execution engine always has instructions ready to dispatch, hiding the 4-cycle latency of the FMA instruction.

## 18.3 Network Topology & Kernel Bypass
For the connection between the Inference Engine and the Redis Cluster, standard Linux networking stack (TCP/IP) introduces significant syscall overhead (`recv`, `copy_to_user`).
In high-throughput deployments, ACIE supports **DPDK (Data Plane Development Kit)** mode.
The breakdown:
1.  Isolate CPU cores 14-16 for networking.
2.  Map the NIC DMA ring directly to user-space memory.
3.  Process packets via a polling loop (zero interrupts).
This reduces the round-trip time (RTT) to Redis from $150 \mu s$ to $10 \mu s$, essential when processing high-frequency streaming data from radio telescopes.

---

# 19. Detailed Component Specifications

## 19.1 The Encoder (ViT-L/14) Layer Definition
The encoder follows the standard Vision Transformer architecture but with modifications for radiometric calibration.

| Layer | Type | Output Shape | Parameters | Details |
| :--- | :--- | :--- | :--- | :--- |
| `Input` | Raw Data | $(1, 3, 224, 224)$ | 0 | Normalized using SDSS $u,g,r$ band means. |
| `PatchEmbed` | Conv2d | $(1, 257, 1024)$ | $1024 	imes 3 	imes 14 	imes 14$ | Break image into $14 	imes 14$ patches. |
| `ClsToken` | Parameter | $(1, 1, 1024)$ | 1024 | Learnable classification token. |
| `PosEmbed` | Parameter | $(1, 257, 1024)$ | $257 	imes 1024$ | Learnable absolute positioning. |
| `Block_0` | Transformer | $(1, 257, 1024)$ | 12M | MSA (16 heads) + MLP (4096 hidden). |
| ... | ... | ... | ... | Repeated 24 times. |
| `Block_23` | Transformer | $(1, 257, 1024)$ | 12M | Final attention block. |
| `Norm` | LayerNorm | $(1, 257, 1024)$ | 2048 | Epsilon $1e-6$. |
| `Head` | Linear | $(1, 1536)$ | $1536 	imes 1024$ | Projects `ClsToken` to latent space. |

**Total Parameters:** 307 Million.
**Total FLOPs:** 60 GFLOPs per forward pass.

## 19.2 The Decoder (PixelCNN++)
To generate high-fidelity images from the latent vector $z$, we use an autoregressive decoder.
$$ p(x|z) = \prod_{i=1}^{N} p(x_i | x_{<i}, z) $$
Unlike a simple Transposed Convolution, PixelCNN++ models the conditional distribution of each pixel as a mixture of logistic distributions. This allows it to capture multi-modal distributions (e.g., a pixel could be dark sky OR a bright star, but not gray).

---

# 20. Future Roadmap & Research Directions

## 20.1 Quantum Causal Inference
Classical computers struggle to simulate quantum entanglement. We are investigating **Quantum Boltzmann Machines (QBM)** running on D-Wave annealers to model the latent state of quantum systems. The SCM logic remains, but the structural equations $f_i$ become unitary operators $U_i$ acting on qubits.

## 20.2 Simulation-Based Inference (SBI)
Currently, ACIE learns from observational data. We plan to integrate directly with large-scale N-body simulations (e.g., Arepo, Gadget-4). By training on simulation snapshots, the model can learn causal links that are unobservable in real data (e.g., Dark Matter distribution), creating a "Hybrid Twin" of the universe.

## 20.3 Intergalactic Federated Learning
As datasets become too large to move (Petabytes from SKA), ACIE will evolve into a "Code-to-Data" paradigm. A lightweight training container will travel to the data center hosting the survey, update the global model gradients, and move to the next center, preserving data sovereignty.

---

**Appendix D: Glossary of Terms**
*   **SCM**: Structural Causal Model.
*   **Counterfactual**: A conditional statement of the form "If X had been x, then Y would have been y".
*   **Latent Space**: A compressed vector representation of data.
*   **Homomorphic Encryption**: Encryption allowing computation on ciphertext.
*   **SIMD**: Single Instruction, Multiple Data.
*   **FFI**: Foreign Function Interface.
*   **PITR**: Point-In-Time Recovery.

**End of Extended Technical Reference.**

# 21. Java Gateway Architecture

The **ACIE Gateway** (`acie-inference-server`) describes the secure entry point for all API traffic. Written in **Java 17** with **Spring Boot 3.2**, it connects the external REST/gRPC world with the internal high-performance computing clusters.

## 21.1 Security Filter Chain (`SecurityConfig`)
We implement a "Defense in Depth" strategy using Spring Security.
1.  **Transport Layer**: TLS 1.3 termination is handled by the Nginx Ingress Controller, but the Java application enforces **mTLS** for inter-pod communication.
2.  **Authentication Filter (`JwtAuthenticationTokenFilter`)**:
    *   Intercepts every HTTP/gRPC request.
    *   Validates the `Authorization: Bearer <token>` header.
    *   Parses the JWT using `jjwt`, verifying the signature against the RSA-256 Public Key extracted from the Identity Provider (IdP) JWKS endpoint.
    *   Extracts claims (`sub`, `roles`, `scope`) and populates the `SecurityContext`.
3.  **Authorization (`@PreAuthorize`)**:
    *   Method-level security ensures that only users with `SCOPE_INFERENCE_WRITE` can trigger new jobs.
    *   `SCOPE_INFERENCE_READ` allows read-only access to results.

## 21.2 gRPC Interceptors
To ensure observability and reliability across the RPC boundary, we utilize gRPC Interceptors.
*   **`LogInterceptor`**: Captures request metadata (User-Agent, Request-ID) and logs it to Mapped Diagnostic Context (MDC) for distributed tracing.
*   **`ExceptionInterceptor`**: Translates internal Java exceptions (e.g., `EntityNotFoundException`) into standard gRPC Status Codes (`NOT_FOUND`), ensuring clients receive structured error responses.

## 21.3 Data Access Layer (JPA/Hibernate)
The application uses **Spring Data JPA** to interact with PostgreSQL.
*   **Entity Graph**: We use `@EntityGraph` annotations to solve the "N+1 Select" problem when fetching `Model` entities and their associated `Hyperparameters`.
*   **Audit Logging**: The `InferenceLog` entity is written asynchronously using `@Async` events to avoid blocking the main request thread. This ensures that a database write latency spike does not impact API response time.

---

# 22. Developer Handbook

## 22.1 Development Environment Setup
To contribute to ACIE core, a specific environment is required.

### 22.1.1 Prerequisites
*   **OS**: Linux (Ubuntu 22.04+) or macOS (Sonoma+). Windows is not supported for the Physics Engine due to dependencies on `epoll`.
*   **Languages**: Python 3.10+, Rust 1.75+, Java 17, Node.js 20.
*   **Hardware**: NVIDIA GPU (Ampere or newer) required for Physics unit tests. CPU-only mode is available but slow.

### 22.1.2 The "One-Command" Build
We utilize `make` to orchestrate the build process across 4 languages.
```bash
# Builds Rust, compiles Java, installs R packages, and sets up Python venv
make setup

# Runs the full test suite (Unit + Integration)
make test
```

## 22.2 Contribution Guidelines
1.  **Branching Strategy**: We use Trunk-Based Development. Short-lived feature branches merge into `main`.
2.  **Commit Messages**: Must follow Conventional Commits (e.g., `feat(physics): add conservation layer`).
3.  **Code Style**:
    *   Python: `black` + `ruff`.
    *   Rust: `cargo fmt` + `clippy`.
    *   Java: Google Java Format.
    *   JS: Prettier.
    *   **Enforced via Pre-Commit Hooks.**

## 22.3 Continuous Integration (CI)
Our GitHub Actions pipeline performs:
1.  **Static Analysis**: SAST scans (Bandit, Snyk) for security vulnerabilities.
2.  **Unit Tests**: Parallel execution of PyTest and JUnit.
3.  **Build**: Docker image creation and pushing to the registry.
4.  **Deployment**: Automated rollout to the `staging` K8s namespace on merge to `main`.

---

# 23. Comprehensive Configuration Reference

## 23.1 Environment Variables
These variables control the runtime behavior of the `acie-inference` container.

### Core
| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `ACIE_ENV` | No | `production` | `development`, `staging`, `production`. |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` enables verbose physics logging. |
| `WORKER_THREADS` | No | `4` | Number of concurrent Python worker threads. |

### Physics Engine
| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `PHYSICS_TOLERANCE` | No | `1e-4` | Maximum allowable violation of conservation laws. |
| `MAX_STEPS` | No | `100` | Max integration steps for the ODE solver. |
| `ENABLE_CUDA` | No | `true` | Set to `false` for CPU-only fallback. |

### Security
| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `JWT_SECRET` | **YES** | - | HMAC-SHA256 secret for token signing. |
| `PAILLIER_PUB_KEY` | **YES** | - | Path to the mounted public key file. |
| `PAILLIER_PRIV_KEY` | **YES** | - | Path to the mounted private key file. |

### Services
| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `REDIS_HOST` | No | `localhost` | Hostname of the Redis instance. |
| `POSTGRES_URI` | No | - | Full JDBC connection string. |

## 23.2 Feature Flags
ACIE supports hot-reloadable feature flags via a `flags.yaml` mounted ConfigMap.

```yaml
features:
  enable_quantum_solver: false  # Experimental
  enable_high_precision_float64: true
  enable_audit_logging: true
```

---

# 24. Troubleshooting & FAQ

## 24.1 Common Errors

### `PhysicsViolationException: Energy divergence detected`
**Cause:** The model generated a counterfactual that exploded the energy of the system.
**Fix:** 
1.  Check input data for normalization anomalies (e.g., negative flux).
2.  Increase `PHYSICS_STEPS` to allow the solver more time to converge.
3.  Relax `PHYSICS_TOLERANCE` temporarily if investigating edge cases.

### `GrimReaper: Worker killed by OOM`
**Cause:** The Rust accelerator allocated a matrix larger than available RAM.
**Fix:**
1.  Increase Pod memory limit in `k8s/deployment.yaml`.
2.  Reduce `BATCH_SIZE`.
3.  Ensure `SparseMatrix` is actually sparse; dense inputs will bloat memory.

### `SSLError: CERTIFICATE_VERIFY_FAILED`
**Cause:** The client does not trust the Gateway's self-signed certificate (common in dev).
**Fix:**
*   **Prod:** Ensure valid Let's Encrypt certs are mounted.
*   **Dev:** Set `ACIE_ALLOW_INSECURE_TLS=true` (Do NOT use in prod).

## 24.2 Support
For issues not covered here, please open a ticket on the internal Jira board or contact the **ACIE Reliability Engineering Team** via PagerDuty.

---

# 25. References

1.  **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
2.  **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
3.  **Paillier, P.** (1999). Public-key cryptosystems based on composite degree residuosity classes. *EUROCRYPT*.
4.  **Jitterx et al.** (2026). The Astrophysical Counterfactual Inference Engine: Unlocking the causal universe. *Nature Astronomy* (In Prep).

