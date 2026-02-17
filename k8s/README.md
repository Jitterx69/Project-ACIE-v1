# Kubernetes Deployment Guide

This directory contains the manifests to deploy the full ACIE stack on Kubernetes.

## Prerequisites
- A Kubernetes cluster (v1.24+)
- `kubectl` configured
- Nginx Ingress Controller installed

## Components
- **acie-inference**: Python-based Inference Engine (with R support).
- **acie-java**: Java Spring Boot Gateway.
- **acie-frontend**: React Dashboard.
- **acie-redis**: Redis Cache.
- **acie-postgres**: PostgreSQL Database.

## Deployment Steps

1. **Create Namespace**
   ```bash
   kubectl apply -f namespace.yaml
   ```

2. **Configuration**
   ```bash
   kubectl apply -f configmap.yaml
   # IMPORTANT: Update secrets-template.yaml with real credentials before applying
   cp secrets-template.yaml secrets.yaml
   # (Edit secrets.yaml)
   kubectl apply -f secrets.yaml
   ```

3. **Infrastructure**
   ```bash
   kubectl apply -f redis.yaml
   kubectl apply -f postgres.yaml
   ```

4. **Applications**
   ```bash
   kubectl apply -f deployment.yaml        # Python Engine
   kubectl apply -f java-deployment.yaml   # Java Gateway
   kubectl apply -f frontend-deployment.yaml # Frontend
   ```

5. **Networking**
   ```bash
   kubectl apply -f service.yaml           # Python Service
   kubectl apply -f ingress.yaml           # Ingress Rules
   ```

## Verification
Access the dashboard at your Ingress IP (or localhost if using port-forwarding).

```bash
kubectl get pods -n acie
```
