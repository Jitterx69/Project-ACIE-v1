#!/bin/bash
set -e

# ACIE Kubernetes Deployment Script
# Deploys ACIE infrastructure to Kubernetes cluster

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🚀 ACIE Kubernetes Deployment${NC}"
echo "======================================"

# Configuration
NAMESPACE="acie"
IMAGE_TAG=${1:-latest}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-acie}

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl not found. Please install kubectl.${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ docker not found. Please install Docker.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites met${NC}"

# Build Docker image
echo -e "\n${YELLOW}📦 Building Docker image...${NC}"
docker build -t ${DOCKER_REGISTRY}/inference:${IMAGE_TAG} -f Dockerfile.production .
docker tag ${DOCKER_REGISTRY}/inference:${IMAGE_TAG} ${DOCKER_REGISTRY}/inference:latest

# Push to registry
echo -e "\n${YELLOW}📤 Pushing to registry...${NC}"
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
docker tag ${DOCKER_REGISTRY}/inference:${IMAGE_TAG} ${DOCKER_REGISTRY}/inference:${GIT_SHA}

echo "Pushing images:"
echo "  - ${DOCKER_REGISTRY}/inference:${IMAGE_TAG}"
echo "  - ${DOCKER_REGISTRY}/inference:${GIT_SHA}"
echo "  - ${DOCKER_REGISTRY}/inference:latest"

# Uncomment to push to registry
# docker push ${DOCKER_REGISTRY}/inference:${IMAGE_TAG}
# docker push ${DOCKER_REGISTRY}/inference:${GIT_SHA}
# docker push ${DOCKER_REGISTRY}/inference:latest

# Create namespace
echo -e "\n${YELLOW}☸️  Creating namespace...${NC}"
kubectl apply -f k8s/namespace.yaml

# Apply configurations
echo -e "\n${YELLOW}⚙️  Applying configurations...${NC}"
kubectl apply -f k8s/configmap.yaml

echo -e "${YELLOW}Note: Secrets must be created manually for security${NC}"
echo -e "Create secrets using: kubectl create secret generic acie-secrets --from-literal=redis-password=... -n ${NAMESPACE}"

# Create PVCs
echo -e "\n${YELLOW}💾 Creating storage...${NC}"
kubectl apply -f k8s/pvc.yaml

# Deploy applications
echo -e "\n${YELLOW}🚢 Deploying applications...${NC}"
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# Wait for rollout
echo -e "\n${YELLOW}⏳ Waiting for deployment to complete...${NC}"
kubectl rollout status deployment/acie-inference -n ${NAMESPACE} --timeout=5m

# Verify deployment
echo -e "\n${GREEN}✅ Deployment complete!${NC}\n"

echo "Pods:"
kubectl get pods -n ${NAMESPACE}

echo -e "\nServices:"
kubectl get svc -n ${NAMESPACE}

echo -e "\nIngress:"
kubectl get ingress -n ${NAMESPACE}

# Get service URL
echo -e "\n${CYAN}Access Information:${NC}"
EXTERNAL_IP=$(kubectl get svc acie-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
echo "External IP: ${EXTERNAL_IP}"
echo "Health check: curl http://${EXTERNAL_IP}/health"

echo -e "\n${GREEN}🎉 Deployment successful!${NC}"
