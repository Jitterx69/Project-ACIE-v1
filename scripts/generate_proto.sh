#!/bin/bash
# Generate gRPC code from proto files

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Create output directory
mkdir -p acie/grpc

# Run protoc
# Try using venv python if available, else system python
if [ -f "venv/bin/python" ]; then
    PYTHON_CMD="./venv/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "Generating gRPC code using $PYTHON_CMD..."
$PYTHON_CMD -m grpc_tools.protoc -I protos --python_out=acie/grpc --grpc_python_out=acie/grpc protos/acie.proto

if [ $? -eq 0 ]; then
    echo "Successfully generated gRPC code in acie/grpc/"
else
    echo "Failed to generate gRPC code. Ensure grpcio-tools is installed."
    echo "pip install grpcio-tools"
    exit 1
fi
