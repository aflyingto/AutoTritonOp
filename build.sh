#!/bin/bash

# Build script for Triton-Ascend operators
# This script compiles and tests the Triton-Ascend operators

set -e

echo "=========================================="
echo "Building Triton-Ascend Operators"
echo "=========================================="

# Activate triton-ascend environment if needed
if [ -d "/home/wpf/triton-ascend" ]; then
    echo "Found triton-ascend directory"
    export PYTHONPATH=/home/wpf/triton-ascend:$PYTHONPATH
fi

# Check if triton-ascend is installed
python3 -c "import triton; print(f'Triton version: {triton.__version__}')" || {
    echo "Error: triton-ascend is not installed or not in PYTHONPATH"
    echo "Please install triton-ascend first"
    exit 1
}

# Check if torch_npu is available
python3 -c "import torch_npu; print(f'torch_npu is available')" || {
    echo "Warning: torch_npu is not available"
    echo "Some tests may fail without torch_npu"
}

echo ""
echo "=========================================="
echo "Testing Operators"
echo "=========================================="

# Change to the AutoTritonOps directory
cd /home/wpf/AutoTritonOps

# Run accuracy tests
echo "Running accuracy comparison tests..."
python3 test_accuracy.py

echo ""
echo "=========================================="
echo "Build Complete"
echo "=========================================="
echo "All operators have been tested successfully!"
echo ""
echo "Generated files:"
echo "  - /home/wpf/AutoTritonOps/ops/vector_add.py"
echo "  - /home/wpf/AutoTritonOps/ops/softmax.py"
echo "  - /home/wpf/AutoTritonOps/ops/layer_norm.py"
echo "  - /home/wpf/AutoTritonOps/ops/flash_attention.py"
echo "  - /home/wpf/AutoTritonOps/ops/matmul.py"
echo "  - /home/wpf/AutoTritonOps/test_accuracy.py"
echo "  - /home/wpf/AutoTritonOps/build.sh"
