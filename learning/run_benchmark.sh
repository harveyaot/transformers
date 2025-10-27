#!/bin/bash
# Convenience script to run DeltaNet benchmarks

set -e  # Exit on error

echo "================================================================"
echo "DeltaNet Attention Benchmark Runner"
echo "================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found. Please install Python 3.7+."
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import torch" 2>/dev/null || {
    echo "❌ PyTorch not found. Installing..."
    pip install torch
}

python -c "import matplotlib" 2>/dev/null || {
    echo "❌ Matplotlib not found. Installing..."
    pip install matplotlib
}

echo "✓ All dependencies satisfied"
echo ""

# Step 1: Quick test
echo "================================================================"
echo "Step 1: Running quick functionality test..."
echo "================================================================"
python quick_test.py
QUICK_TEST_RESULT=$?

if [ $QUICK_TEST_RESULT -ne 0 ]; then
    echo ""
    echo "❌ Quick test failed! Please check the error above."
    exit 1
fi

echo ""
echo "✅ Quick test passed!"
echo ""

# Ask user if they want to continue
read -p "Run full benchmark? This may take 10-30 minutes. (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Benchmark cancelled. You can run it later with:"
    echo "  python test_attn.py"
    exit 0
fi

# Step 2: Full benchmark
echo ""
echo "================================================================"
echo "Step 2: Running full benchmark..."
echo "================================================================"
echo ""
echo "Testing sequence lengths: 128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K"
echo ""
echo "⏱️  This will take a while. Go grab a coffee! ☕"
echo ""

# Run with nice to avoid hogging system
nice -n 10 python test_attn.py

BENCHMARK_RESULT=$?

echo ""
if [ $BENCHMARK_RESULT -eq 0 ]; then
    echo "================================================================"
    echo "✅ Benchmark completed successfully!"
    echo "================================================================"
    echo ""
    echo "Results saved to:"
    echo "  📊 attention_benchmark.png"
    echo ""
    echo "To view the plot:"
    if command -v open &> /dev/null; then
        echo "  open attention_benchmark.png"
    elif command -v xdg-open &> /dev/null; then
        echo "  xdg-open attention_benchmark.png"
    else
        echo "  Use your system's image viewer"
    fi
else
    echo "❌ Benchmark failed. Check the error above."
    exit 1
fi

