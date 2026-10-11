# deploy/pi_setup.sh
# ------------------
# Run this on a fresh Raspberry Pi 4/5 (64-bit Raspberry Pi OS) to
# install all dependencies and build the NEON kernel library.
#
# Usage:
#   chmod +x pi_setup.sh && ./pi_setup.sh

set -e

echo "=== CXR-SIMD Pi Setup ==="

# ---- System dependencies ----
sudo apt-get update -qq
sudo apt-get install -y cmake g++ python3-pip python3-venv git

# ---- Python environment ----
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install onnxruntime numpy torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ---- Build NEON kernels ----
echo ""
echo "=== Building NEON kernels ==="
cd kernels
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ../..

echo ""
echo "=== Running validation ==="
./kernels/build/validate

echo ""
echo "=== Running micro-benchmark (kernel level) ==="
./kernels/build/bench

echo ""
echo "=== Running Python benchmark ==="
python deploy/run_inference.py \\
    --model quantize/onnx/int8_ptq.onnx \\
    --backend onnxruntime \\
    --runs 100

echo ""
echo "=== Setup complete. ==="
echo "Next: python deploy/run_inference.py --compare --model quantize/onnx/int8_ptq.onnx"
