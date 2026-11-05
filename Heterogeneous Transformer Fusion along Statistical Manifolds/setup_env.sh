#!/bin/bash
set -e
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade scipy pot matplotlib
python -c "import torch, scipy, ot; print(torch.__version__, scipy.__version__); print(\"sm_120:\", \"sm_120\" in torch.cuda.get_arch_list())"
