#!/bin/bash
# Dừng script ngay lập tức nếu có bất kỳ lệnh nào bị lỗi
sudo apt update
sudo apt install python3.14-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./algebra
pip install -e ./analysis
pip install -e ./arithmetic
pip install ipython sympy numpy matplotlib
pip install pytest
