#!/bin/bash
set -e

echo "=============================="
echo " Creating virtual environment"
echo "=============================="

# Choose your Python
PYTHON=python

# Create venv
$PYTHON -m venv cnn_env

# Activate it
source cnn_env/bin/activate

echo "=============================="
echo " Upgrading pip"
echo "=============================="
pip install --upgrade pip setuptools wheel

echo "=============================="
echo " Installing required packages"
echo "=============================="

# Core scientific stack
pip install numpy pandas matplotlib

# TensorFlow + Keras (CPU version)
pip install tensorflow keras

# CV + clustering + image IO
pip install opencv-python scikit-learn pillow

# Progress, utils
pip install tqdm

echo "=============================="
echo " Installation complete"
echo "=============================="
echo "To activate the environment later, run:"
echo "    source cnn_env/bin/activate"