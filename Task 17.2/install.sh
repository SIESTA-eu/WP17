#!/bin/bash

# BERT NER Fine-tuning Environment Setup Script

echo "🚀 Setting up BERT NER Fine-tuning Environment..."

# Create conda environment
echo "📦 Creating conda environment 'bert-ner'..."
conda create -n bert-ner python=3.9 -y

# Activate environment
echo "🔄 Activating environment..."
conda activate bert-ner

# Install PyTorch with CUDA support (adjust CUDA version as needed)
echo "⚡ Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install core packages via conda
echo "📚 Installing core packages..."
conda install pandas numpy scikit-learn -c conda-forge -y

# Install transformers and related packages via pip
echo "🤖 Installing transformers and NLP packages..."
pip install transformers>=4.51.0
pip install accelerate>=0.26.0
pip install datasets
pip install seqeval
pip install tokenizers

# Install progress bars and utilities
echo "📊 Installing utilities..."
pip install tqdm
pip install ipywidgets  # For notebook progress bars

# Install Jupyter if needed
echo "📓 Installing Jupyter..."
conda install jupyter notebook ipykernel -c conda-forge -y

# Register kernel
echo "🔗 Registering kernel..."
python -m ipykernel install --user --name bert-ner --display-name "BERT NER"

echo "✅ Environment setup complete!"
echo ""
echo "To use this environment:"
echo "  conda activate bert-ner"
echo ""
echo "🎯 Your environment is ready for BERT NER fine-tuning!"