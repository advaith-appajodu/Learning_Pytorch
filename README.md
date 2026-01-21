# Learning_PyTorch

This repository contains a collection of PyTorch projects built to develop a strong practical understanding of deep learning models, training dynamics, and evaluation techniques. It spans work from basic feedforward neural networks to convolutional neural networks and Transformer-based sequence-to-sequence models.

The primary goal of this repository is hands-on learning: implementing models from scratch, experimenting with architectures and optimization strategies, and understanding common failure modes such as overfitting, instability, and poor generalization.

---

## Project Overview

### Feedforward Neural Networks
- Implemented multi-layer fully connected neural networks in PyTorch
- Explored the impact of:
  - Network depth
  - Optimizers (SGD, Adam, AdamW)
  - Learning rate selection
  - Batch normalization, dropout, and weight decay
- Achieved strong performance after targeted optimization and regularization

---

### Convolutional Neural Networks (CIFAR-10)
- Built CNNs from scratch for image classification on CIFAR-10
- Iteratively improved architectures using:
  - Increased channel depth
  - Batch normalization
  - Adaptive pooling
  - Dropout and weight decay
- Diagnosed and mitigated overfitting through:
  - Data augmentation (random cropping and rotation)
  - Learning rate tuning
- Final models demonstrate stable training and improved generalization

---

### Transformer (English to Telugu Translation)
- Implemented a Transformer-based sequence-to-sequence model in PyTorch
- Trained using teacher forcing for efficient convergence
- Evaluated using BLEU score with:
  - Indic-language tokenization
  - Corpus-level BLEU
  - Smoothing for stability
- Identified and fixed end-of-sequence handling errors that caused hallucinated outputs
- Supports autoregressive decoding during inference

---

## Tools and Technologies
- PyTorch
- Weights and Biases (WandB) for experiment tracking
- Hugging Face Datasets
- Indic NLP Library

---

## Purpose
This repository serves as a learning-focused workspace for:
- Understanding deep learning training dynamics
- Exploring architectural and optimization trade-offs
- Gaining intuition for regularization and evaluation metrics
- Transitioning from vision models to sequence-to-sequence Transformers

---

## Notes
- Large model weights are not included due to GitHub size limits
- Code prioritizes clarity and experimentation over production readiness
