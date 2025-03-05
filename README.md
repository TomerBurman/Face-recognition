# Face Recognition with Siamese Networks

This repository implements a face recognition pipeline using Siamese networks. The project covers the entire workflow from data preprocessing and augmentation to pair generation, model training, evaluation, and inference.

## Project Overview

The goal of this project is to build a robust face recognition system using a Siamese network architecture. The key components are:

- **Data Preparation & Augmentation:**  
  Processes raw face images (from datasets such as CASIA-WebFace or LFW) by detecting and aligning faces, and applies data augmentation to expand the dataset.

- **Pair Generation:**  
  Generates positive (same person) and negative (different persons) image pairs for training the Siamese network.

- **Siamese Network:**  
  Implements a Siamese network using the Embedding model, using MobileNetV2 as the underlying backbone network
  
  The network uses contrastive loss to learn an embedding space where images of the same person are close and those of different persons are far apart.

- **Training & Evaluation:**  
  Provides training utilities with Keras generators and callbacks (ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau) to help prevent overfitting. The evaluation script computes standard metrics (accuracy, precision, recall, F1 score, ROC AUC) on a test set of generated pairs.

- **Inference & Embedding Extraction:**  
  After training, the model can be used for face verification by computing the Euclidean distance between embeddings. An additional module extracts embeddings for visualization (using t-SNE or PCA) and further analysis.



