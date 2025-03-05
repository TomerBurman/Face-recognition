"""
data_generator.py

This module defines a Keras Sequence generator that yields balanced batches of image pairs
and their corresponding labels. Each batch contains an equal number of positive (same person)
and negative (different persons) pairs.
"""

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
import random

class PairDataGenerator(Sequence):
    def __init__(self, pairs, labels, batch_size=32, target_size=(224, 224)):
        """
        Initializes the generator.

        Args:
            pairs (list): List of tuples (img_path1, img_path2).
            labels (list): List of labels for each pair (1 for positive, 0 for negative).
            batch_size (int): Number of samples per batch (must be even for balanced sampling).
            target_size (tuple): The target size to which images will be resized.
        """
        assert batch_size % 2 == 0, "Batch size must be even for balanced sampling."
        self.batch_size = batch_size
        self.target_size = target_size
        
        # Separate pairs by their labels.
        self.positive_pairs = [pair for pair, label in zip(pairs, labels) if label == 1]
        self.negative_pairs = [pair for pair, label in zip(pairs, labels) if label == 0]
        
        # Shuffle the lists at the start.
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        pos_batches = len(self.positive_pairs) // (self.batch_size // 2)
        neg_batches = len(self.negative_pairs) // (self.batch_size // 2)
        return min(pos_batches, neg_batches)

    def __getitem__(self, idx):
        """
        Generates one batch of data.
        """
        half_batch = self.batch_size // 2
        
        # Randomly sample positive and negative pairs.
        batch_positive = random.sample(self.positive_pairs, half_batch)
        batch_negative = random.sample(self.negative_pairs, half_batch)
        
        # Combine pairs and labels.
        batch_pairs = batch_positive + batch_negative
        batch_labels = [1] * half_batch + [0] * half_batch
        
        # Shuffle the combined batch.
        combined = list(zip(batch_pairs, batch_labels))
        random.shuffle(combined)
        batch_pairs, batch_labels = zip(*combined)
        
        batch_X1, batch_X2 = [], []
        for img_path1, img_path2 in batch_pairs:
            img1 = self.preprocess_image(img_path1)
            img2 = self.preprocess_image(img_path2)
            # Use zeros if an image fails to load.
            batch_X1.append(img1 if img1 is not None else np.zeros((*self.target_size, 3), dtype='float32'))
            batch_X2.append(img2 if img2 is not None else np.zeros((*self.target_size, 3), dtype='float32'))
        
        return (np.array(batch_X1), np.array(batch_X2)), np.array(batch_labels)

    def preprocess_image(self, image_path):
        """
        Loads and preprocesses an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            numpy.ndarray: The preprocessed image as a float32 array.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        # Convert from BGR (OpenCV default) to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.target_size)
        # Convert to float32; values remain in [0,255].
        return image_resized.astype('float32')

    def on_epoch_end(self):
        """
        Shuffles the positive and negative pairs independently after each epoch.
        """
        random.shuffle(self.positive_pairs)
        random.shuffle(self.negative_pairs)
