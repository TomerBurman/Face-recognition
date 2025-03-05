"""
trainer.py

This module defines the Trainer class, which prepares data pairs for the Siamese network,
builds the model, orchestrates the training process using Keras generators, and saves the
generated image pairs for further analysis.
"""

import os
import tensorflow as tf
from data_preparation import DataPairGenerator
from data_generator import PairDataGenerator
from siamese_model import EmbeddingModel, SiameseNetwork
import random
import cv2
import numpy as np

class Trainer:
    def __init__(self, processed_dir, batch_size=32, target_size=(224,224), num_pairs_per_person=2):
        """
        Initializes the Trainer.

        Args:
            processed_dir (str): Directory with processed images (organized by person).
            batch_size (int): Batch size for training.
            target_size (tuple): Target image size.
            num_pairs_per_person (int): Number of positive pairs per person to generate.
        """
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.target_size = target_size
        
        # Generate pairs using DataPairGenerator.
        data_pair_gen = DataPairGenerator(self.processed_dir)
        self.pairs, self.labels = data_pair_gen.create_image_pairs(num_pairs_per_person=num_pairs_per_person)
        
        # Shuffle pairs and labels.
        combined = list(zip(self.pairs, self.labels))
        random.shuffle(combined)
        self.pairs, self.labels = zip(*combined)
        self.pairs = list(self.pairs)
        self.labels = list(self.labels)
        
        # Split pairs into training and validation sets (80/20 split).
        num_samples = len(self.pairs)
        split_index = int(0.8 * num_samples)
        self.train_pairs = self.pairs[:split_index]
        self.train_labels = self.labels[:split_index]
        self.val_pairs = self.pairs[split_index:]
        self.val_labels = self.labels[split_index:]
        print("Total pairs:", len(self.pairs))
        
        # Create Keras generators.
        self.train_gen = PairDataGenerator(self.train_pairs, self.train_labels, batch_size=self.batch_size, target_size=self.target_size)
        self.val_gen = PairDataGenerator(self.val_pairs, self.val_labels, batch_size=self.batch_size, target_size=self.target_size)
        print("Train generator length:", len(self.train_gen), "Validation generator length:", len(self.val_gen))

    def build_model(self):
        """
        Builds and compiles the Siamese network model.

        Returns:
            tf.keras.Model: The compiled Siamese model.
        """
        embedding_model = EmbeddingModel(input_shape=(*self.target_size, 3))
        self.siamese_network = SiameseNetwork(embedding_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.siamese_network.compile(optimizer=optimizer)
        return self.siamese_network.model
    
    def train_model(self, epochs=20, callbacks=None):
        """
        Trains the Siamese network using the data generators.

        Args:
            epochs (int): Number of epochs.
            callbacks (list): List of Keras callbacks.

        Returns:
            History: Training history.
        """
        steps_per_epoch = len(self.train_gen)
        validation_steps = len(self.val_gen)
        history = self.siamese_network.train(
            self.train_gen,
            self.val_gen,
            steps_per_epoch,
            validation_steps,
            epochs=epochs,
            callbacks=callbacks
        )
        return history

    def save_pairs_to_folders(self, output_dir="../saved_pairs"):
        """
        Saves the generated image pairs to separate folders for positive and negative pairs.
        Each pair is read from disk, resized to the target size, concatenated horizontally,
        and saved as an image file.

        Args:
            output_dir (str): Base directory where the pairs will be saved.
        """
        pos_dir = os.path.join(output_dir, "positive_pairs")
        neg_dir = os.path.join(output_dir, "negative_pairs")
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)
        
        for idx, ((img_path1, img_path2), label) in enumerate(zip(self.pairs, self.labels)):
            # Load and resize images.
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)
            if img1 is None or img2 is None:
                continue
            img1 = cv2.resize(img1, self.target_size)
            img2 = cv2.resize(img2, self.target_size)
            
            # Concatenate images horizontally.
            concatenated = np.hstack((img1, img2))
            # Determine output folder based on label.
            folder = pos_dir if label == 1 else neg_dir
            output_path = os.path.join(folder, f"pair_{idx:05d}.jpg")
            cv2.imwrite(output_path, concatenated)
            print(f"Saved pair {idx} to {output_path}")

if __name__ == '__main__':
    # Example usage for training:
    trainer = Trainer(processed_dir='../data_processed', batch_size=32, target_size=(224, 224), num_pairs_per_person=2)
    model = trainer.build_model()
    
    # Set up callbacks.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('../models/siamese_best.h5', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    history = trainer.train_model(epochs=20, callbacks=callbacks)
    print("Training complete.")
    
    # Save the generated pairs for inspection.
    trainer.save_pairs_to_folders(output_dir="../saved_pairs")
