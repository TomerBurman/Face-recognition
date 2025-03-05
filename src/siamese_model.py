"""
siamese_model.py

This module defines the Siamese network architecture using MobileNetV2 as the underlying 
embedding model. The embedding model extracts a 128-dimensional feature vector from an input image,
and the Siamese network computes the Euclidean distance between two such embeddings.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
import tensorflow.keras.backend as K

class EmbeddingModel:
    def __init__(self, input_shape=(224, 224, 3), embedding_dim=128):
        """
        Initializes the embedding model.

        Args:
            input_shape (tuple): Shape of input images.
            embedding_dim (int): Dimension of the output embedding.
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self.build_model()
    
    def build_model(self):
        """
        Builds the embedding model using MobileNetV2 pre-trained on ImageNet.

        Returns:
            tf.keras.Model: The embedding model.
        """
        # Load MobileNetV2 without the top classification layers.
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )
        # Unfreeze the last 20 layers for fine-tuning.
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        
        inputs = Input(shape=self.input_shape)
        # Preprocess input as expected by MobileNetV2.
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x)
        # Add Dense layer with L2 regularization.
        x = layers.Dense(self.embedding_dim, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        # Optionally add dropout.
        x = layers.Dense(self.embedding_dim, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        # L2 normalize the embeddings.
        outputs = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
        
        model = Model(inputs, outputs, name="mobilenetv2_embedding")
        return model

    def get_model(self):
        """
        Returns the underlying embedding model.
        """
        return self.model

class SiameseNetwork:
    def __init__(self, embedding_model):
        """
        Initializes the Siamese network.

        Args:
            embedding_model (EmbeddingModel): An instance of the embedding model.
        """
        self.embedding_model = embedding_model
        self.model = self.build_siamese_model()
    
    def euclidean_distance(self, vectors):
        """
        Computes the Euclidean distance between two embeddings.

        Args:
            vectors (list): A list containing two tensors.

        Returns:
            Tensor: Euclidean distance.
        """
        x, y = vectors
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, 1e-6))
    
    def build_siamese_model(self):
        """
        Builds the Siamese network model.

        Returns:
            tf.keras.Model: The Siamese network model.
        """
        input_a = Input(shape=self.embedding_model.input_shape)
        input_b = Input(shape=self.embedding_model.input_shape)
        
        processed_a = self.embedding_model.get_model()(input_a)
        processed_b = self.embedding_model.get_model()(input_b)
        
        distance = layers.Lambda(self.euclidean_distance)([processed_a, processed_b])
        model = Model([input_a, input_b], distance)
        return model

    @staticmethod
    def contrastive_loss(y_true, y_pred, margin=1):
        """
        Contrastive loss function.

        Args:
            y_true (Tensor): Ground-truth labels (1 for similar, 0 for dissimilar).
            y_pred (Tensor): Predicted Euclidean distances.
            margin (float): Margin for dissimilar pairs.

        Returns:
            Tensor: Loss value.
        """
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    def compile(self, optimizer='adam'):
        """
        Compiles the Siamese model.

        Args:
            optimizer (str or tf.keras.optimizers.Optimizer): Optimizer to use.
        """
        self.model.compile(optimizer=optimizer, loss=SiameseNetwork.contrastive_loss)
    
    def train(self, train_gen, validation_gen, steps_per_epoch, validation_steps, epochs=20, callbacks=None):
        """
        Trains the Siamese network.

        Args:
            train_gen: Training data generator.
            validation_gen: Validation data generator.
            steps_per_epoch (int): Steps per epoch.
            validation_steps (int): Validation steps.
            epochs (int): Number of epochs.
            callbacks (list): List of callbacks.

        Returns:
            History: Training history.
        """
        history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_gen,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        return history

if __name__ == '__main__':
    embedding_model = EmbeddingModel(input_shape=(224, 224, 3), embedding_dim=128)
    siamese_network = SiameseNetwork(embedding_model)
    siamese_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    siamese_network.model.summary()
