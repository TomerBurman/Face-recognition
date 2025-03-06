"""
main.py

Main entry point for training the Siamese network and extracting embeddings.
This script builds the model, trains it with callbacks, and then computes embeddings 
from the processed images.
"""

import tensorflow as tf
from trainer import Trainer
from siamese_model import EmbeddingModel
from embeddings_extractor import EmbeddingsExtractor

def main():
    processed_dir = '../data_processed'  # Directory containing processed images
    trainer = Trainer(processed_dir, batch_size=32, target_size=(224, 224), num_pairs_per_person=5)
    model = trainer.build_model()
    
    # Set up callbacks: ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('../models/siamese_best.h5', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    history = trainer.train_model(epochs=20, callbacks=callbacks)
    trainer.save_pairs_to_folders("../saved_pairs")
    print("Training complete.")

    # Extract embeddings after training using the trained embedding model.
    # embedding_model = trainer.siamese_network.embedding_model.get_model()
    # extractor = EmbeddingsExtractor(processed_dir, embedding_model)
    # embeddings = extractor.compute_embeddings()
    # extractor.save_embeddings(embeddings, '../models/embeddings.pkl')
    # print("Embeddings extracted and saved.")

if __name__ == '__main__':
    main()
