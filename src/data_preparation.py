"""
data_preparation.py

This module generates image pairs for training a Siamese network. It creates positive pairs 
(from the same person) and negative pairs (from different persons) using processed images.
"""

import os
import random

class DataPairGenerator:
    def __init__(self, processed_dir):
        """
        Initializes the generator with processed images.

        Args:
            processed_dir (str): Directory containing processed images organized by person.
        """
        self.processed_dir = processed_dir
        self.data_dict = self.load_processed_data()

    def load_processed_data(self):
        """
        Loads image file paths from the processed directory.

        Returns:
            dict: Mapping from person names to a list of image file paths.
        """
        data = {}
        persons = os.listdir(self.processed_dir)
        for person in persons:
            person_dir = os.path.join(self.processed_dir, person)
            if os.path.isdir(person_dir):
                images = [
                    os.path.join(person_dir, f)
                    for f in os.listdir(person_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                if images:
                    data[person] = images
        return data

    def create_image_pairs(self, num_pairs_per_person=5):
        """
        Creates positive and negative image pairs.

        Args:
            num_pairs_per_person (int): Number of positive pairs per person.

        Returns:
            tuple: (pairs, labels)
                pairs: List of tuples (img_path1, img_path2).
                labels: List of labels (1 for positive, 0 for negative).
        """
        pairs = []
        labels = []
        persons = list(self.data_dict.keys())
        
        # Generate positive pairs.
        for person in persons:
            images = self.data_dict[person]
            if len(images) < 2:
                continue
            for _ in range(num_pairs_per_person):
                img1, img2 = random.sample(images, 2)
                pairs.append((img1, img2))
                labels.append(1)
        
        # Generate negative pairs (same number as positive pairs).
        num_positive = len(pairs)
        for _ in range(num_positive):
            person1, person2 = random.sample(persons, 2)
            img1 = random.choice(self.data_dict[person1])
            img2 = random.choice(self.data_dict[person2])
            pairs.append((img1, img2))
            labels.append(0)
        
        return pairs, labels

if __name__ == '__main__':
    processed_dir = '../data_processed'
    generator = DataPairGenerator(processed_dir)
    pairs, labels = generator.create_image_pairs(num_pairs_per_person=5)
    print(f"Generated {len(pairs)} pairs: {sum(labels)} positive pairs and {len(labels) - sum(labels)} negative pairs.")
