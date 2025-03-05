import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import math
import random
import tensorflow as tf

class DataPreprocessor:
    def __init__(self, input_dir, output_dir, target_size=(224, 224),
                 confidence_threshold=0.90, max_angle=45, blur_threshold=100,
                 augment=True, num_augmented=10):
        """
        Initializes the preprocessor.

        Args:
            input_dir (str): Directory of raw images, structured as one folder per person.
            output_dir (str): Directory to save processed images.
            target_size (tuple): Output image size.
            confidence_threshold (float): Minimum face detection confidence.
            max_angle (float): Maximum allowed rotation angle in degrees.
            blur_threshold (float): Threshold for blur detection (not used since blurred images are discarded).
            augment (bool): Whether to apply data augmentation.
            num_augmented (int): Number of augmented copies per face.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.max_angle = max_angle
        self.blur_threshold = blur_threshold
        self.augment = augment
        self.num_augmented = num_augmented
        self.detector = MTCNN()
        os.makedirs(self.output_dir, exist_ok=True)

    def align_face(self, image, keypoints):
        """
        Aligns the face image using the positions of the left and right eyes.
        Rotates the face so that the eyes are horizontal.

        Args:
            image (numpy.ndarray): Cropped face image in RGB.
            keypoints (dict): Contains 'left_eye' and 'right_eye'.

        Returns:
            numpy.ndarray or None: The aligned face image, or None if alignment fails.
        """
        left_eye = keypoints.get("left_eye")
        right_eye = keypoints.get("right_eye")
        if left_eye is None or right_eye is None:
            print("[Info] Skipping face due to missing left or right eye keypoints.")
            return None

        delta_y = right_eye[1] - left_eye[1]
        delta_x = right_eye[0] - left_eye[0]
        angle = math.degrees(math.atan2(delta_y, delta_x))
        if abs(angle) > self.max_angle:
            print(f"[Info] Skipping face due to extreme rotation: {angle:.2f}°")
            return None

        eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                       (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                 flags=cv2.INTER_CUBIC)
        return aligned

    def data_aug(self, face):
        """
        Applies data augmentation to the face image:
        - Random brightness adjustment.
        - Random contrast adjustment.
        - Random horizontal flip.
        - Random JPEG quality adjustment.
        - Random saturation adjustment.
        
        Args:
            face (numpy.ndarray): Input face image in RGB (float32, [0,255]).
            
        Returns:
            list: A list of augmented images (NumPy arrays).
        """
        # Convert face to TensorFlow tensor.
        face_tensor = tf.convert_to_tensor(face, dtype=tf.float32)
        augmented_list = []
        for _ in range(self.num_augmented):
            aug_img = face_tensor  # start from original face
            
            # Generate random seeds
            seed1 = (np.random.randint(10000), np.random.randint(10000))
            seed2 = (np.random.randint(10000), np.random.randint(10000))
            seed3 = (np.random.randint(10000), np.random.randint(10000))
            seed4 = (np.random.randint(10000), np.random.randint(10000))
            seed5 = (np.random.randint(10000), np.random.randint(10000))
            
            aug_img = tf.image.stateless_random_brightness(aug_img, max_delta=0.02, seed=seed1)
            aug_img = tf.image.stateless_random_contrast(aug_img, lower=0.6, upper=1.0, seed=seed2)
            aug_img = tf.image.stateless_random_flip_left_right(aug_img, seed=seed3)
            
            # Convert to uint8 before JPEG quality adjustment
            aug_img_uint8 = tf.cast(tf.clip_by_value(aug_img, 0, 255), tf.uint8)
            aug_img_uint8 = tf.image.stateless_random_jpeg_quality(
                aug_img_uint8, min_jpeg_quality=90, max_jpeg_quality=100, seed=seed4
            )
            # Convert back to float32
            aug_img = tf.cast(aug_img_uint8, tf.float32)
            
            aug_img = tf.image.stateless_random_saturation(aug_img, lower=0.9, upper=1.0, seed=seed5)
            
            augmented_list.append(aug_img.numpy())
        return augmented_list


    def process_image(self, image_path):
        """
        Processes a single image: detects, aligns, applies augmentation,
        and resizes the face.
        
        Returns a list of processed face images.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"[Warning] Could not read image: {image_path}")
            return []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(image_rgb)
        if len(results) == 0:
            print(f"[Info] No face detected in image: {image_path}")
            return []

        processed_faces = []
        for result in results:
            if result['confidence'] < self.confidence_threshold:
                print(f"[Info] Skipping face in {image_path} due to low confidence: {result['confidence']:.2f}")
                continue

            x, y, width, height = result['box']
            x, y = max(0, x), max(0, y)
            face = image_rgb[y:y+height, x:x+width]

            if "keypoints" in result:
                aligned_face = self.align_face(face, result["keypoints"])
                if aligned_face is None:
                    print(f"[Info] Skipping face in {image_path} due to alignment/angle issues.")
                    continue
                face = aligned_face

            # Generate augmented copies.
            if self.augment:
                augmented_faces = self.data_aug(face)
            else:
                augmented_faces = [face]

            # Resize each augmented face to target size.
            for aug_face in augmented_faces:
                face_resized = cv2.resize(aug_face, self.target_size)
                face_float32 = face_resized.astype('float32')
                processed_faces.append(face_float32)

        return processed_faces

    def process_dataset(self):
        """
        Processes the entire dataset.
        For each image, if exactly one face is detected, it generates augmented copies and saves them.
        Otherwise, it discards the image.
        """
        persons = os.listdir(self.input_dir)
        for person in persons:
            person_input_dir = os.path.join(self.input_dir, person)
            if not os.path.isdir(person_input_dir):
                continue

            person_output_dir = os.path.join(self.output_dir, person)
            os.makedirs(person_output_dir, exist_ok=True)

            for image_name in os.listdir(person_input_dir):
                image_path = os.path.join(person_input_dir, image_name)
                faces = self.process_image(image_path)
                if faces:
                    # Expect exactly num_augmented copies for a valid face.
                    if len(faces) == self.num_augmented:
                        for i, face in enumerate(faces):
                            face_bgr = cv2.cvtColor(face.astype('uint8'), cv2.COLOR_RGB2BGR)
                            base, ext = os.path.splitext(image_name)
                            output_filename = f"{base}_aug_{i}{ext}"
                            output_path = os.path.join(person_output_dir, output_filename)
                            cv2.imwrite(output_path, face_bgr)
                            print(f"[Success] Processed and saved: {output_path}")
                    else:
                        print(f"[Info] Discarding image {image_path} because it produced {len(faces)} faces (expected {self.num_augmented}).")
                else:
                    print(f"[Info] Skipping image: {image_path}")

    def is_blurry(self, image):
        """
        Checks if an image is blurry by computing the Laplacian variance.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        # Convert gray image to uint8 if it's not already.
        if gray.dtype != 'uint8':
            gray = gray.astype('uint8')
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance < self.blur_threshold


if __name__ == '__main__':
    preprocessor = DataPreprocessor(
        input_dir='../lfw-deepfunneled/',
        output_dir='../data_processed',
        target_size=(224, 224),
        confidence_threshold=0.90,
        max_angle=45,
        blur_threshold=100,
        augment=True,
        num_augmented=10
    )
    preprocessor.process_dataset()
