
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from concurrent.futures import ThreadPoolExecutor
import random
import glob

# Function to load and resize an image
def load_and_resize_image(file, directory, target_size=(128, 128)):
    img_path = os.path.join(directory, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    return img_to_array(img) / 255.0  # Normalize pixel values

# Function to display random images with paths
def display_random_images_with_paths(input_images, output_images, input_files, output_files, num_samples=5):
    indices = random.sample(range(len(input_images)), num_samples)

    plt.figure(figsize=(15, 5))

    for i, index in enumerate(indices, 1):
        # Display Input Image
        plt.subplot(2, num_samples, i)
        plt.imshow(input_images[index].squeeze(), cmap='gray')
        plt.title(f'Input Image (Index: {index})\nPath: {input_files[index]}')
        plt.axis('off')

        # Display Output Image
        plt.subplot(2, num_samples, i + num_samples)
        plt.imshow(output_images[index].squeeze(), cmap='gray')
        plt.title(f'Output Image (Index: {index})\nPath: {output_files[index]}')
        plt.axis('off')

    plt.show()

def load_images_from_directory(directory, color_mode=cv2.IMREAD_GRAYSCALE):
    image_paths = glob.glob(os.path.join(directory, '*.png'))
    images = [cv2.imread(path, color_mode) for path in image_paths]
    return np.array(images)

def create_masks_from_ground_truth(ground_truth_images):
    masks = []
    for gt_image in ground_truth_images:
        _, mask = cv2.threshold(gt_image, 127, 1, cv2.THRESH_BINARY)
        mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)
    return np.array(masks)

def resize_images(images, size=(128, 128)):
    resized_images = [cv2.resize(img, size, interpolation=cv2.INTER_AREA) for img in images]
    return np.array(resized_images)

def load_test_images(directory, target_size=(128, 128)):
    images = []
    filenames = os.listdir(directory)
    for filename in filenames:
        img_path = os.path.join(directory, filename)
        img = load_img(img_path, color_mode='grayscale', target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1] if that's how the model was trained
        images.append(img_array)
    return np.array(images)
