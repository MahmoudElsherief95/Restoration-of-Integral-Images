
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2  # Ensure OpenCV is installed: pip install opencv-python
from utils import load_test_images  # Assuming utils.py is in the same directory

# Path to the saved model
checkpoint_path = 'denoising_model_best.h5'

# Load the saved model
denoising_model = load_model(checkpoint_path, custom_objects={'masked_mse': masked_mse})  # Include masked_mse if it was used

# Load test images (modify the directory path as needed)
test_images_directory = 'path/to/test/images'
X_test = load_test_images(test_images_directory)

# Predict using the denoising model
predictions = denoising_model.predict(X_test)

# Function to resize and visualize the test images and their denoised versions
def visualize_predictions(test_images, predictions, num_samples=3):
    for i in range(num_samples):
        original = test_images[i].squeeze()
        denoised = predictions[i].squeeze()

        # Resize images for display
        original_resized = cv2.resize(original, (512, 512))
        denoised_resized = cv2.resize(denoised, (512, 512))

        plt.figure(figsize=(10, 5))

        # Display original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_resized, cmap='gray')
        plt.title('Original Image')

        # Display denoised image
        plt.subplot(1, 2, 2)
        plt.imshow(denoised_resized, cmap='gray')
        plt.title('Denoised Image')

        plt.show()

# Visualize the results
visualize_predictions(X_test, predictions)
