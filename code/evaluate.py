
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from utils import load_test_images  # Assuming utils.py is in the same directory

# Load the trained model
checkpoint_path = 'denoising_model_best.h5'
denoising_model = load_model(checkpoint_path)

# Load validation data (You might need to modify this part based on how you stored your validation data)
# X_val, y_val = ...

# Visualize training history (Assuming 'history' object is available)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize the denoising_model prediction on a validation sample
sample_index = 38  # You can change this index
prediction = denoising_model.predict(X_val[sample_index].reshape(1, 128, 128, 1))
ground_truth = y_val[sample_index].squeeze()

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(X_val[sample_index].squeeze(), cmap='gray')
plt.title('Input Image')

plt.subplot(1, 3, 2)
plt.imshow(ground_truth, cmap='gray')
plt.title('Ground Truth Image')

plt.subplot(1, 3, 3)
predicted_image = prediction.squeeze()
plt.imshow(predicted_image, cmap='gray')
plt.title('Denoising Model Prediction')

plt.show()

# Calculate and print the metrics for the validation sample
mse = mean_squared_error(ground_truth, predicted_image)
psnr = peak_signal_noise_ratio(ground_truth, predicted_image, data_range=predicted_image.max() - predicted_image.min())
ssim = structural_similarity(ground_truth, predicted_image)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr}")
print(f"Structural Similarity Index (SSIM): {ssim}")

# Calculate average metrics on validation set
total_mse, total_psnr, total_ssim = 0, 0, 0
num_images = len(X_val)

for i in range(num_images):
    prediction = denoising_model.predict(X_val[i].reshape(1, 128, 128, 1)).squeeze()
    ground_truth = y_val[i].squeeze()

    mse = mean_squared_error(ground_truth, prediction)
    psnr = peak_signal_noise_ratio(ground_truth, prediction, data_range=prediction.max() - prediction.min())
    ssim = structural_similarity(ground_truth, prediction)

    total_mse += mse
    total_psnr += psnr
    total_ssim += ssim

average_mse = total_mse / num_images
average_psnr = total_psnr / num_images
average_ssim = total_ssim / num_images

print(f"Average Mean Squared Error (MSE): {average_mse}")
print(f"Average Peak Signal-to-Noise Ratio (PSNR): {average_psnr}")
print(f"Average Structural Similarity Index (SSIM): {average_ssim}")
