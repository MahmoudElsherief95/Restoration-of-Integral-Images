
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model_definition import denoising_unet_model  # Assuming model_definition.py is in the same directory
from utils import load_images_from_directory, create_masks_from_ground_truth, resize_images  # Importing utility functions

# Custom loss function
def masked_mse(y_true, y_pred):
    global masks
    y_true_masked = tf.multiply(tf.cast(y_true, tf.float32), tf.cast(masks, tf.float32))
    y_pred_masked = tf.multiply(tf.cast(y_pred, tf.float32), tf.cast(masks, tf.float32))
    return tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))

# Define the learning rate schedule
def lr_schedule(epoch, lr):
    if epoch > 0 and epoch % 10 == 0:
        return lr * 0.1
    return lr

# Load and preprocess data
input_images_dir = 'Integrals'
ground_truth_dir = 'GT'
input_images = load_images_from_directory(input_images_dir)
ground_truth_images = load_images_from_directory(ground_truth_dir)
input_images_resized = resize_images(input_images)
ground_truth_images_resized = resize_images(ground_truth_images)

# Create masks from resized ground truth images
masks = create_masks_from_ground_truth(ground_truth_images_resized)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(input_images_resized, ground_truth_images_resized, test_size=0.2, random_state=42)

# Initialize or load the model
checkpoint_path = 'denoising_model_checkpoint.h5'
if os.path.exists(checkpoint_path):
    denoising_model = load_model(checkpoint_path, custom_objects={'masked_mse': masked_mse})
else:
    denoising_model = denoising_unet_model()

# Compile the model
denoising_model.compile(optimizer=Adam(learning_rate=0.0001), loss=masked_mse, metrics=['mae'])

# Set up the callbacks
checkpoint = ModelCheckpoint("denoising_model_best.h5", save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the denoising model
history = denoising_model.fit(
    x=X_train, 
    y=y_train, 
    batch_size=1,  # Or any other batch size
    validation_data=(X_val, y_val),
    epochs=1,  # Or any other number of epochs
    callbacks=[checkpoint, lr_scheduler],
    verbose=1
)
