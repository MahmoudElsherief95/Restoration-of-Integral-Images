
Project Overview:
------------------

├── code
│   ├── __init__.py
│   ├── model_definition.py
│   ├── preprocessing.py
│ 
│   ├── train.py
│   ├── evaluate.py
│   ├── test.py
│   └── utils.py
├── input_images
│   ├── integral_001.png
│   ├── integral_002.png
│   ├── ...
│   └── integral_XXX.png
├── ground_truth
│   ├── gt_001.png
│   ├── gt_002.png
│   ├── ...
│   └── gt_XXX.png
├── output
│   └── <model weights and results>
└── Readme.txt

This project focuses on image restoration for integral images using a deep learning approach with a U-Net model.
The U-Net is adapted for grayscale integral images, aiming at improving image quality by reducing noise and enhancing details.



U-Net Model for Image Restoration:
-----------------------------------

The U-Net architecture, known for its effectiveness in various image processing tasks, is here employed for 
the restoration of grayscale integral images.

It consists of convolutional layers for up-convolutional layers for image reconstruction, making it 
ideal for dealing with the nuances of grayscale integral images.

The model is designed to learn complex patterns in integral images, enabling effective restoration
with a focus on maintaining image integrity.



Directory Structure:
--------------------

'/code'

  /utils: Utility functions for data loading and preprocessing.
  Model_definition.py: Model definition script.
  train.py: Script for training the model.
  evaluate.py: Script for evaluating the model.
  test.py: Script for testing the model on new data.

/output: Directory for saved model weights and evaluation results.


Getting Started:
----------------

Preparation:

  - Install Python 3.x, TensorFlow, OpenCV, and other required libraries.
  - Clone or download this repository.

Data Setup:

  - Place your integral images and corresponding ground truth data in the respective directories.
  - Update paths in utility scripts as needed.

Training the Model:

  - Run python train.py in the /code directory to train the model.
  - Model weights will be saved in the /output directory.

Evaluating the Model:

  - Use python evaluate.py to assess the model's performance on the validation set.
  - Evaluation results, including metrics and visualizations, will be saved in the /output directory.

Testing the Model:

  - For testing with new integral images, place them in a specified test directory.
  - Run python test.py to apply the model and save the restored images.

Model Weights:

  - For large models, consider hosting the weights on a cloud platform and provide a download link.


Additional Notes:
-----------------

Make sure to set the correct paths in the scripts.
Adjust training parameters in train.py as needed.
The model architecture details are available in /code/model_definition.py.


Project Link (whole folders and files) :
----------------

https://drive.google.com/drive/folders/14R2BKU7IuPpJ5QeVCN3KPRNSQz-BMxUv?usp=share_link