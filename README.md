Handwritten Digit Recognizer
This project offers a simple and interactive application to recognize handwritten digits using a pre-trained Convolutional Neural Network (CNN). Built with Python's tkinter for the user interface and TensorFlow/Keras for the machine learning backend, it lets you load an image of a digit and instantly get a prediction with confidence scores.

Features
Intuitive GUI: Easy-to-use tkinter interface.
Robust Preprocessing: Automatically converts various image inputs into the 28x28 grayscale format required by the model, including color inversion, binarization, and centering.
MNIST Model: Uses a pre-trained CNN for accurate digit classification.
Instant Predictions: Get real-time digit predictions and confidence levels.
Probability Breakdown: See the probability for each digit (0-9).

Getting Started
Prerequisites

Ensure you have Python 3.8+ and the following libraries installed:

pip install Pillow opencv-python numpy tensorflow

This project requires a pre-trained Keras model named mnist_model.h5. You can generate this file by running the training script (see below) or place an existing one in the project's root directory.

Click "Load Image", select a handwritten digit image, and see the prediction!
