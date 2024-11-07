Cat vs. Dog Image Classification

This project uses a Convolutional Neural Network (CNN) built on TensorFlow and pre-trained MobileNetV2 model from TensorFlow Hub to classify images as either cats or dogs.

Table of Contents
1. Project Overview
2. Dataset
3. Requirements
4. Code Explanation

Project Overview

The goal of this project is to build a model that classifies images as either cats or dogs. It uses a MobileNetV2 pre-trained model as the feature extractor and trains a custom classifier on top of it.

Dataset

This project uses a dataset of labeled cat and dog images, where images are labeled in their file names (e.g., `cat.4352.jpg` or `dog.8298.jpg`).

- The images are initially resized to 224x224 pixels.
- Each image is labeled: **1** for dog and **0** for cat.

Requirements

To run this project, the following libraries are required:
- TensorFlow
- TensorFlow Hub
- OpenCV
- NumPy
- PIL
- Matplotlib
- Sklearn
- Glob

Install dependencies with:
pip install tensorflow tensorflow-hub opencv-python-headless numpy pillow matplotlib scikit-learn


Code Explanation

-  Image Preprocessing :  Resizes images to 224x224 pixels and converts them to RGB.
-  Labeling : Labels the images as 1 (dog) or 0 (cat) based on file names.
-  Data Loading : Loads images into a NumPy array.
-  Model Definition : Uses the MobileNetV2 pre-trained model as the base and adds a dense layer for classification.
-  Model Training : Compiles and trains the model with sparse categorical cross-entropy.
-  Prediction : Takes a user-input image path, preprocesses it, and predicts if the image is a cat or a dog.


