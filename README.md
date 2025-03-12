# LEGO Lab

LEGO Lab is a computer vision project designed to process and analyze images of LEGO pieces. The repository includes code for image preprocessing (with mean shift filtering), dataset splitting, XML annotation processing, and training a ResNet18 model for LEGO piece count regression. Additionally, it offers interactive evaluation and further training options.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Image Preprocessing](#image-preprocessing)
- [Dataset Splitting](#dataset-splitting)
- [XML Annotation Processing](#xml-annotation-processing)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Gradio Interface](#gradio-interface)
- [Git Setup & .gitignore](#git-setup--gitignore)
- [License](#license)

## Overview

LEGO Lab is a modular project that processes a large dataset of LEGO images, extracts and caches annotations from XML files, splits the dataset into training, validation, and test subsets, and trains a dynamically quantized ResNet18 model to predict the number of LEGO pieces in an image. The project also provides an interactive terminal interface for further training and evaluation and a Gradio-based web interface for testing the model.

## Features

- **Environment Check:** Reports versions for OpenCV, PyTorch, and CUDA availability.
- **Image Preprocessing:** Applies mean shift filtering in HSV color space after resizing images.
- **Dataset Splitting:** Splits processed images into training, validation, and test sets using an 80/20 (and further 80/20) split.
- **XML Annotation Processing:** Parses XML files to count LEGO pieces per image and caches the results.
- **Model Training & Evaluation:** Uses a pretrained ResNet18 (with a modified final layer) for regression. The model is dynamically quantized for efficient CPU inference.
- **Interactive Terminal:** Users can continue training the model or evaluate its performance interactively.
- **Gradio Interface:** A web interface for uploading images and testing model predictions.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/KalharPandya/lego-lab.git
   cd lego-lab
