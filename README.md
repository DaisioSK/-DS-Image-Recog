# Handwritten Digit Classification

This repository contains Jupyter Notebooks for training and evaluating handwritten digit classifiers using the MNIST and EMNIST datasets.

## Notebooks

### 1. MNIST Notebook (`mnist.ipynb`)
- Uses the MNIST dataset (0-9 digits).
- Implements a Convolutional Neural Network (CNN) for digit classification.
- Includes data preprocessing, model training, and evaluation.
- Example code snippet:
  ```python
  import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import numpy as np
import matplot
  ```

### 2. EMNIST Notebook (`EMNIST.ipynb`)
- Uses the EMNIST dataset (extended MNIST with letters and digits).
- Implements a deep learning model for classifying handwritten characters.
- Covers data augmentation and optimization techniques.
- Example code snippet:
  ```python
  import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing
from PIL import Image

  ```

## Features
- Preprocessing handwritten digit data (alphabet + number).
- Training a CNN-based model for digit classification.
- Predicting digit classes from user-provided images.

## Requirements
To run these notebooks, install the required dependencies:
```bash
pip install torch torchvision matplotlib numpy pandas
```

## Running the Notebooks
To execute the notebooks:
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Navigate to the desired notebook and run the cells.

## Results
- The models achieve high accuracy in handwritten digit recognition.
- The EMNIST model extends classification to alphabets and digits.

## Future Improvements
- Experimenting with different architectures like ResNet.
- Implementing transfer learning for improved accuracy.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DaisioSK/Handwriting-Recog.git
   cd handwriting-recognition