# CNN Image Classifier for CIFAR-10

From a beginner of machine learning. This repository implements a Convolutional Neural Network (CNN) model using PyTorch to classify images in the **CIFAR-10** dataset. The model consists of three convolutional layers and three fully connected layers for multi-class classification.

## 📌 Features

- Three convolutional layers with RELU activation and max pooling.
- Fully connected layers for classification (supports 10 classes in CIFAR-10).
- Designed for image classification tasks.

---

## 🛠️ Model Architecture

Input: 32x32x3 (CIFAR-10 images)

Layer 1: Conv2d(3 → 64) + ReLU + MaxPool(2x2) 

Layer 2: Conv2d(64 → 128) + ReLU + MaxPool(2x2) 

Layer 3: Conv2d(128 → 256) + ReLU + MaxPool(2x2)

Flatten: 256 * 4 * 4 → 128 (fc1) → 64 (fc2) → 10 (Output classes)

---

## 🚀 Installation

Ensure that you have **Python 3.x** and **PyTorch** installed. You can set up the environment with:

```bash
# Clone the repository
git clone https://github.com/xlssong/CIFAR_image_classififcation.git
cd CIFAR_image_classififcation

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

# Install dependencies
pip install torch torchvision
```

---
## 📜 code architecture
CIFAR_image_classififcation/

├── model.py          # CNN Model Definition

├── config.py        # configurations

├── data_utils.py    # load dataset

├── train.py         # Training Script

├── interference.py  # Interference Script 

└── README.md        # Project Documentation


📜 License

This project is licensed under the MIT License.

🤝 Contributing

Feel free to open issues and pull requests to improve the project!

📧 Contact

Author: Xiaoling Song

Email: songxiaoling31@gmail.com
