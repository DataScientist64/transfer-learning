# Image Classification with Transfer Learning (VGG16)

## Problem
Build an image classifier using deep learning that can generalise 
well on a limited dataset — applicable to retail product recognition, 
medical image classification, and manufacturing quality control.

## Approach
- Loaded pre-trained VGG16 model (trained on ImageNet) as the base
- Froze base layers to retain learned features
- Added custom fully-connected classification layers on top
- Fine-tuned on target dataset with data preprocessing and augmentation
- Evaluated on held-out test set

## Results
| Model              | Accuracy |
|--------------------|----------|
| Test Accuracy  | 87%      |

Transfer learning allowed strong performance despite a limited 
training dataset — demonstrating the practical value of pre-trained 
deep learning models.

## Tech Stack
Python · TensorFlow · Keras · NumPy · Matplotlib

## How to Run
1. Clone the repo
2. pip install -r requirements.txt
3. Open transfer_learning_vgg16.ipynb in Jupyter
