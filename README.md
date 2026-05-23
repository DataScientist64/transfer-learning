# Image Classification with Transfer Learning (VGG16)

## Problem
Build an image classifier using deep learning that generalises well 
on a limited dataset — applicable to retail product recognition, 
medical image classification, and manufacturing quality control.

## Approach
- Loaded pre-trained VGG16 model (trained on ImageNet) as the base
- **Stage 1 — Feature Extraction:** Froze all VGG16 layers, trained 
  only the custom classification head
- **Stage 2 — Fine-Tuning:** Unfroze top layers and retrained 
  end-to-end to boost performance
- Evaluated both stages using precision, recall, and F1-score

## Results

### Stage 1 — Feature Extraction (frozen VGG16 base)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| O     | 0.77      | 0.88   | 0.82     |
| R     | 0.86      | 0.74   | 0.80     |
| **Overall Accuracy** | | | **81%** |

### Stage 2 — Fine-Tuned Model
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| O     | 0.86      | 0.88   | 0.87     |
| R     | 0.88      | 0.86   | 0.87     |
| **Overall Accuracy** | | | **87%** |

Fine-tuning improved accuracy from 81% to 87% — demonstrating the 
value of unfreezing deeper layers on domain-specific data.

## Tech Stack
Python · TensorFlow · Keras · NumPy · Matplotlib · Scikit-Learn

## How to Run
1. Clone the repo
2. pip install -r requirements.txt
3. Open transferlearning.ipynb in Jupyter
