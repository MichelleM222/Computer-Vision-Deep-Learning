# Computer Vision & Deep Learning

### Image Classification Using CNNs, Transfer Learning, and Deep Learning Architectures

A Computer Vision project focused on image classification and object recognition using deep learning architectures, transfer learning, and real-time prediction with Python.

The project explores different neural network approaches to identify and classify images of playing cards, comparing custom CNN architectures with pre-trained models.

## Project Overview

This project explores the application of Artificial Intelligence and Computer Vision techniques for image classification.

Multiple deep learning architectures were implemented and evaluated, including custom Convolutional Neural Networks (CNNs), transfer learning models, and transformer-based architectures.

The project also includes a real-time prediction component using a webcam and computer vision techniques.

## Objectives

- Develop image classification models using Deep Learning.
- Compare custom CNN architectures with pre-trained models.
- Apply Transfer Learning and Fine-Tuning techniques.
- Explore Transformer-based architectures for computer vision.
- Implement real-time image recognition using a webcam.
- Analyze model predictions and classification performance.

## Technologies Used

- Python
- TensorFlow / Keras
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Computer Vision
- Deep Learning
- Transfer Learning

## Models Implemented

### Custom CNN
Development of a custom convolutional neural network architecture for image classification.

### Transfer Learning

Pre-trained architectures explored:

- MobileNetV2
- ResNet50
- VGG16
- Swin Transformer

### Fine-Tuning

Fine-tuning experiments were conducted to adapt pre-trained models to the image classification task.

## Project Structure

```text
Computer-Vision-Deep-Learning/
│
├── images/
├── models/
│
├── captura_objeto.jpg
├── captura_cartas.py
├── CNN_arquitectura_propia.py
├── create_dataset.py
├── MobileNetV2_transfer_learning.py
├── MobileNetV2_fine_tuning.py
├── ResNet50_transfer_learning.py
├── ResNet50_fine_tuning.py
├── VGG16_transfer_learning.py
├── VGG16_fine_tuning.py
├── SwinTiny_transfer_learning.py
├── SwinTiny_fine_tuning.py
├── ModeloCNNPreentrenadoFactory.py
├── ModeloTransformerTimmFactory.py
├── 9_prediccion_tiempo_real.py
├── requirements.txt
└── README.md
```

## Real-Time Prediction

The project includes a real-time prediction script that uses a webcam to capture images and perform classification using trained deep learning models.

The implementation integrates:

- Webcam image capture.
- Image preprocessing.
- Deep learning inference.
- Classification output.

## Installation

Clone the repository:

```bash
git clone https://github.com/MichelleM22/Computer-Vision-Deep-Learning.git
```

Navigate to the project directory:

```bash
cd Computer-Vision-Deep-Learning
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Execution

To run the real-time prediction module:

```bash
python 9_prediccion_tiempo_real.py
```

Other scripts can be executed individually to explore the different model architectures and training approaches.

## Applications

Computer Vision techniques can be applied to:

- Automated image recognition.
- Object classification.
- Visual inspection systems.
- Intelligent image processing.
- Real-time recognition applications.

## Limitations

This project represents an academic exploration of
