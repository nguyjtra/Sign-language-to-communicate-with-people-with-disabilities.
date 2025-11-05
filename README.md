# ASL Alphabet Recognition
This project is a computer vision system designed to automatically recognize and classify static hand gestures corresponding to the American Sign Language (ASL) alphabet. The system classifies each image into one of its alphabet categories.

The goal is to develop a tool that can help bridge the communication gap for the deaf and speech-impaired community by enabling potential real-time translation of sign language. This project serves as a proof-of-concept for assistive tools in classrooms, public spaces, and mobile apps.

## 📊 Dataset
Our dataset consists of **over 20,000 static images** used for training and testing.
* The data is organized into **labeled folders** that match each alphabet letter gesture.
* We believe this dataset is highly suitable for image classification tasks within computer vision.

This project will apply several key technologies, including:
* **Convolutional Neural Networks (CNN)**
* **Transfer Learning** using the powerful **VGG16** network
* **Background Removal** techniques to isolate the hand gestures

Our task is to build a system that can analyze an image of a hand gesture and predict the character it represents. For this project, we will be focusing on a sample of 5 letters: **V, L, E, F, and B**.

## 🔬 Methodology
We use a Convolutional Neural Network (CNN) architecture to classify the hand sign images. Our approach is structured as follows:

* **Baseline Model:** We start by implementing a standard shallow CNN (inspired by architectures like LeNet or simple VGG variants) to establish a baseline performance.

* **Improved Model:** We then explore and implement several enhancements to improve upon the baseline:
    * **Data Augmentation:** Applying transformations to the training data to increase model robustness and prevent overfitting.
    * **Transfer Learning:** Fine-tuning a pre-trained network (like VGG, ResNet, or MobileNet) to leverage features learned from broader, large-scale image datasets.
    * **Regularization:** Incorporating Dropout and Batch Normalization for more stable training and better generalization.
### 🛠️ Tech Stack
* PyTorch
* Keras
* TensorFlow

## 📈 Quantitative Evaluation
We evaluate and compare the models using standard metrics to demonstrate the effectiveness of our enhancements.
* **Test-set Accuracy:** To evaluate and compare the overall performance of the baseline and improved models.
* **Per-class Precision:** To provide a detailed assessment of how well the models perform across different categories (letters).
* **Confusion Matrix:** To visualize classification errors and better understand the strengths and weaknesses of each approach.

---

## 🚀 Getting Started

.......

## 💻 Usage

......
## 👥 Contributors
* **Nguyen Tran**- khoa2042002@gmail.com
* **Jacky Choi** - jzchoi@ucdavis.edu
* **Quynh Trinh** - qhtrinh@ucdavis.edu
