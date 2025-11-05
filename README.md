
##Project Proposal
Group Members: Nguyen Tran, Jacky Choi, Quynh Trinh
Emails: qhtrinh@ucdavis.edu, jzchoi@ucdavis.edu, 
Problem Statement
Our goal of this project is to develop a computer vision system that automatically recognizes hand gestures corresponding to American Sign Language (ASL) alphabets from static images. The system will classify each image into one of the sign language alphabet categories, helping those who are hard of hearing or speech-impaired can communicate easily with people.
Motivation
The deaf and speech-impaired communities use sign language everyday as a communication tool to connect with people. However, most people do not understand sign language, which creates a barrier to involvement, education, and daily interactions. Automating sign language recognition could enable instant text or voice translation and foster greater accessibility, social equity, and convenience. This project could serve as a proof of concept for assistive tools used in classrooms, public spaces, and mobile apps, helping break communication barriers in multilingual contexts.
Data Source & Description
Our dataset consists of over 20,000 images which can be used for training or testing purposes.  It also consists of:
Labeled folders that match to each alphabet letter gesture.
Static images of hand gestures representing ASL alphabets.
We believe it will be suitable for image classification tasks within computer vision.
Methodology
We will use a convolutional neural network (CNN) architecture to classify hand sign images into their respective alphabet categories.
Baseline: starting with standard shallow CNN, inspired by architectures like LeNet or simple VGG variants.
Main method and explore improvements:
Data augmentation to increase model robustness.
Transfer learning: fine-tuning a pre-trained network to leverage features learned from broader image datasets.
Incorporating dropout and batch normalization for better generalization.
Standard packages for implementation:
Pytorch
Keras
TensorFlow
After that, we will enhance the model’s performance to the simple CNN baseline, evaluating with standard metrics, such as accuracy and confusion matrix.
Quantitative Evaluation
After implementing, we will:
Use test-set accuracy to evaluate and compare both baseline and improved models
Report other metrics like per-class precision to provide a detailed assessment of how well the models perform across different categories
Might also include confusion matrix to visualize classification errors and better understand the strengths and weaknesses of each approach
The results will show the performance gains achieved by the improved method compared to the baseline, demonstrating the effectiveness of the enhancements made.
