import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical 
from models import AE_w_predicter

# Load cifar10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)

model = AE_w_predicter(10)
model._compile()
model.train(train_images, train_labels, 256, 10)