# importing stuff 
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# prints version
print (tf.__version__)

# importing mnist dataset 
mnist = tf.keras.datasets.mnist

# training data and testing data set equal to dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#pixel values of images [0,255], divide by 255.0 to convert to float and scale from 0->1
x_train, x_test = x_train / 255.0, x_test / 255.0

# builds the actual model
# sequential is used when each layer has a single input tesnsor and a single output tensor 
# layers = function w/ known mathematical structure, reusable and trainable variables
# Flatten, Dense, and Dropout layers are used in this model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# each example model returns vector of logits scores
predictions = model(x_train[:1]).numpy()
predictions

# nn.softmax converts logits to probabilities
tf.nn.softmax(predictions).numpy()

# loss function takes ground truth values, returns scalar loss = to negative log probabilitiy of true class, loss = 0 if model is certain it is the correct class
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer = 'adam',
                loss = loss_fn,
                metrics = ['accuracy'])

# model.fit adjusts parameters, minimizing loss
model.fit(x_train, y_train, epochs = 5)

# model.evaluate checks models performance on a validation/test set
model.evaluate(x_test, y_test, verbose = 2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
probability_model(x_test[:5])