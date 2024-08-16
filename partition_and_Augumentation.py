#########################################################################################################################################################
#                                              pt2 partition and augment data
"""
BEST PRACTICE: the model is trained ("taught") from the training partition, but at the same time we use the validation partition, to inform 
how we build the neural network. EX. If we see the model is performing well on the training par, but not on the validation set->
We might need to try some regularization, or change our nn architecture.

^^ regarding our test set-> we dont touch it until the very end. We use it to see how well our model is performing
"""
# We're going to apply image augmentation to build our data set-> randomly crop/adjust our images

#                                              2. Review Dataset and Build Image Loading Function

#########################################################################################################################################################
#                                              2.1 Import TF and Depencies (Deps)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # setting the logging level to only show errors

import tensorflow as tf
import cv2
import json # needed to load our json files into our python environment
import numpy as np # helps us with our data pre-processing
from matplotlib import pyplot as plt # used to visualize our images

#########################################################################################################################################################
#                                              2.2 Limit GPU Memory Growth
# Good practice to avoid "Out of Memory" (OOM) errors by setting GPU Memory Consumption Growth. 
gpus = tf.config.experimental.list_physical_devices('GPU') # grabbing all the gpus in our system
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) # setting memory growth to true

tf.config.list_physical_devices('GPU') # checking to see if our memory growth is set to true

#########################################################################################################################################################
#                                              2.3 Load Image into TF Data Pipeline

                                    # SUPER IMPORTANT THAT YOU USE THE CORRECT PATH
images = tf.data.Dataset.list_files('data\\images\\*.jpg', shuffle=False) # loading our images into our data pipeline.. '*' is a "wildcard search"
# *jpg looks for anything ending in .jpg in terms of its file path, in the tensorflow data pipeline

#uncomment the following line to find the full filepath of an image:
#images.as_numpy_iterator().next()

def load_image(x):
    byte_img = tf.io.read_file(x) # takes in file path, return byte-encoded image
    img  = tf.io.decode_jpeg(byte_img) # decoding the image file
    return img
# using the above function below, we are going to pass in the full file path:
images = images.map(load_image) # applies our "load_image" map function to each element in our data set
images.as_numpy_iterator().next() # iterate through our images, and grab the first image

"""
At this point we now have our images loaded into our tensorflow data pipeline.
"type(images)" will return a "tensorflow.python.data.ops.dataset_ops.MapDataset"

"""

#########################################################################################################################################################
#                                              2.4 View Raw Images with Matplotlib

image_generator = images.batch(4).as_numpy_iterator() # batch our images into groups of 4. Rather than return 1 iamge, 
# its going to return the number of images inside our batch. This is also going to allow us to iterate through our images

plot_images = image_generator.next() # grab the next batch of images ### in tensorflow we can run this line multiple times to get the next batch of images


# loop through and view our images, using matplotlib's subplots class--> also this presents our images in order, to make them random, remove the shuffle from line 35
fig, ax = plt.subplots(ncols=4, figsize=(20,20)) # creating a figure with 4 columns, and a size of 20x5
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show() 