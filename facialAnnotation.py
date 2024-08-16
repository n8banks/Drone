"""
label image library is ass. Use labelme
Label me permits more than boundary box annotation, permits for keypoint annotation

//Data augmentation: use albumentations--> library permits us to take our dataset and apply:
    -random cropping
    -brightness
    -flips
    -gamma shifts and rgb shifts
        + This all gets our base dataset and increases its data by 30x
            + if we had 100 images, this would give us 3000, which should effectively be enough to build up our model
This library not only augments your images, it annoates them at the same time
    -ex: if you had boundary boxes and had to crop on your dataset, this library keeps our boundary box coordinates in place


An object detection model is really a
    -classification model to classify what our object is
    -regression model, trying to estimate our coordinates of our boundary boxes. 
        + All thats needed to create a box is a set of opposing upper/lower side coordinates, then from there create the other 2
Before going to train our deep learning model, we need to define our losses
For our CLASSIFICATION COMPONENT: a binary crossentropy loss, (a common practice for classification models)
    -face classification loss
    -localization loss-->estimate how far off our predictions were actually drawing our box. ensures we our as close as possible to representing our object
        "we're going to do it slightly differenttly"
            taking x coordinate and comparing it to the predicted 'x̂' coordinate, and our y coordinate to the predicted 'ŷ' coordinate
            ∑(x-x̂)² + ∑(y-ŷ)²   ➔➔➔ how far off our prediction was for the top corner coordinate component
            + 
            ∑(w-ŵ)² + ∑(h-ĥ)²   ➔➔➔ evaluates how far off our width 'w' and height 'h' predictions were
        We're comparing our true value to our predicted value, we're doing this for our width and height

        
Once we've defined our loses, we're going to use the keras functional api to build our model.
We're using a vgg16 model (classification model for images)-> its been pre-trained on a ton of data already, we can use it in our model,
then add our final 2 layers: our classification model and our regression model, to be able to give us our binding boxes
This overall model will give us 5 different values(outputs) we can use to do detections
    1st set: a range between 0 or 1, representing whether or not a face (object) has been detected
    2nd set: 4 vals- [x1, y1, x2, y2] the coords for our box
"""

"""
This is not face detection, this is end to end object detection pipeline
Below we are going to do bounding box detection 
"""
#########################################################################################################################################################
#                                                   1. Setup and Get Data (as well as a little annotation)
#                                                   1.1 Install dependencies and setup
#   pip install labelme tensorflow tensorflow-gpu opencv-python matplotlib albumentations


import os #makes things a lot easier to navigate through different file paths, combine them, and create new ones
import time # when we collect our images, we want a little bit of time to move around

# allows you to create a uniform unique identifier (unique file names for our iamges, instead of going "image1" etc)
import uuid # ex: uuid.uuid1() ➔➔➔ 4b5f3b3e-7b3b-11ec-8d3d-080027c0b7b1. 
import cv2 # that shit

#########################################################################################################################################################
#                                                   1.2 Collect Images Using OpenCV 

# from here below--> to "cv2.destroyAllWindows()", this takes in raw camera input
IMAGES_PATH = os.path.join('data','images') # our images are going into folder-"data", then into folder-"images"
number_images = 30 #gonna start at 30, move the camera a lil bit, then collect another 30, repeat a 3rd time--> we want like 100

# [CHATGPT CODE 🤢🤮]-->  modify existing code to change output directory: captured images are saved in data->labels directory instead of data->images
# if we dont change the annotation folder/output fodler,  it'll save the annotation to a different folder, we just need to move it. 
""" 
    #TODO --> go into json file for each image, to check their directory. It should be face in this case
    "shapes": [
        {
        "label": "face",
        ...
"""
    #TODO --> it'll save some time if we auto save our annotations, so we dont have to hit save every time. 

# Create the directory if it doesn't exist
os.makedirs(IMAGES_PATH, exist_ok=True)

cap = cv2.VideoCapture(0) #establishing video connection with camera, passing in our camera#

for imgnum in range(number_images): # looking for our range of images
    print('Collecting image {}'.format(imgnum)) #print out "Collecting Image" and which image number we're at
    ret, frame = cap.read() #read from capture device, returns val for success/failure as well as the frame itself
    if not ret:
        print("Failed to capture image")
        break
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg') # defines a unique file name for the file we'll be passing through
    cv2.imwrite(imgname, frame) #writing out the above unique file name
    cv2.imshow('frame', frame) #also display it on screen

    # SLEEPING FOR (HALF A SECOND) between each frame, giving us time to move around/move our heads around/in+out of frame.
        # this will look extremely laggy but its deliberate, produces our first 30 images (900), and places them into our data->images folder
    # TIP: we also want to give the camera images of us outside the screen so it can have asome negative samples
        # TIPcont: next 30 iamges, we can move a little further away from the camera. Then next 30 a little closer, cover our face a couple times, etc
            # get creative^
        # TIPcont: if we have images already, we can just annotate them as well
    time.sleep(0.5) 

# standard cv2 break function--> "q" to break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()

# Annotate Images with LabelMe
#########################################################################################################################################################
#                                               1.3 Annotate Images with LabelMe

# In cmd, type in "labelme" Opens up the labelme app, allows us to annotate our images
#TODO how are we supposed to "hit edit/create rectangle/label what type of class this is. Im just gonna name it "face"--> to make our crosshair in jupyter, without jupyter
    #in jupyter, if he hits 'a' (left) or 'd' (right) on his keyboard (wasd) it allows him to loop through each of his images. he labels each image of his face as "face"
    #  blurry images work fine, corners of your face work fine as well. for images without a face in the frame, do nothing.
    #  for some reason at 28:50, his iamge is saved to folder 'd', not sure if he just messed up his saving, or its code related 

"""
{
  "version": "5.5.0",
  "flags": {},
  "shapes": [
    {
      "label": "face",
      "points": [
        [
          286.3461538461538,                                        <<<<<< 
          135.19230769230768                                             ^
        ],                                                               ^
        [                                                                ^
          457.5,                                                         ^
          333.26923076923083                                             ^
        ]                                                                ^
      ],                                                                 ^
      "group_id": null,                                                  ^
      "description": "",                                                 ^
      "shape_type": "rectangle",                                         ^
      "flags": {},                                                       ^
      "mask": null                                                       ^
    }                                                                    ^
  ],                                                                     ^
  "imagePath": "..\\images\\0aca0dd2-5611-11ef-bd46-14d424c94eb2.jpg",   ^
-----------------------------------------                                ^
ABOVE IS WHAT AN APPROPRIATE ANNOTATION LOOKS LIKE                       ^
          286.3461538461538, ---> width ----------------------------------
          135.19230769230768 ---> height
        The first set of coords is the top left corner of the box, the second set is the bottom right corner of the box
"""




# DEEP LEARNING FOR BEGINNERS TUTORIAL-> Nicholas Renotte youtube. If you want to use tensorflow through your GPU, you need to install tensorflow-gpu
