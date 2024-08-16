import cv2

"""
Bonus
- learn to use ord(' ') from below ex:
    Could variables represeenting hand-gestures be represented by ord('q')?
    As in like ord('whatever the hand gesture is')?
    if cv2.waitKey(1) & 0xFF == ord('q'): break



"""




###################################################################################################
"""BASIC IMAGE OPERATIONS
-image should be in same directory as file
-cv2.imread() function is used to read an image from a file.
    1, 0 , -1 as flags to determine how the image should be read
        1: LOADS color image
        0: LOADS grayscale image
        -1: LOADS unchanged image (with "alpha channels" // wtf is that)
"""
img = cv2.imread('image_67199233.JPG', -1)
# img = cv2.imread('WRONG NAME', -1)
# print(img)
# This will return None as the image is not found
###################################################################################################
"""
-cv2.imshow() function is used to display an image in a window.
cv2.imshow('any name here, name of window for image', img)
    -cv2.waitKey(0) will display the window infinitely until any key is pressed.
        Any variable can be used in place of 0, to close the window after that many milliseconds.
        (5000ms = 5sec)
        Advised to do cv2.waitKey(0) & 0xFF on 64 bit machines
    -cv2.destroyAllWindows() will destroy all the windows we created.
    -cv2.destroyWindow('name of window') will destroy a specific window.
"""
cv2.imshow('image', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
###################################################################################################
"""
cv2.imwrite() function is used to save an image to any file.
cv2.imwrite('name you want to give to image.png', image you want to save)
    function also returns true if image is saved successfully, false otherwise.
"""

# capture any value of your waitKey() referenced in line 20, 27 by setting it 
# to a variable and passing it to the waitKey() function at line 42

img2 = cv2.imread('someImage.jpg', 1)
cv2.imshow('fuckName', img2)
k = cv2.waitKey(0)

if k == 27: #when pressing esc, this function is called
    cv2.destroyAllWindows()
elif k == ord('s'): #if someone wants to use the 's' key
    cv2.imwrite('fuckNameCopy', img2) #copies img2 into fuckNameCopy file
    cv2.destroyAllWindows
###################################################################################################
""" IMAGE PROCESSING 
CAPTURE LIVESTREAM FROM CAMERA:

-if you want to read a file name from a specific file, you can pass that into VideoCapture()
-if you have multiple cameras (line 58), you can index using 1 or 2, instead of 0, -1. 

"""
cap = cv2.VideoCapture('fileName.avi or fileName.mp4')
otherCapture = cv2.VideoCapture('device index of your camera') #usually either 0 or -1
cap.read() #returns true if frame is available

#while loop to capture indefinetely
while(True): 
   # 
    ret, frameVariable = cap.read()
    # .read returns true if frame is available
    # frameVariable will get the frame saved by .read()
    # ret will be T/F depending on whether frame is available

    cv2.imshow('frameyFramer', frameVariable) #frameyFramer is name of window displaying the image


     #if q is pressed, break out of loop
    if cv2.waitKey(1) & 0xFF == ord('q'): break 
cap.release() # free resources
cap.destroyAllWindows()
###################################################################################################
""" IMAGE PROPERTIES """

#To use the capture-> videoCapture class
#To save the video-> videoWriter class
    #fourcc code is a 4-byte code used to specify video codec

cap = cv2.VideoCapture('fileName.avi or fileName.mp4')
otherCapture = cv2.VideoCapture('device index of your camera') #usually either 0 or -1
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('fileName.avi')
cap.read() #returns true if frame is available

print(cap.isOpened()) #checks if video is open or not, returns true if open
while(cap.isOpened):  #same^ and if this line gives you false, do cap.open() to open it
    ret, frameVariable = cap.read(frameVariable)

    #This "get" method can have prop IDs as arguments, to find properties such as height+width of frame
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    #converts frame to grayscale as example color conversion
    gray = cv2.cvtColor(frameVariable, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frameyFramer', frameVariable)
    if cv2.waitKey(1) & 0xFF == ord('q'): break 
cap.release() # free resources
cap.destroyAllWindows()
###################################################################################################
""" MOUSE EVENTS (Tutorial video #8) """
# (Many different types of click events)
# The following creates a black image, dispalys x y coords of where we click on the image
import numpy as np
import cv2

#iterating over all function names, variable names, etc
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

# click_event(event, x y coords of image where we are clicking with our mouse, flags, parameters)
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ', ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', ' + str(y)
        cv2.putText(img, strXY, (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]

#creating a black image, also wtf is this following line
img = np.zeros((512, 512, 3), np.uint8)
cv2.imshow('image', img)

cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
###################################################################################################
""" MOUSE EVENTS (Tutorial video #9) """
# For ex. A lil summ for line-drawing on an image of a map using mouse events
import numpy as np
import cv2
 
def click_event2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        # when adding the below -1, it becomes color-fill/"paintbucket" tool rather than just the outline
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        #whereever mouse is clicked will saved in the array
        points.append((x, y))
        if len(points) >=2:
                                 # -1 to get last value in an array, and -2 for 2nd last, etc
            cv2.line(img, points[-1], points[-2], (255, 0, 0, 5))
        cv2.imshow('image', img)
img = np.zeros((512, 512, 3), np.uint8)
#above if you used the 

cv2.imshow('image', img)
points = []
cv2.setMouseCallback('image', click_event2)

cv2.waitKey(0)
cv2.destroyAllWindows()

#################
""" MOUSE EVENTS (Tutorial video #9) """
# For ex. A lil summ for line-drawing on an image of a map using mouse events
import numpy as np
import cv2
 
def click_event2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[x, y, 1]
        red = img[x, y, 2]
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        myColorImage = np.zeros((512, 512, 3), np.uint8)

        myColorImage[:] = [blue, green, red]

        cv2.imwrite('colorImage.png', myColorImage)

img = cv2.imread('image_67199233.JPG')
cv2.imshow('image', img)
points = []
cv2.setMouseCallback('image', click_event2)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Is it faster to make individual calls to lets say lights on the engines, moving the drone, etc 
# all separetely or is it faster to make one call to the drone to do all of those things at once?