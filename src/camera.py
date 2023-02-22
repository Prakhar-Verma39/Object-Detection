# Detecting Objects in Real Time with OpenCV deep learning library

# Algorithm:
# Reading stream video from camera --> Loading YOLO v3 Network -->
# --> Reading frames in the loop --> Getting blob from the frame -->
# --> Implementing Forward Pass --> Getting Bounding Boxes -->
# --> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
# --> Showing processed frames in OpenCV Window

# Result:
# Window with Detected Objects, Bounding Boxes and Labels in Real Time


# Importing needed libraries
import numpy as np
import cv2
from playsound import playsound

"""
Start of:
Reading stream video from camera
"""

# Defining 'VideoCapture' object
# and reading stream video from camera
camera = cv2.VideoCapture(0)

# Preparing variable for writer
# that we will use to write processed frames
writer = None

# Preparing variable for video count
video_count = 1

# Preparing variable for adding delay -- d_count , recording time -- active_count , locking the writer -- lock
d_count = 0
active_count = 0
lock = True

# Preparing variables for spatial dimensions of the frames
h, w = None, None

"""
End of:
Reading stream video from camera
"""

"""
Start of:
Loading YOLO v3 network
"""

# Loading COCO class labels from file
# Opening file

with open('..\\data\\plastic\\classes.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = []
    for line in f:
        labels += [line.strip()]

# Loading trained YOLO v3 Objects Detector
# with the help of 'dnn' library from OpenCV

network = cv2.dnn.readNetFromDarknet('..\\data\\plastic\\yolov3_project.cfg',
                                     '..\\data\\plastic\\yolov3_project_best3.weights')

#  unconnected output layers' names that we need from YOLO v3 algorithm
layers_names_output = ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
threshold = 0.3

"""
End of:
Loading YOLO v3 network
"""

"""
Start of:
Reading frames in the loop
"""

# Defining loop for catching frames
while camera.isOpened():
    # Capturing frame-by-frame from camera
    _, frame = camera.read()
    # print("frame", frame.shape)

    # Getting spatial dimensions of the frame
    # we do it only once from the very beginning
    # all other frames have the same dimension
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = frame.shape[:2]
        # print(h, w)
    """
    Start of:
    Getting blob from current frame
    """

    # Getting blob from input image

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)  # blob.shape - tuple =>
    # (no. of images, no. of channels, width, height)
    # print("blob", blob.shape)

    """
    End of:
    Getting blob from current frame
    """

    """
    Start of:
    Implementing Forward pass
    """

    # Implementing forward pass with our blob and only through output layers
    network.setInput(blob)  # setting blob as input to the network
    output_from_network = network.forward(layers_names_output)

    """
    End of:
    Implementing Forward pass
    """

    """
    Start of:
    Getting bounding boxes
    """

    # Preparing lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []
    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # print(detected_objects.shape)  # (85,)

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial image size
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    """
    End of:
    Getting bounding boxes
    """

    """
    Start of:
    Non-maximum suppression
    """

    # It is needed to make sure that data type of the boxes is 'int'
    # and data type of the confidences is 'float'
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)
    """
    End of:
    Non-maximum suppression
    """

    """
    Start of:
    Drawing bounding boxes and labels
    """

    # Defining counter for detected objects
    Plastic_bag = 0
    Tin_can = 0
    Bottle = 0

    # Checking if there is at least one detected object
    # after non-maximum suppression
    if len(results) > 0:

        # Going through indexes of results
        for i in results.flatten():

            if int(class_numbers[i]) == 0:
                Plastic_bag += 1
            elif int(class_numbers[i]) == 1:
                Tin_can += 1
            else:
                Bottle += 1

            # Getting current bounding box coordinates,
            # its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Drawing bounding box on the original current frame
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          [0, 255, 0], 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.2f}%'.format(labels[int(class_numbers[i])],
                                                    confidences[i] * 100)

            # Putting text with label and confidence on the original image
            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 2)
            d_count += 1
            lock = False

    if lock is False:
        active_count += 1
        """
        Start of:
        Writing processed frame into the file
        """
        # Initializing writer
        # we do it only once from the very beginning
        # when we get spatial dimensions of the frames
        if writer is None:
            # Constructing code of the codec
            # to be used in the function VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Writing current processed frame into the video file
            writer = cv2.VideoWriter(f'..\\surveillance\\video{video_count}.mp4', fourcc, 8,
                                     (frame.shape[1], frame.shape[0]), True)

        # Write processed current frame to the file
        writer.write(frame)
        """
        End of:
        Writing processed frame into the file
        """
    print(d_count)

    # Incrementing video count to save next video
    if active_count > 50:
        active_count = 0
        lock = True
        writer = None
        video_count += 1

    if d_count > 50:
        d_count = 0
        # playsound('sound1.wav', block=False)  # playing a warning sound
    """
    End of:
    Drawing bounding boxes and labels
    """

    """
    Start of:
    Displaying count of objects
    """

    # Putting text with counter on the original image
    cv2.putText(frame, f'Plastic bag : {Plastic_bag}', (5, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, [0, 0, 255], 2)
    cv2.putText(frame, f'Tin can : {Tin_can}', (5, 65),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, [0, 0, 255], 2)
    cv2.putText(frame, f'Bottle : {Bottle}', (5, 100),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, [0, 0, 255], 2)

    """
    End of:
    Displaying count of objects
    """

    """
    Start of:
    Showing processed frames in OpenCV Window
    """

    # Showing results obtained from camera in Real Time

    # Showing current frame with detected objects
    # Giving name to the window with current frame
    # And specifying that window is resizable
    cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO v3 Real Time Detections', frame)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

    """
    End of:
    Showing processed frames in OpenCV Window
    """

"""
End of:
Reading frames in the loop
"""

# Releasing camera
camera.release()
# Destroying all opened OpenCV windows
cv2.destroyAllWindows()
