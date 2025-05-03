import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize mixer for sound alarm
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar Cascade classifiers for face and eyes
face = cv2.CascadeClassifier(r'haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_righteye_2splits.xml')

# Label list for prediction (Closed or Open)
lbl = ['Closed', 'Open']

# Load the pre-trained Keras model in the new format
model = load_model('models/cnnCat2_new_format')

# Get the current working directory
path = os.getcwd()

# Start the video capture (use webcam, 0 is the default camera)
cap = cv2.VideoCapture(0)

# Define font for text on frames and initialize variables
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

# Main loop to process video feed and detect drowsiness
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Convert the frame to grayscale for easier processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes using the classifiers
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw a rectangle at the bottom of the frame for displaying status
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Detect faces and draw rectangles around them
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Process the right eye
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255.0
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)

        # Predict the state of the right eye (Closed or Open)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        lbl = 'Open' if rpred[0] == 1 else 'Closed'
        break

    # Process the left eye
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255.0
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)

        # Predict the state of the left eye (Closed or Open)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        lbl = 'Open' if lpred[0] == 1 else 'Closed'
        break

    # Check if both eyes are closed and increase the score
    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Ensure the score doesn't go below zero
    score = max(score, 0)

    # Display the score on the frame
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Trigger the alarm if the score exceeds the threshold (indicating drowsiness)
    if score > 15:
        # Save a snapshot when drowsiness is detected
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()  # Play the alarm sound
        except:  # If the sound is already playing, ignore the error
            pass

        # Draw a thick red rectangle around the frame as a warning
        thicc = thicc + 2 if thicc < 16 else thicc - 2
        thicc = max(thicc, 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    # Display the processed frame
    cv2.imshow('Drowsiness Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
