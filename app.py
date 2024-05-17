import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageOps 
from tensorflow import keras
from keras.models import load_model

st.write("""
# Mood Classifier
""")

file = st.file_uploader("Choose photo from the computer", type=["jpg", "png"])

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

if file is not None:
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)  # Use np.frombuffer to handle binary data

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and extract ROIs
    rois = []
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = image[y:y + h, x:x + w]
        rois.append(roi)

    # Display the original image
    st.image(image, channels="BGR", caption='Original Image')

    # Display the image with detected faces
    st.image(image, channels="BGR", caption=f'Image with {len(faces)} face(s) detected')

    if rois:
        # Load the pre-trained emotion classifier model
        model = load_model('model (1).h5')

        # Make predictions for each detected face
        for roi in rois:
            roi_gray = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
            roi_gray = roi_gray.astype('float') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            string="OUTPUT : "
            st.success(string)
            prediction = model.predict(roi_gray)
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)
            cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image with predicted emotions
        st.image(image, channels="BGR", caption='Image with Predicted Emotions')
