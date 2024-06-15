import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from keras.models import load_model

# Load pre-trained model
model = load_model('bestmodelprediction.h5')

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Directly specify the cascade classifier path
        cascade_path = 'path/to/your/opencv/haarcascades/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def transform(self, frame):
        # Convert the image to grayscale
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If no faces are detected, return the original image
        if len(faces) == 0:
            return img

        # For each detected face, predict the corresponding emotion
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = gray[y:y+h, x:x+w]

            # Resize the face ROI to match the input size of the model
            face_roi = cv2.resize(face_roi, (48, 48))

            # Normalize the pixel values to be between 0 and 1
            face_roi = face_roi / 255.0

            # Reshape the face ROI to be a 4D tensor with shape (1, height, width, depth)
            face_roi = face_roi.reshape(1, face_roi.shape[0], face_roi.shape[1], 1)

            # Predict the emotion using the model
            preds = model.predict(face_roi)

            # Get the index of the predicted emotion
            emotion_index = preds.argmax(axis=1)[0]

            # Define a list of emotion labels
            emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Add the predicted emotion label to the image
            cv2.putText(img, emotions[emotion_index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

def main():
    # Create a Streamlit window for displaying the video feed and predicted emotions
    st.title('Real-time Face Emotion Detection')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Start the video stream
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == '__main__':
    main()
