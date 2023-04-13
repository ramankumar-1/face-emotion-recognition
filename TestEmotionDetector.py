import os
import cv2
import numpy as np
from keras.models import model_from_json
import time

emotion_dict = {0:"Angry", 1:"Disgusted", 2:"Fearful", 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"}

os.chdir(r"C:/Users/raman/Desktop/FER-Project")

json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)
cap.set(3,640) # width
cap.set(4,480) # height

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0,255,0),6)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predicting the emotions
        emotion_prediction = emotion_model.predict(cropped_img)

        max_val=np.max(emotion_prediction)
        maxindex=np.argmax(emotion_prediction)

        cv2.putText(frame, emotion_dict[maxindex]+":"+str(max_val*100)+"%", 
                    (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()