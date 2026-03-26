import webbrowser
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model


RUN_DURATION = 60  

START_TIME = time.time()


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

model = load_model('emotion_model.hdf5', compile=False)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

music_dict = {
    "Happy": "https://www.youtube.com/results?search_query=happy+songs",
    "Sad": "https://www.youtube.com/results?search_query=sad+songs",
    "Angry": "https://www.youtube.com/results?search_query=calm+music",
    "Neutral": "https://www.youtube.com/results?search_query=chill+music"
}


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


stable_emotion = None
emotion_count = 0

opened_emotion = None
last_open_time = 0
COOLDOWN = 10  

while True:

    
    if time.time() - START_TIME > RUN_DURATION:
        print("⏱ Time limit reached. Exiting program...")
        break

    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        
        if emotion == stable_emotion:
            emotion_count += 1
        else:
            stable_emotion = emotion
            emotion_count = 1

        current_time = time.time()

       
        if emotion_count >= 5:

            if (
                emotion in music_dict and
                emotion != opened_emotion and
                current_time - last_open_time > COOLDOWN
            ):
                print("🎯 Stable Emotion Detected:", emotion)

                webbrowser.open(music_dict[emotion])

                opened_emotion = emotion
                last_open_time = current_time

                emotion_count = 0  

        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Emotion Music Player", frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Program stopped by user")
        break

cap.release()
cv2.destroyAllWindows()