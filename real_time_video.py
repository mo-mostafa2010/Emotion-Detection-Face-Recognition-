from keras.preprocessing.image import img_to_array #library for Machine LEarning 
import imutils #to Read the frame
import face_recognition #library for face recognition
import cv2 #open the camera
from keras.models import load_model #machine learning"env"
import numpy as np #for math forth deminssoin 
import psutil #for memmorey usage and processing
 
# parameters for loading data and images
# load moduels
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"] # Array of Emotions


#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
video_capture = cv2.VideoCapture(0)
mostafa_image = face_recognition.load_image_file("mostafa.jpg")
mostafa_face_encoding = face_recognition.face_encodings(mostafa_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    mostafa_face_encoding
]
known_face_names = [
    "Mohamed Mostafa"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    #Minimize memorey usage to 0.25 so get faster runing time
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (30, 30, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    #reading the frame
    #frame = imutils.resize(frame) Speed up the run time by getting one frame for both emotion and face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray)
    canvas = np.zeros((400, 400, 4))
    frameClone = frame
    #frameClone = frame
    if len(faces) > 0:
        faces = sorted(faces,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0] #lamda return infiinty to make the program run in real time, if replaced with number would be sec
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = preds
        label = EMOTIONS[preds.argmax()] 
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                    # draw the label + probability bar on the canvas
                # emoji_face = feelings_faces[np.argmax(preds)]
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                (0, 0, 255), 2)
#    for c in range(0, 3):
#        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)

    cv2.imshow('Face Detection', frame)
    cv2.imshow("Emotoin Detection", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("percente of CPU Usage is ",psutil.cpu_percent())
print("percente of Disk Usage is ",psutil.disk_usage('/'))
print("virtual Memorey Usage is ", psutil.virtual_memory())
video_capture.release()
cv2.destroyAllWindows()
