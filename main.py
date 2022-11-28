import face_recognition
import cv2
import numpy as np


# 人臉辨識步驟：
# 1. 人臉偵測：將圖片轉為黑白，利用ＨＯＧ找出圖片中臉部部分
# 2. 投射面孔：定位出臉部的68個點位，保留平行線扭曲圖像，目的為了可以辨識出各角度的臉部
# 3. 編碼人臉：通過訓練過的神經網路產生出128個測量值。 訓練過程通過一次查看 3 張面部圖像來工作：
#       #1.)加載已知人的訓練面部圖像
#       #2.)加載同一個人的另一張照片
#       #3.)加載一個完全不同的人的照片
#       然後該算法查看它當前為這三幅圖像中的每幅圖像生成的測量值。然後它稍微調整神經網絡，以確保它為 #1 和 #2 生成的測量值稍微接近，同時確保 #2 和 #3 的測量值稍微分開
# 4. 從編碼中找人名：利用機器學習演算法（SVM分類器）從數據庫找出與測試圖具有最接近測量值的人

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
victor_image = face_recognition.load_image_file("victor.jpg",mode='RGB')
victor_face_encoding = face_recognition.face_encodings(victor_image)[0]

# Load a second sample picture and learn how to recognize it.
joy_image = face_recognition.load_image_file("joy.jpg",mode='RGB')
joy_face_encoding = face_recognition.face_encodings(joy_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    victor_face_encoding,
    joy_face_encoding
]
known_face_names = [
    "Victor Fu",
    "Joy Low"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6) # tolerance相似度設定值，越小越嚴格
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

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
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()