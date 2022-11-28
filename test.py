import face_recognition

image = face_recognition.load_image_file("victor.jpg")
my_face_encoding = face_recognition.face_encodings(image)[0]

print(my_face_encoding)

