# print("hi")

import face_recognition
# import os
# import sys
#
# sys.argv[1]

tolerance = 0.5

print("loading known image")
known_image = face_recognition.load_image_file("taylor_resize1.jpg")
known_image_encoding = face_recognition.face_encodings(known_image)[0]

print("loading unknown image")
unknown_image = face_recognition.load_image_file("brian_resize3.jpg")
unknown_image_encoding = face_recognition.face_encodings(unknown_image)

results = face_recognition.compare_faces(known_image_encoding, unknown_image_encoding, tolerance)
face_distances = face_recognition.face_distance(known_image_encoding, unknown_image_encoding)

print(results)

if True in results:
    # print("match")
    print("match: " + str((1 - face_distances[0]) * 100) + "%")
else:
    # print("not match")
    print("not match: " + str((1 - face_distances[0]) * 100) + "%")
