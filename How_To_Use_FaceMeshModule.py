from FaceMeshModule import Face
import cv2

# read the image
image = cv2.imread('fm4.jpeg')
image = cv2.resize(image,(800,550))

'''
create the object of Face class
provide all the arguments 

static_image_mode: True by default. Set it to False if the you are using it on a video or live streaming
 
max_num_faces: Maximum number of faces to detect

min_tracking_confidence: Tracking confidence threshold

min_detection_confidence: Detection confidence threshold
'''
face_mesh = Face(static_image_mode=True,max_num_faces=1,min_tracking_confidence=0.5,min_detection_confidence=0.5)


# Run the inference
bool = face_mesh.dectect(image)  # bool = True if face/s detected else False

# To draw the landmarks on eyebrows with connections
# only = 'face' or 'eyes' or 'eyebrows' or 'lips' or 'face_oval'
image = face_mesh.draw_landmarks(only="face_oval")

# To get the face landmarks
lm = face_mesh.face_landmarks()
print(lm)

# To draw all the points (xn,yn) by yourselves
for location in lm[0]["face"]:
    cv2.circle(image,location,1,(255,255,255),1)
    break

# To get the landmarks of eyes only
lm = face_mesh.eye_landmarks()
# print(lm)

# To get the landmarks of lips only
lm = face_mesh.lips_landmarks()
# print(lm)

# To get the landmarks of oval surrounding the face
lm = face_mesh.face_oval_landmarks()
# print(lm)

# To get the landmarks of eyebrows only
lm = face_mesh.eyebrows_landmarks()
# print(lm)


# Display the image
cv2.imshow("image",image)

# Provide the waiting time. 0 is for infinite amount of waiting time
cv2.waitKey(0)

# Destroy the window
cv2.destroyAllWindows()
