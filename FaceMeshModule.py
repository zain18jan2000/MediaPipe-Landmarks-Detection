import cv2
import mediapipe as mp
import itertools


'''
 This module is written to make the use of mediapipe for face landmarks detections easy for all
 mediapipe version = 0.9.1.0 is used while writing this module. This will probably work with other versions as well
 but it is not guaranteed. 
 Not all the functionalities of mediapipe can be accessed through this module. Hence more work can be done 
 on it in future.  
 '''

class Face:
    def __init__(self,static_image_mode = True,max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.face_mesh = mp.solutions.face_mesh
        self.face_mesh_img = self.face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                     max_num_faces=max_num_faces,
                                                     min_detection_confidence=min_detection_confidence,
                                                     min_tracking_confidence=min_tracking_confidence)
        self.image = None
        self.face_mesh_results = False

        self.drawings = mp.solutions.drawing_utils
        self.drawing_style = mp.solutions.drawing_styles

        self.right_eye_max_ids = [4,1,15,8]  # in out top down
        self.left_eye_max_ids = [7,9,2,11]   # in out top down

    # use this function to run the inference
    def dectect(self,image,draw = True):
        self.image = image
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.face_mesh_results = self.face_mesh_img.process(rgb_image)

        if self.face_mesh_results.multi_face_landmarks:
            return True
        else:
            return False

    '''
    use this method to draw the landmarks and their connections    
    
    only = 'face' to draw all the face landmark and their connection
    only = 'eyes' for only eyes
    only = 'lips' for only lips
    only = 'face_oval' only for only face oval
    only = 'eyebrows' for only eyebrows
    
    more functionalities can be added, for example, draw the landmarks and connections of only left eye  
    '''
    def draw_landmarks(self, only = "face"):
        if self.face_mesh_results.multi_face_landmarks:
            for face_landmarks in self.face_mesh_results.multi_face_landmarks:

                if only == "face":
                    self.drawings.draw_landmarks(image=self.image,
                                        landmark_list=face_landmarks, connections=self.face_mesh.FACEMESH_TESSELATION,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=self.drawing_style.get_default_face_mesh_tesselation_style())

                elif only == "eyes":
                    self.drawings.draw_landmarks(image=self.image,
                                                 landmark_list=face_landmarks,
                                                 connections=self.face_mesh.FACEMESH_LEFT_EYE,
                                                 landmark_drawing_spec=None,
                                                 connection_drawing_spec=self.drawing_style.get_default_face_mesh_tesselation_style())

                    self.drawings.draw_landmarks(image=self.image,
                                                 landmark_list=face_landmarks,
                                                 connections=self.face_mesh.FACEMESH_RIGHT_EYE,
                                                 landmark_drawing_spec=None,
                                                 connection_drawing_spec=self.drawing_style.get_default_face_mesh_tesselation_style())

                elif only == "lips":
                    self.drawings.draw_landmarks(image=self.image,
                                                 landmark_list=face_landmarks,
                                                 connections=self.face_mesh.FACEMESH_LIPS,
                                                 landmark_drawing_spec=None,
                                                 connection_drawing_spec=self.drawing_style.get_default_face_mesh_tesselation_style())

                elif only == "face_oval":
                    self.drawings.draw_landmarks(image=self.image,
                                         landmark_list=face_landmarks,
                                         connections=self.face_mesh.FACEMESH_FACE_OVAL,
                                         landmark_drawing_spec=None,
                                         connection_drawing_spec=self.drawing_style.get_default_face_mesh_tesselation_style())

                elif only == "eyebrows":
                    self.drawings.draw_landmarks(image=self.image,
                                         landmark_list=face_landmarks,
                                         connections=self.face_mesh.FACEMESH_LEFT_EYEBROW,
                                         landmark_drawing_spec=None,
                                         connection_drawing_spec=self.drawing_style.get_default_face_mesh_tesselation_style())

                    self.drawings.draw_landmarks(image=self.image,
                                                 landmark_list=face_landmarks,
                                                 connections=self.face_mesh.FACEMESH_RIGHT_EYEBROW,
                                                 landmark_drawing_spec=None,
                                                 connection_drawing_spec=self.drawing_style.get_default_face_mesh_tesselation_style())

        else:
            print("NO LANDMARKS TO DRAW")

        return self.image

    # get the landmarks of eyes using this function
    def eye_landmarks(self, only_four = False):
        eye_land_marks = []

        if self.face_mesh_results.multi_face_landmarks:
            for face_no, landmarks in enumerate(self.face_mesh_results.multi_face_landmarks):
                if only_four == False:
                    RIGHTEYE_IDXS = list(set(itertools.chain(*self.face_mesh.FACEMESH_RIGHT_EYE)))
                    LEFTEYE_IDXS = list(set(itertools.chain(*self.face_mesh.FACEMESH_LEFT_EYE)))

                    right_eye_landmarks = []
                    left_eye_landmarks = []

                    for left_eye_idx, right_eye_idx in zip(LEFTEYE_IDXS, RIGHTEYE_IDXS):
                        X = int(landmarks.landmark[right_eye_idx].x * self.image.shape[1])
                        Y = int(landmarks.landmark[right_eye_idx].y * self.image.shape[0])
                        right_eye_landmarks.append((X, Y))

                        X = int(landmarks.landmark[left_eye_idx].x * self.image.shape[1])
                        Y = int(landmarks.landmark[left_eye_idx].y * self.image.shape[0])
                        left_eye_landmarks.append((X, Y))

                    eye_land_marks.append({"id": face_no,
                                           "R_eye": right_eye_landmarks,
                                           "L_eye": left_eye_landmarks})

                else:

                    right_eye_landmarks = []
                    left_eye_landmarks = []

                    for left_eye_idx, right_eye_idx in zip(self.left_eye_max_ids, self.right_eye_max_ids):
                        X = int(landmarks.landmark[right_eye_idx].x * self.image.shape[1])
                        Y = int(landmarks.landmark[right_eye_idx].y * self.image.shape[0])
                        right_eye_landmarks.append((X, Y))

                        X = int(landmarks.landmark[left_eye_idx].x * self.image.shape[1])
                        Y = int(landmarks.landmark[left_eye_idx].y * self.image.shape[0])
                        left_eye_landmarks.append((X, Y))

                    eye_land_marks.append({"id": face_no,
                                           "R_eye": right_eye_landmarks,
                                           "L_eye": left_eye_landmarks})

        return eye_land_marks

    # get the landmarks of lips using this function
    def lips_landmarks(self):

        All_lips_landmarks = []
        if self.face_mesh_results.multi_face_landmarks:
            for face_no, landmarks in enumerate(self.face_mesh_results.multi_face_landmarks):
                LIPS_IDXS = list(set(itertools.chain(*self.face_mesh.FACEMESH_LIPS)))
                lips_landmarks = []
                for lips_idx in LIPS_IDXS:
                    X = int(landmarks.landmark[lips_idx].x * self.image.shape[1])
                    Y = int(landmarks.landmark[lips_idx].y * self.image.shape[0])
                    lips_landmarks.append((X, Y))

                All_lips_landmarks.append({
                    "id" : face_no,
                    "lips" : lips_landmarks
                })

        return All_lips_landmarks

    # get the landmarks of oval surrounding the face using this function
    def face_oval_landmarks(self):

        All_FaceOval_landmarks = []
        if self.face_mesh_results.multi_face_landmarks:
            for face_no, landmarks in enumerate(self.face_mesh_results.multi_face_landmarks):
                OVAL_IDXS = list(set(itertools.chain(*self.face_mesh.FACEMESH_FACE_OVAL)))
                oval_landmarks = []
                for oval_idx in OVAL_IDXS:
                    X = int(landmarks.landmark[oval_idx].x * self.image.shape[1])
                    Y = int(landmarks.landmark[oval_idx].y * self.image.shape[0])
                    oval_landmarks.append((X, Y))

                All_FaceOval_landmarks.append({
                    "id" : face_no,
                    "face_oval" : oval_landmarks
                })

        return All_FaceOval_landmarks

    # get the all the face landmarks using this function
    def face_landmarks(self):
        TESSELATION_LANDMARKS = []
        if self.face_mesh_results.multi_face_landmarks:
            for face_no, landmarks in enumerate(self.face_mesh_results.multi_face_landmarks):
                ALL_IDXS = list(set(itertools.chain(*self.face_mesh.FACEMESH_TESSELATION)))
                all_landmarks = []
                for all_idx in ALL_IDXS:
                    X = int(landmarks.landmark[all_idx].x * self.image.shape[1])
                    Y = int(landmarks.landmark[all_idx].y * self.image.shape[0])
                    all_landmarks.append((X, Y))

                TESSELATION_LANDMARKS.append({
                    "id": face_no,
                    "face": all_landmarks
                })

        return TESSELATION_LANDMARKS

    # get the landmarks of eyebrows using this method
    def eyebrows_landmarks(self, only_four = False):
        eyebrows_land_marks = []

        if self.face_mesh_results.multi_face_landmarks:
            for face_no, landmarks in enumerate(self.face_mesh_results.multi_face_landmarks):
                if only_four == False:
                    RIGHTEYEBROW_IDXS = list(set(itertools.chain(*self.face_mesh.FACEMESH_RIGHT_EYEBROW)))
                    LEFTEYEBROW_IDXS = list(set(itertools.chain(*self.face_mesh.FACEMESH_LEFT_EYEBROW)))

                    right_eyebrow_landmarks = []
                    left_eyebrow_landmarks = []

                    for left_eyebrow_idx, right_eyebrow_idx in zip(LEFTEYEBROW_IDXS, RIGHTEYEBROW_IDXS):
                        X = int(landmarks.landmark[right_eyebrow_idx].x * self.image.shape[1])
                        Y = int(landmarks.landmark[right_eyebrow_idx].y * self.image.shape[0])
                        right_eyebrow_landmarks.append((X, Y))

                        X = int(landmarks.landmark[left_eyebrow_idx].x * self.image.shape[1])
                        Y = int(landmarks.landmark[left_eyebrow_idx].y * self.image.shape[0])
                        left_eyebrow_landmarks.append((X, Y))

                    eyebrows_land_marks.append({"id": face_no,
                                           "R_eyebrow": right_eyebrow_landmarks,
                                           "L_eyebrow": left_eyebrow_landmarks})

        return eyebrows_land_marks




