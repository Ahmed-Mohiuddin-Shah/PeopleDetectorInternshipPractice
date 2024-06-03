from deepface import DeepFace
from mtcnn import MTCNN
import cv2

# using deepface to detect faces
def detect_face(frame):
    embeddings = []
    # detect faces
    faces = DeepFace.extract_faces(frame, detector_backend='yunet', enforce_detection=False)

    for face in faces:
        # print(face)
        confidence = face['confidence']
        if confidence < 0.70:
            continue
        detected_face = frame[face['facial_area']['y']:face['facial_area']['y']+face['facial_area']['h'], face['facial_area']['x']:face['facial_area']['x']+face['facial_area']['w']]
        embedding = DeepFace.represent(detected_face, model_name = 'Facenet', enforce_detection = False)
        embeddings.append(embedding)
        print(embeddings)
        # print(face)
    #     {'face': array([[[0.9372549 , 0.96862745, 0.96078431],
    #     [0.8745098 , 0.91372549, 0.90588235],
    #     [0.81176471, 0.85882353, 0.84705882],
    #     ...,
    #     [0.61568627, 0.52156863, 0.53333333],
    #     [0.63529412, 0.52156863, 0.52941176],
    #     [0.63137255, 0.52156863, 0.50980392]],

    #    [[0.86666667, 0.89803922, 0.89019608],
    #     [0.8       , 0.83921569, 0.83137255],
    #     [0.77254902, 0.81960784, 0.80784314],
    #     ...,
    #     [0.60784314, 0.51372549, 0.5254902 ],
    #     [0.60784314, 0.50588235, 0.50980392],
    #     [0.62352941, 0.52156863, 0.50588235]],

    #    [[0.86666667, 0.88627451, 0.88627451],
    #     [0.81176471, 0.83529412, 0.83137255],
    #     [0.76862745, 0.78823529, 0.78039216],
    #     ...,
    #     [0.59215686, 0.50588235, 0.50196078],
    #     [0.61176471, 0.50980392, 0.51764706],
    #     [0.62745098, 0.5254902 , 0.51764706]],

    #    ...,

    #    [[0.49803922, 0.49803922, 0.50588235],
    #     [0.38823529, 0.39215686, 0.41568627],
    #     [0.38039216, 0.38823529, 0.42352941],
    #     ...,
    #     [0.61960784, 0.48627451, 0.42352941],
    #     [0.62352941, 0.49019608, 0.43137255],
    #     [0.61176471, 0.4745098 , 0.42745098]],

    #    [[0.41176471, 0.40784314, 0.43529412],
    #     [0.39215686, 0.38823529, 0.42352941],
    #     [0.37254902, 0.36862745, 0.40784314],
    #     ...,
    #     [0.60784314, 0.47843137, 0.42352941],
    #     [0.60392157, 0.4745098 , 0.41960784],
    #     [0.61176471, 0.48235294, 0.43137255]],

    #    [[0.37647059, 0.38431373, 0.41176471],
    #     [0.38431373, 0.38431373, 0.42745098],
    #     [0.33333333, 0.32941176, 0.37647059],
    #     ...,
    #     [0.6       , 0.48627451, 0.43921569],
    #     [0.60392157, 0.49019608, 0.44313725],
    #     [0.6       , 0.48627451, 0.43921569]]]), 'facial_area': {'x': 524, 'y': 356, 'w': 27, 'h': 34, 'left_eye': (541, 367), 'right_eye': (530, 369)}, 'confidence': 0.85}

        x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame

# using cv2 to capture video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = detect_face(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()