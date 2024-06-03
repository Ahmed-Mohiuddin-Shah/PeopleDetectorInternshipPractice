from deepface import DeepFace
import cv2

# using deepface to detect faces
def detect_face(frame):
    # detect faces
    detected_faces = DeepFace.extract_faces(frame, detector_backend='mtcnn')
    # draw rectangle around the detected faces
    for face in detected_faces:
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