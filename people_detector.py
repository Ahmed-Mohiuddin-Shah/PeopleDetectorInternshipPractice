import tkinter as tk
from functions import CustomCanvas, SelectionMenu, handle_face

from PIL import Image, ImageTk
import cv2

#sqlite3
import sqlite3

# Create a connection to the database
conn = sqlite3.connect('datbase.db')

# Create a cursor
c = conn.cursor()

# Create a table
c.execute("""CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY,
    name TEXT,
    encodings BLOB
)""")

def detect_face(cap, detector, customCanvas, window):
    ret, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]
    detector.setInputSize((width, height))

    detections = detector.detect(frame)
    faces = detections[1]

    if faces is not None:
        for face in faces:
            frame = handle_face(frame, face)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = customCanvas.update_frame(frame)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    customCanvas.create_image(0, 0, image=photo, anchor=tk.NW)
    customCanvas.image = photo

    if customCanvas.check_face_in_shapes(faces):
        customCanvas.alert.config(text="Face detected in the shape")
    else:
        customCanvas.alert.config(text="Face not detected in the shape")
    
    window.after(1, detect_face, cap, detector, customCanvas, window)

def main():
    scoreThreshold = 0.5
    nmsThreshold = 0.3
    backendTarget = 3
    targetId = 0
    size = (320, 320)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = tk.Tk()
    window.title("Face Detector")

    window.geometry("1280x900")

    selection_menu = SelectionMenu(window, "Shape", ["Rectangle", "Circle", "Triangle", "Random Shape"], 1, 0)
    customCanvas = CustomCanvas(window, selection_menu, "People Detector", 1280, 720)
    
    title_label = tk.Label(window, text="Face Detector", font=('Arial', 24))
    title_label.grid(row=0, column=0, columnspan=7)

    set_shape_button = tk.Button(window, text="Save Shape", font=('Arial', 16), command=lambda: customCanvas.save_shape())
    set_shape_button.grid(row=1, column=5)

    clear_button = tk.Button(window, text="Clear", font=('Arial', 16) , command=lambda: customCanvas.clear_shapes())
    clear_button.grid(row=1, column=6)

    detector = cv2.FaceDetectorYN.create(
        model="./face_detection_yunet_2023mar.onnx",
        config="",
        input_size=size,
        score_threshold=scoreThreshold,
        nms_threshold=nmsThreshold,
        backend_id=backendTarget,
        target_id=targetId,
    )

    detect_face(cap, detector, customCanvas, window)

    window.mainloop()
    # Release handle to the webcam
    should_exit = True
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
