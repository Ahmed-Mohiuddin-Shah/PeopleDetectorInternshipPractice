import datetime
import tkinter as tk
from functions import *
from deepface import DeepFace
from PIL import Image, ImageTk
import cv2

#sqlite3
import sqlite3



def detect_face(cap, customCanvas, window, embeddings):
    ret, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]
    
    new_embeddings = []

    # detect faces
    detections = DeepFace.extract_faces(frame, detector_backend='yunet', enforce_detection=False)

    if detections is not None:
        for detection in detections:
            if detection["confidence"] < 0.70:
                continue
            face = detection['facial_area']
            detected_face = frame[face['y']:face['y']+face['h'], face['x']:face['x']+face['w']]
            confidence = detection['confidence']
            embedding_new = DeepFace.represent(detected_face, model_name = 'Facenet', enforce_detection = False)

            for embedding in embedding_new:
                embedding['location_tuple'] = (face['x'], face['y'], face['w'], face['h'])
                new_embeddings.append(embedding)

            person_id, embedding_new_single = check_face_exists(embeddings, embedding_new)
            if person_id is None:
                conn = sqlite3.connect('datbase.db')
                for embedding in embedding_new_single:
                    embedding = ", ".join(map(str ,embedding['embedding']))
                    # print(embedding)
                    save_face(embedding, conn)
                conn.close()

            handle_face(frame, face, confidence)
            
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = customCanvas.update_frame(frame)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    customCanvas.create_image(0, 0, image=photo, anchor=tk.NW)
    customCanvas.image = photo

    faces_in_shapes = customCanvas.check_face_in_shapes(new_embeddings)

    if len(faces_in_shapes) != 0:
        customCanvas.alert.config(text="Face detected in the shape")
        conn = sqlite3.connect('datbase.db')
        c = conn.cursor()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for face in faces_in_shapes:
            # print(face)
            face = face['embedding']
            id = get_id_from_embedding(embeddings, face)
            c.execute("INSERT INTO detectedtimestamps (timestamp, person_id) VALUES (?, ?)", (timestamp, id))
        conn.commit()
    else:
        customCanvas.alert.config(text="Face not detected in the shape")
    
    window.after(1, detect_face, cap, customCanvas, window, embeddings)

def init_db():
        # Create a connection to the database
    conn = sqlite3.connect('datbase.db')

    # Create a cursor
    c = conn.cursor()

    # Create a table
    c.execute("""CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY,
        encodings BLOB
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS shapes (
        id INTEGER PRIMARY KEY,
        start_x INTEGER,
        start_y INTEGER,
        end_x INTEGER,
        end_y INTEGER,
        points TEXT,
        shape TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS detectedtimestamps (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        person_id INTEGER
    )""")

    # Commit the connection
    conn.commit()
    conn.close()

def get_embeddings():
    conn = sqlite3.connect('datbase.db')
    c = conn.cursor()
    c.execute("SELECT * FROM people")
    rows = c.fetchall()
    conn.close()

    info = {}

    #make dictionary

    for row in rows:
        #convert row[1] to list of floats
        embedding = [float(i) for i in row[1].split(",")]
        info[row[0]] = embedding
    return info

def get_shapes():
    conn = sqlite3.connect('datbase.db')
    c = conn.cursor()
    c.execute("SELECT * FROM shapes")
    rows = c.fetchall()
    conn.close()

    info = {}
    shape_info= {}

    #make dictionary

    for row in rows:
        shape_info['start_x'] = row[1]
        shape_info['start_y'] = row[2]
        shape_info['end_x'] = row[3]
        shape_info['end_y'] = row[4]
        points = row[5].split(", ") if row[5] is not None else []
        new_list = []

        for i in range(0, len(points)-1, 2):
            points[i] = int(points[i])
            points[i+1] = int(points[i+1])
            new_list.append((points[i], points[i+1]))

        shape_info['points'] = new_list
        shape_info['shape'] = row[6]
        info[row[0]] = shape_info
    return info

def main():
    init_db()

    embeddings = get_embeddings()

    shapes = get_shapes()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = tk.Tk()
    window.title("Face Detector")

    window.geometry("1280x900")

    selection_menu = SelectionMenu(window, "Shape", ["Rectangle", "Circle", "Triangle", "Random Shape"], 1, 0)
    customCanvas = CustomCanvas(window, selection_menu, "People Detector", 1280, 720, pre_load_shapes=shapes)
    
    title_label = tk.Label(window, text="Face Detector", font=('Arial', 24))
    title_label.grid(row=0, column=0, columnspan=7)

    set_shape_button = tk.Button(window, text="Save Shape", font=('Arial', 16), command=lambda: customCanvas.save_shape())
    set_shape_button.grid(row=1, column=5)

    clear_button = tk.Button(window, text="Clear", font=('Arial', 16) , command=lambda: customCanvas.clear_shapes())
    clear_button.grid(row=1, column=6)

    detect_face(cap, customCanvas, window, embeddings)

    window.mainloop()
    # Release handle to the webcam
    should_exit = True
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
