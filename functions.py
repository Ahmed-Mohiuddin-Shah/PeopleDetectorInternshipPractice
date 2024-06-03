import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

class SelectionMenu:
    def __init__(self, window, title, options, row, column):
        self.window = window
        self.title = title
        self.options = options
        self.row = row
        self.column = column

        self.string_var = tk.StringVar(window, self.options[0])

        self.radio_buttons = []
        for i, option in enumerate(self.options):
            radio_button = tk.Radiobutton(window, text=option, value=option, variable=self.string_var)
            radio_button.grid(row=self.row, column=self.column + i)
            self.radio_buttons.append(radio_button)

    def get_selected_option(self):
        return self.string_var.get()

    def get_selected_index(self):
        return self.options.index(self.string_var.get())
    
    def set_selected_option(self, option):
        self.string_var.set(option)

    def set_selected_index(self, index):
        self.string_var.set(self.options[index])

class CustomCanvas:
    def __init__(self, window, selection_menu, title, width=640, height=480, frame=None):
        self.window = window
        self.frame = frame
        self.selection_menu = selection_menu
        self.window.title(title)
        self.canvas = tk.Canvas(window, width=width, height=height)
        self.canvas.grid(row=2, column=0, columnspan=7)
        self.alert = tk.Label(window, text="No face detected in shapes", font=('Arial', 16))
        self.alert.grid(row=3, column=0, columnspan=7)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.rect = None
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.image = None
        self.shapes = []
        self.points = []
        self.last_shape = None

    def on_mouse_click(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def on_mouse_drag(self, event):
        self.end_x = event.x
        self.end_y = event.y
        switcher = {
            "Rectangle": Rectangle,
            "Circle": Circle,
            "Triangle": Triangle,
            "Random Shape": RandomShape
        }
        if self.last_shape is not None:
            self.frame = self.last_shape.draw(self.frame)

        if self.selection_menu.get_selected_option() == "Random Shape":
            self.points.append((self.end_x, self.end_y))
            self.last_shape = switcher[self.selection_menu.get_selected_option()](self.points) 
            return

        self.last_shape = switcher[self.selection_menu.get_selected_option()](self.start_x, self.start_y, self.end_x, self.end_y)

    def on_mouse_release(self, event):
        self.end_x = event.x
        self.end_y = event.y
        switcher = {
            "Rectangle": Rectangle,
            "Circle": Circle,
            "Triangle": Triangle,
            "Random Shape": RandomShape
        }

        if self.selection_menu.get_selected_option() == "Random Shape":
            self.points.append((self.end_x, self.end_y))
            last_shape = switcher[self.selection_menu.get_selected_option()](self.points)
            self.points = []
            return

        last_shape = switcher[self.selection_menu.get_selected_option()](self.start_x, self.start_y, self.end_x, self.end_y)
        self.rect = last_shape.draw(self.frame)

    def save_shape(self):
        if self.last_shape is not None:
            self.shapes.append(self.last_shape)
        self.last_shape = None
    
    def run(self):
        self.window.mainloop()

    def create_image(self, x, y, image, anchor=tk.NW):
        self.canvas.create_image(x, y, image=image, anchor=anchor)
    
    def update_frame(self, frame):
        self.frame = frame
        if self.last_shape is not None:
            frame = self.last_shape.draw(frame)
        if len(self.shapes) == 0:
            return frame
        for shape in self.shapes:
            frame = shape.draw(frame)
        return frame
    
    def clear_shapes(self):
        self.shapes = []
        self.last_shape = None
    
    def get_shapes(self):
        return self.shapes
    
    def check_face_in_shapes(self, faces):
        if faces is None:
            return False
        for face in faces:
            for shape in self.shapes:
                if shape.shape_type == "Rectangle":
                    if shape.is_face_in_rectangle(face):
                        return True
                elif shape.shape_type == "Circle":
                    if shape.is_face_in_circle(face):
                        return True
                elif shape.shape_type == "Triangle":
                    if shape.is_face_in_triangle(face):
                        return True
                elif shape.shape_type == "Random Shape":
                    if shape.is_face_in_random_shape(face):
                        return True
                # return False
            # return False
        return False

def handle_face(frame: np.ndarray, face: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2, *_ = face
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    frame = cv2.circle(
        frame,
        center=(x1 + x2 // 2, y1 + y2 // 2),
        radius=(x2)//2,
        color=(0, 255, 0),
        thickness=3,
    )

    cv2.putText(
        frame,
        f"{face[14]:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (36, 255, 12),
        2,
    )

    return frame

class Shape:
    def __init__(self, shape_type):
        self.shape_type = shape_type

class Rectangle(Shape):
    def __init__(self, start_x, start_y, end_x, end_y):
        super().__init__("Rectangle")
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
    
    def __str__(self):
        return f"Rectangle coordinates: ({self.start_x}, {self.start_y}), ({self.end_x}, {self.end_y})"
    
    def draw(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 255, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return cv2.rectangle(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 255, 0), 2)
    
    def is_face_in_rectangle(self, face):
        x1, y1, x2, y2, *_ = face
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        face_circle = Circle(x1, y1, x1+x2, y1+y2)

        circle_center = face_circle.get_center()

        return self.start_x < circle_center[0] < self.end_x and self.start_y < circle_center[1] < self.end_y
        
    
    def is_face_outside_rectangle(self, face):
        return not self.is_face_in_rectangle(face)
    
    def get_center(self):
        return (self.start_x + self.end_x) // 2, (self.start_y + self.end_y) // 2
    
    def get_area(self):
        return (self.end_x - self.start_x) * (self.end_y - self.start_y)
    
    def get_perimeter(self):
        return 2 * (self.end_x - self.start_x + self.end_y - self.start_y)
    
    def get_diagonal(self):
        return ((self.end_x - self.start_x) ** 2 + (self.end_y - self.start_y) ** 2) ** 0.5
    
    def get_aspect_ratio(self):
        return (self.end_x - self.start_x) / (self.end_y - self.start_y)

class Circle(Shape):
    def __init__(self, start_x, start_y, end_x, end_y):
        super().__init__("Circle")
        self.center_x = (start_x + end_x) // 2
        self.center_y = (start_y + end_y) // 2
        self.radius = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5 // 2
    
    def __str__(self):
        return f"Circle center: ({self.center_x}, {self.center_y}), radius: {self.radius}"
    
    def draw(self, frame):
        overlay = frame.copy()
        cv2.circle(overlay, (self.center_x, self.center_y), int(self.radius), (0, 255, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return cv2.circle(frame, (self.center_x, self.center_y), int(self.radius), (0, 255, 0), 2)
    
    def is_face_in_circle(self, face):
        x1, y1, x2, y2, *_ = face
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        face_circle = Circle(x1, y1, x1+x2, y1+y2)

        circle_center = face_circle.get_center()

        return (circle_center[0] - self.center_x) ** 2 + (circle_center[1] - self.center_y) ** 2 < self.radius ** 2

    def is_face_outside_circle(self, face):
        return not self.is_face_in_circle(face)
    
    def get_area(self):
        return np.pi * self.radius ** 2
    
    def get_perimeter(self):
        return 2 * np.pi * self.radius
    
    def get_diameter(self):
        return 2 * self.radius
    
    def get_circumference(self):
        return 2 * np.pi * self.radius
    
    def get_aspect_ratio(self):
        return 1
    
    def get_center(self):
        return self.center_x, self.center_y

class Triangle(Shape):
    def __init__(self, start_x, start_y, end_x, end_y):
        super().__init__("Triangle")
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

        self.points = [(start_x, end_y), (end_x, end_y), ((start_x + end_x) // 2, start_y)]
    
    def __str__(self):
        return f"Triangle coordinates: ({self.start_x}, {self.start_y}), ({self.end_x}, {self.end_y})"
    
    def draw(self, frame):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(self.points)], (0, 255, 0))
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return cv2.polylines(frame, [np.array(self.points)], True, (0, 255, 0), 2)
    
    def is_face_in_triangle(self, face):
        x1, y1, x2, y2, *_ = face
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        face_circle = Circle(x1, y1, x1+x2, y1+y2)

        circle_center = face_circle.get_center()

        return cv2.pointPolygonTest(np.array(self.points), circle_center, False) >= 0
    
    def is_face_outside_triangle(self, face):
        return not self.is_face_in_triangle(face)
    
    def get_area(self):
        return 0
    
    def get_perimeter(self):
        return 0
    
    def get_aspect_ratio(self):
        return 0
    
class RandomShape(Shape):
    def __init__(self, points):
        super().__init__("Random Shape")
        self.points = points

    def __str__(self):
        return f"Random Shape coordinates: {self.points}"
    
    def draw(self, frame):
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(self.points)], (0, 255, 0))
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return cv2.polylines(frame, [np.array(self.points)], False, (0, 255, 0), 2)
    
    def is_face_in_random_shape(self, face):
        x1, y1, x2, y2, *_ = face
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        face_circle = Circle(x1, y1, x1+x2, y1+y2)

        circle_center = face_circle.get_center()

        return cv2.pointPolygonTest(np.array(self.points), circle_center, False) >= 0
    
    def is_face_outside_random_shape(self, face):
        return not self.is_face_in_random_shape(face)
    
    def get_area(self):
        return 0
    
    def get_perimeter(self):
        return 0
    
    def get_aspect_ratio(self):
        return 0
