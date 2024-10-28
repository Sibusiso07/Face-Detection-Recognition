import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import face_recognition
from mtcnn import MTCNN
import threading

# Initialize known face encodings and names
known_image_1 = face_recognition.load_image_file('known_faces/Bailu.jpg')
known_image_2 = face_recognition.load_image_file('known_faces/Keanu.jpg')
known_encoding_1 = face_recognition.face_encodings(known_image_1)[0]
known_encoding_2 = face_recognition.face_encodings(known_image_2)[0]
known_face_encodings = [known_encoding_1, known_encoding_2]
known_face_names = ["Bailu", "Keanu"]

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection & Recognition")
        self.root.geometry("800x600")

        # Heading
        self.title_label = tk.Label(root, text="Face Detection & Recognition", font=("Arial", 20))
        self.title_label.pack(pady=10)

        # Left frame for uploaded photo
        self.left_frame = tk.Frame(root, width=400, height=500, bg="white")
        self.left_frame.pack(side="left", fill="both", expand=True)
        self.image_label = tk.Label(self.left_frame, text="Upload Image Here", font=("Arial", 12))
        self.image_label.pack(pady=20)

        # Right frame for loading/processed image
        self.right_frame = tk.Frame(root, width=400, height=500, bg="gray")
        self.right_frame.pack(side="right", fill="both", expand=True)
        self.result_label = tk.Label(self.right_frame, text="Loading...", font=("Arial", 12), fg="white")
        self.result_label.pack(pady=20)

        # Result text area for detected names
        self.result_text = tk.StringVar()
        self.result_label_text = tk.Label(self.right_frame, textvariable=self.result_text, font=("Arial", 12), fg="white", bg="gray")
        self.result_label_text.pack()

        # Upload and process button
        self.upload_button = tk.Button(self.left_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)
        self.process_button = tk.Button(self.left_frame, text="Search", command=self.process_image, state="disabled")
        self.process_button.pack(pady=10)

        self.image_path = None

    def upload_image(self):
        # Select and display image
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            img = Image.open(self.image_path)
            img.thumbnail((300, 300))
            img = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img)
            self.image_label.image = img
            self.process_button.config(state="normal")
            self.result_text.set("")  # Clear previous results

    def process_image(self):
        if not self.image_path:
            return
        self.result_label.config(text="Processing...")
        self.process_button.config(state="disabled")
        threading.Thread(target=self.run_face_recognition).start()

    def run_face_recognition(self):
        # Load and process image
        image = cv2.imread(self.image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detector = MTCNN()
        faces = detector.detect_faces(image_rgb)
        face_locations = [(face['box'][1], face['box'][0] + face['box'][2], face['box'][1] + face['box'][3], face['box'][0]) for face in faces]
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

        names = []
        unknown_count = 0

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            else:
                unknown_count += 1
            names.append(name)

        # Display names and count unknowns
        names_text = ", ".join(name for name in names if name != "Unknown")
        unknown_text = f"{unknown_count} unknown detected" if unknown_count else ""
        self.result_text.set(f"Detected: {names_text} {unknown_text}")

        # Draw bounding boxes and labels
        for (top, right, bottom, left), name in zip(face_locations, names):
            color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
            cv2.rectangle(image_rgb, (left, top), (right, bottom), color, 2)
            cv2.putText(image_rgb, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Update result image in GUI
        processed_img = Image.fromarray(image_rgb)
        processed_img.thumbnail((300, 300))
        processed_img = ImageTk.PhotoImage(processed_img)
        self.result_label.config(image=processed_img, text="")
        self.result_label.image = processed_img
        self.process_button.config(state="normal")

root = tk.Tk()
app = FaceRecognitionApp(root)
root.mainloop()