# import cv2
# import matplotlib.pyplot as plt
# from mtcnn import MTCNN
#
# # Read the image using cv2
# image = cv2.imread('uploads/1.jpg')
#
# # Convert the image from BGR to RGB format as MTCNN works with RGB images
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Initialize MTCNN face detector
# detector = MTCNN()
#
# # Perform face detection
# faces = detector.detect_faces(image_rgb)
#
# # Drawing the bounding box
# for face in faces:
#     x, y, width, height = face['box']
#     cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 4)
#
# # Displaying the image with bounding boxes
# plt.figure(figsize=(20, 10))
# plt.imshow(image_rgb)
# plt.axis('off')  # Optional: Hide axis
# plt.show()
#


# import cv2
# import matplotlib.pyplot as plt
# from mtcnn import MTCNN
# import face_recognition
#
# # Step 1: Load the known images and encode them
# # Known face images and their labels (assume we have 2 known faces)
# known_image_1 = face_recognition.load_image_file('known_faces/Bailu.jpg')
# known_image_2 = face_recognition.load_image_file('known_faces/Keanu.jpg')
#
# # Encode the known face images to create face encodings
# known_encoding_1 = face_recognition.face_encodings(known_image_1)[0]
# known_encoding_2 = face_recognition.face_encodings(known_image_2)[0]
#
# # Create a list of known face encodings and labels
# known_face_encodings = [known_encoding_1, known_encoding_2]
# known_face_names = ["Bailu", "Keanu"]
#
# # Step 2: Load the test image and convert it to RGB (MTCNN requires RGB images)
# image = cv2.imread('uploads/9.jpg')
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Step 3: Initialize MTCNN detector and detect faces
# detector = MTCNN()
# faces = detector.detect_faces(image_rgb)
#
# # List to hold face locations (top, right, bottom, left)
# face_locations = []
# for face in faces:
#     x, y, width, height = face['box']
#     # Append in face_recognition format (top, right, bottom, left)
#     face_locations.append((y, x + width, y + height, x))
#
# # Step 4: Recognize faces using face_recognition library
# # Get the face encodings for the detected faces in the image
# face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
#
# # Loop over each detected face and compare it with known faces
# for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#     # Compare the detected face encoding with known face encodings
#     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#     name = "Unknown"  # Default name if no match is found
#
#     # If a match was found, use the first match
#     if True in matches:
#         matched_idx = matches.index(True)
#         name = known_face_names[matched_idx]
#
#     # Step 5: Draw bounding boxes and label the faces
#     if True in matches:
#         cv2.rectangle(image_rgb, (left, top), (right, bottom), (0, 255, 0), 4)
#     else:
#         cv2.rectangle(image_rgb, (left, top), (right, bottom), (255, 0, 0), 4)
#     cv2.putText(image_rgb, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#
# # Step 6: Display the result image with recognized faces
# plt.figure(figsize=(20, 10))
# plt.imshow(image_rgb)
# plt.axis('off')  # Hide axis
# plt.show()


import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import face_recognition
from mtcnn import MTCNN
import numpy as np
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

