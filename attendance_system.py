import cv2
import dlib
import numpy as np
import pandas as pd

# Paths to the Excel and model files (ensure to update the paths)
excel_path = r"attendance.xlsx"
shape_predictor_path = r'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = r'dlib_face_recognition_resnet_model_v1.dat'

# Initialize Dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
face_rec = dlib.face_recognition_model_v1(face_rec_model_path)

def get_face_encoding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    face = faces[0]
    shape = predictor(gray, face)
    encoding = np.array(face_rec.compute_face_descriptor(image, shape))
    return encoding

# List of student images and their corresponding labels
student_images = [
    ('Student1.jpg', 'Kamalesh'),
    ('Student2.jpg', 'Vinston'),
    ('Student3.jpg', 'Vinayak'),
    ('Student4.jpg', 'Pranav'),
    ('Student5.jpg', 'Adithya')
]

# Load and encode each student image along with its label
student_encodings = []
student_labels = []
for image_path, label in student_images:
    image = cv2.imread(image_path)
    encoding = get_face_encoding(image)
    if encoding is not None:
        student_encodings.append(encoding)
        student_labels.append(label)

# Initialize attendance DataFrame and count dictionary
attendance_df = pd.DataFrame(student_labels, columns=['Student'])
attendance_df['Status'] = 'A'  # Default to 'A' for absent
count = {label: 0 for label in student_labels}  # Dictionary to count detections for each student

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        face_shape = predictor(gray, face)
        face_encoding = np.array(face_rec.compute_face_descriptor(frame, face_shape))

        # Check each detected face encoding against all student encodings
        matched_label = "Unknown"
        for student_encoding, label in zip(student_encodings, student_labels):
            if np.linalg.norm(student_encoding - face_encoding) < 0.52:
                matched_label = label
                attendance_df.loc[attendance_df['Student'] == label, 'Status'] = 'P'
                break

        # Display matched label
        cv2.putText(frame, matched_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    (0, 255, 0) if matched_label != "Unknown" else (0, 0, 255), 2)

    cv2.imshow('Face Detection', frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save updated attendance to Excel
attendance_df.to_excel(excel_path, index=False)
print("Attendance has been recorded.")
