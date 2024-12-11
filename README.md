# Face Recognition-Based Attendance System

This project leverages **Dlib** and **OpenCV** to create an automated attendance system that uses face recognition to mark students as present. It captures faces from a webcam feed, matches them with pre-encoded student images, and updates an attendance sheet stored in an Excel file. This solution is designed for use in environments such as classrooms to automate the process of marking attendance.

## Features
- Face detection using Dlib's frontal face detector.
- Face recognition using Dlib's pre-trained face recognition model.
- Real-time video feed for capturing and matching faces.
- Attendance is marked as **Present** (P) for recognized faces and **Absent** (A) for unknown faces.
- The attendance data is saved in an **Excel file** for record-keeping.

## Prerequisites
Before running the project, make sure you have the following installed:
- Python 3.x
- OpenCV (`opencv-python` package)
- Dlib (`dlib` package)
- Pre-trained `dlib` models:
   - `shape_predictor_68_face_landmarks.dat`
   - `dlib_face_recognition_resnet_model_v1.dat`
- Pandas (`pandas` package)
- Numpy (`numpy` package)

### Installing dependencies:
1. Clone the repository or download the code.
2. Install the required dependencies:
  -  ```pip install opencv-python dlib pandas numpy```
3. Pre-trained `dlib` models:
   - `shape_predictor_68_face_landmarks.dat`
   - `dlib_face_recognition_resnet_model_v1.dat`

### Usage :

-Place the images of the students you wish to recognize in the same directory as the script or modify the paths in the code accordingly.

-Update the list of student images in the code with the filenames of the student photos and their labels.

-Run the script:
-  '''python attendance_system.py'''

-The script will open the webcam and start detecting faces. It will attempt to match each face with the pre-stored images and mark attendance as Present for recognized faces.

-Press 'q' to quit the program.

-After the session ends, the attendance will be saved in the Excel file, where each student's attendance status is updated.


