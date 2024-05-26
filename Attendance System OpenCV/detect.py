import cv2
import os
import face_recognition

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
students_folder = r'F:\Instant-DA\Projects\Attendance-Detection(OpenCV)\students'

# Load known face encodings and names
known_face_encodings = []
known_face_names = []
for filename in os.listdir(students_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        image_path = os.path.join(students_folder, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0] 
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])  
registered_students = set()
absent_students = set(known_face_names)

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    for (x, y, w, h), face_encoding in zip(faces, face_encodings):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        
        for match, name in zip(matches, known_face_names):
            if match:
                if name not in registered_students:
                    registered_students.add(name)  
                    absent_students.discard(name)  
                
    for i, student_name in enumerate(registered_students):
        cv2.putText(frame, f"Registered: {student_name}", (frame.shape[1]//2, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    for i, student_name in enumerate(absent_students):
        cv2.putText(frame, f"Absent: {student_name}", (10, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
