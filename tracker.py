import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Face Mesh
# We only need the basic mesh, but refine_landmarks gives better iris tracking if you want that later.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 2. Access Webcam (Index 0 is default)
cap = cv2.VideoCapture(0)

# --- MACOS SPECIFIC FIX ---
# This ensures the window thread runs in the background so it doesn't freeze.
cv2.startWindowThread() 
cv2.namedWindow('Forehead Tracker', cv2.WINDOW_NORMAL)
# --------------------------

print("System Active. Press 'ESC' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 3. Pre-Processing (Optimization)
    # Mark as not writeable to pass by reference (faster memory access)
    image.flags.writeable = False
    # Convert BGR (OpenCV standard) to RGB (MediaPipe standard)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 4. Post-Processing
    image.flags.writeable = True
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get dimensions to normalize coordinates
            h, w, _ = image.shape
            
            # --- EXTRACT DATA POINT: LANDMARK #10 (Forehead Center) ---
            lm_10 = face_landmarks.landmark[10]
            
            # Convert normalized (0.0-1.0) to pixel coordinates
            cx, cy = int(lm_10.x * w), int(lm_10.y * h)

            # Visual Feedback: Draw Green Dot
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            
            # Data Feedback: Print Coordinates on Screen
            coord_text = f"X: {cx}, Y: {cy}"
            cv2.putText(image, coord_text, (cx + 10, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Console Log for Engineering Analysis (Optional)
            # print(f"Log: {cx}, {cy}")

    # 5. Display
    cv2.imshow('Forehead Tracker', image)

    # --- MACOS SPECIFIC FIX ---
    # waitKey(1) is faster than waitKey(5) and prevents lag on Mac
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
# Add this specifically for Mac to close windows cleanly
cv2.waitKey(1)
#here are the key commands to run this-> source venv/bin/activate 
#Then run this-> python tracker.py
#control + C to stop the code 