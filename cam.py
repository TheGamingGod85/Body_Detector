import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# Initialize MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Face Mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Pose Detection
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_frame(frame):
    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    results_hands = hands.process(rgb_frame)

    # Process the image with MediaPipe Face Mesh
    results_face_mesh = face_mesh.process(rgb_frame)

    # Process the image with MediaPipe Pose Detection
    results_pose = pose.process(rgb_frame)

    # Draw hand landmarks and connections on the image
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw face mesh landmarks and connections on the image
    if results_face_mesh.multi_face_landmarks:
        for face_landmarks in results_face_mesh.multi_face_landmarks:
            # Draw face mesh landmarks
            for idx, landmark in enumerate(face_landmarks.landmark):
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
                
            # Draw face mesh connections
            for connection in mp_face_mesh.FACEMESH_CONTOURS:
                pt1 = (int(face_landmarks.landmark[connection[0]].x * width), int(face_landmarks.landmark[connection[0]].y * height))
                pt2 = (int(face_landmarks.landmark[connection[1]].x * width), int(face_landmarks.landmark[connection[1]].y * height))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 1)

    # Draw pose landmarks and connections on the image, excluding face landmarks
    if results_pose.pose_landmarks:
        for idx, landmark in enumerate(results_pose.pose_landmarks.landmark):
            # Exclude face landmarks (use indices 0-11 and 23-32 from pose_landmarks)
            if idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    
    return frame
