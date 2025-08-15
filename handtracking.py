import mediapipe as mp
import cv2
import numpy as np
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants for drawing
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# Define the drawing function
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through detected hands
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get bounding box for text
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness text
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# Callback function for async detection
def process_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print((result.hand_landmarks[0][12].x - result.hand_landmarks[0][0].x) ** 2 + (result.hand_landmarks[0][12].y - result.hand_landmarks[0][0].y) ** 2 + (result.hand_landmarks[0][12].z - result.hand_landmarks[0][0].z) ** 2)
    # print(result.hand_world_landmarks[0][9].x, result.hand_world_landmarks[0][9].y, result.hand_world_landmarks[0][9].z)

    
    # x_mean = 0.0
    # y_mean = 0.0
    # z_mean = 0.0
    # for i in [0, 1, 5, 9, 13, 17]:
    #     x_mean += result.hand_world_landmarks[0][i].x
    #     y_mean += result.hand_world_landmarks[0][i].y
    #     z_mean += result.hand_world_landmarks[0][i].z
    # print(x_mean/6, y_mean/6, z_mean/6)    

    # x_coords = [result.hand_world_landmarks[0][i].x for i in [0, 4, 8, 12, 16, 20]]
    # y_coords = [result.hand_world_landmarks[0][i].y for i in [0, 4, 8, 12, 16, 20]]
    # z_coords = [result.hand_world_landmarks[0][i].z for i in [0, 4, 8, 12, 16, 20]]
    # x_min = min(x_coords)
    # x_max = max(x_coords)
    # y_min = min(y_coords)
    # y_max = max(y_coords)
    # z_min = min(z_coords)
    # z_max = max(z_coords)

    # print(np.mean([x_max, x_min]), np.mean([y_max, y_min]), np.mean([z_max, z_min]))

    global annotated_image
    if result.hand_landmarks:  # Check if landmarks are detected
        annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)

# Initialize HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=process_result
)

# Global variable to store the annotated image
annotated_image = None

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process frame with timestamp
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # Display the annotated image if available
        if annotated_image is not None:
            # Convert back to BGR for OpenCV display
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Hand Landmarker', annotated_image_bgr)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()