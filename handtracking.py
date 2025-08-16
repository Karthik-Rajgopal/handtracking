import mediapipe as mp
import cv2
import numpy as np
import time
import math

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Credit card dimensions (standard)
# CREDIT_CARD_WIDTH = 8.56  # cm
# CREDIT_CARD_HEIGHT = 5.4  # cm
SPECS_WIDTH = 13.5 # cm

# Constants for drawing
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
DISTANCE_COLOR = (255, 0, 0)  # red for distances
CALIBRATION_COLOR = (0, 255, 255)  # yellow for calibration
CARD_COLOR = (255, 255, 0)  # cyan for card detection

# Calibration state
calibration_mode = False
calibration_points = []
pixels_per_cm = 0.0
calibration_complete = False

# Measurement state
measurement_mode = False
measurement_points = []

# Global variables to store landmarks
current_landmarks = []
current_handedness = []

# Depth Estimation Variables

test_depths = 0.2, 0.4  # m
pixels_per_cm_list = []

def calculate_distance_2d(point1, point2, image_width, image_height):
    """Calculate 2D pixel distance between two points"""
    x1, y1 = int(point1.x * image_width), int(point1.y * image_height)
    x2, y2 = int(point2.x * image_width), int(point2.y * image_height)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def draw_credit_card_ui(image, hand_landmarks_list, handedness_list):
    """Draw credit card specific interface"""
    height, width, _ = image.shape
    
    # Draw instructions
    if not calibration_complete:
        cv2.putText(image, "CREDIT CARD CALIBRATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, CALIBRATION_COLOR, 2)
        cv2.putText(image, "Press 'c' to start calibration", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, CALIBRATION_COLOR, 1)
        cv2.putText(image, "Hold credit card in view", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, CALIBRATION_COLOR, 1)
        cv2.putText(image, "Press '1' to mark top-left corner", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, CALIBRATION_COLOR, 1)
        cv2.putText(image, "Press '2' to mark bottom-right corner", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, CALIBRATION_COLOR, 1)
        # cv2.putText(image, f"Card size: {CREDIT_CARD_WIDTH}cm x {CREDIT_CARD_HEIGHT}cm", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.5, CARD_COLOR, 1)
    
    # Draw measurement instructions
    cv2.putText(image, "Press 'm' to start measurement mode", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, DISTANCE_COLOR, 1)
    cv2.putText(image, "Press '1' and '2' to select measurement points", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, DISTANCE_COLOR, 1)
    cv2.putText(image, "Press 'r' to reset calibration", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1)
    cv2.putText(image, "Press 'q' to quit", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1)
    
    # Show calibration status
    if calibration_complete:
        cv2.putText(image, f"Calibrated: {pixels_per_cm:.2f} pixels/cm", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1)
        cv2.putText(image, "Credit card calibration complete!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1)
    
    # Draw calibration points if in calibration mode
    if calibration_mode and len(calibration_points) > 0:
        for i, point in enumerate(calibration_points):
            x, y = int(point[0] * width), int(point[1] * height)
            cv2.circle(image, (x, y), 12, CALIBRATION_COLOR, -1)
            cv2.putText(image, str(i+1), (x+18, y+5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, CALIBRATION_COLOR, 2)
        
        # Draw rectangle for credit card
        if len(calibration_points) == 2:
            x1, y1 = int(calibration_points[0][0] * width), int(calibration_points[0][1] * height)
            x2, y2 = int(calibration_points[1][0] * width), int(calibration_points[1][1] * height)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), CARD_COLOR, 2)
            
            # Calculate and display pixel distances
            pixel_width = abs(x2 - x1)
            pixel_height = abs(y2 - y1)
            
            cv2.putText(image, f"Pixel width: {pixel_width}", (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, CALIBRATION_COLOR, 1)
            cv2.putText(image, f"Pixel height: {pixel_height}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, CALIBRATION_COLOR, 1)
            
            # Calculate pixels per cm
            pixels_per_cm_width = pixel_width / SPECS_WIDTH
            # pixels_per_cm_height = pixel_height / CREDIT_CARD_HEIGHT
            # avg_pixels_per_cm = (pixels_per_cm_width + pixels_per_cm_height) / 2
            
            cv2.putText(image, f"Pixels/cm (width): {pixels_per_cm_width:.2f}", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, CALIBRATION_COLOR, 1)
            # cv2.putText(image, f"Pixels/cm (height): {pixels_per_cm_height:.2f}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 
            #             0.5, CALIBRATION_COLOR, 1)
            # cv2.putText(image, f"Average pixels/cm: {avg_pixels_per_cm:.2f}", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 
                        # 0.5, (0, 255, 0), 1)
    
    # Draw measurement points if in measurement mode
    if measurement_mode and len(measurement_points) > 0:
        for i, point in enumerate(measurement_points):
            x, y = int(point[0] * width), int(point[1] * height)
            cv2.circle(image, (x, y), 8, DISTANCE_COLOR, -1)
            cv2.putText(image, str(i+1), (x+12, y+4), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, DISTANCE_COLOR, 2)
        
        # Draw line and calculate distance
        if len(measurement_points) == 2 and calibration_complete:
            x1, y1 = int(measurement_points[0][0] * width), int(measurement_points[0][1] * height)
            x2, y2 = int(measurement_points[1][0] * width), int(measurement_points[1][1] * height)
            cv2.line(image, (x1, y1), (x2, y2), DISTANCE_COLOR, 2)
            
            # Calculate real-world distance
            pixel_dist = calculate_distance_2d(
                type('Point', (), {'x': measurement_points[0][0], 'y': measurement_points[0][1]})(),
                type('Point', (), {'x': measurement_points[1][0], 'y': measurement_points[1][1]})(),
                width, height
            )
            real_distance = pixel_dist / pixels_per_cm
            
            # Display distance
            cv2.putText(image, f"Distance: {real_distance:.1f} cm", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, DISTANCE_COLOR, 2)
            cv2.putText(image, f"({real_distance/2.54:.1f} inches)", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, DISTANCE_COLOR, 1)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Store landmarks globally for key handling
    global current_landmarks, current_handedness
    current_landmarks = hand_landmarks_list
    current_handedness = handedness_list

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

    # Draw UI elements
    draw_credit_card_ui(annotated_image, hand_landmarks_list, handedness_list)

    return annotated_image

# Callback function for async detection
def process_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global annotated_image
    if result.hand_landmarks:  # Check if landmarks are detected
        annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)

def handle_key_press(key):
    """Handle keyboard input for calibration and measurement"""
    global calibration_mode, measurement_mode, calibration_points, measurement_points, pixels_per_cm_list
    global pixels_per_cm, calibration_complete
    
    if key == ord('c'):
        # Toggle calibration mode
        calibration_mode = not calibration_mode
        measurement_mode = False
        calibration_points = []
        measurement_points = []
        pixels_per_cm_list = []
        print("Credit card calibration mode:", "ON" if calibration_mode else "OFF")
        
    elif key == ord('m'):
        # Toggle measurement mode
        measurement_mode = not measurement_mode
        calibration_mode = False
        calibration_points = []
        measurement_points = []
        print("Measurement mode:", "ON" if measurement_mode else "OFF")
        slope = (pixels_per_cm_list[1] - pixels_per_cm_list[0])/(test_depths[1] - test_depths[0])
        intercept = pixels_per_cm_list[1] - slope * test_depths[1]
        
    elif key == ord('r'):
        # Reset calibration
        calibration_complete = False
        pixels_per_cm = 0.0
        calibration_points = []
        measurement_points = []
        print("Calibration reset")
        
    elif key == ord('1') and (calibration_mode or measurement_mode):
        # Set first point using index finger tip
        if current_landmarks and len(current_landmarks) > 0:
            point = current_landmarks[0][8]  # index finger tip
            if calibration_mode:
                calibration_points = [(point.x, point.y)]
                print("Credit card corner 1 set (top-left)")
            elif measurement_mode:
                measurement_points = [(point.x, point.y)]
                print("Measurement point 1 set")
                
    elif key == ord('2') and (calibration_mode or measurement_mode):
        # Set second point using index finger tip
        if current_landmarks and len(current_landmarks) > 0:
            point = current_landmarks[0][8]  # index finger tip
            if calibration_mode and len(calibration_points) == 1:
                calibration_points.append((point.x, point.y))
                print("Credit card corner 2 set (bottom-right)")
                
                # Auto-calibrate using credit card dimensions
                height, width = 480, 640  # approximate image size
                x1, y1 = int(calibration_points[0][0] * width), int(calibration_points[0][1] * height)
                x2, y2 = int(calibration_points[1][0] * width), int(calibration_points[1][1] * height)
                
                pixel_width = abs(x2 - x1)
                pixel_height = abs(y2 - y1)
                
                # Calculate pixels per cm using both dimensions
                pixels_per_cm_width = pixel_width / SPECS_WIDTH
                # pixels_per_cm = (pixels_per_cm_width + pixels_per_cm_height) / 2
                pixels_per_cm_list = [pixels_per_cm_width]
                
                calibration_complete = True
                print(f"Credit card calibration complete!")
                print(f"Width: {pixel_width}px = {SPECS_WIDTH}cm ({pixels_per_cm_width:.2f} px/cm)")
                # print(f"Height: {pixel_height}px = {CREDIT_CARD_HEIGHT}cm ({pixels_per_cm_height:.2f} px/cm)")
                print(f"Average: {pixels_per_cm:.2f} pixels/cm")
                    
            elif measurement_mode and len(measurement_points) == 1:
                measurement_points.append((point.x, point.y))
                print("Measurement point 2 set")

    elif key == ord('3') and (calibration_mode or measurement_mode):
        # Set first point using index finger tip
        if current_landmarks and len(current_landmarks) > 0:
            point = current_landmarks[0][8]  # index finger tip
            if calibration_mode:
                calibration_points.append((point.x, point.y))
                print("Credit card corner 3 set (top-left)")
            elif measurement_mode:
                measurement_points.append((point.x, point.y))
                print("Measurement point 3 set")
    
    elif key == ord('4') and (calibration_mode or measurement_mode):
        # Set third point using index finger tip
        if current_landmarks and len(current_landmarks) > 0:
            point = current_landmarks[0][8]  # index finger tip
            if calibration_mode and len(calibration_points) == 3:
                calibration_points.append((point.x, point.y))
                print("Credit card corner 4 set (bottom-right)")
                
                # Auto-calibrate using credit card dimensions
                height, width = 480, 640  # approximate image size
                x1, y1 = int(calibration_points[2][0] * width), int(calibration_points[2][1] * height)
                x2, y2 = int(calibration_points[3][0] * width), int(calibration_points[3][1] * height)
                
                pixel_width = abs(x2 - x1)
                pixel_height = abs(y2 - y1)
                
                # Calculate pixels per cm using both dimensions
                pixels_per_cm_width = pixel_width / SPECS_WIDTH
                pixels_per_cm_list.append(pixels_per_cm_width)
                # pixels_per_cm = (pixels_per_cm_width + pixels_per_cm_height) / 2
                
                calibration_complete = True
                print(f"Credit card calibration complete!")
                print(f"Width: {pixel_width}px = {SPECS_WIDTH}cm ({pixels_per_cm_width:.2f} px/cm)")
                # print(f"Height: {pixel_height}px = {CREDIT_CARD_HEIGHT}cm ({pixels_per_cm_height:.2f} px/cm)")
                print(f"Average: {pixels_per_cm:.2f} pixels/cm")
                    
            elif measurement_mode and len(measurement_points) == 3:
                measurement_points.append((point.x, point.y))
                print("Measurement point 4 set")

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

    print("Credit Card Hand Tracking with Measurement")
    print("Controls:")
    print("  'c' - Toggle credit card calibration mode")
    print("  'm' - Toggle measurement mode")
    print("  '1' - Set first point (index finger tip)")
    print("  '2' - Set second point (index finger tip)")
    print("  '3' - Set first point (index finger tip) - More Distance")
    print("  '4' - Set second point (index finger tip) - More Distance")
    print("  'r' - Reset calibration")
    print("  'q' - Quit")
    print("\nCredit Card Calibration Instructions:")
    print("1. Press 'c' to enter calibration mode")
    print("2. Hold a credit card in view")
    print("3. Press '1' to mark the top-left corner of the card")
    print("4. Press '2' to mark the bottom-right corner of the card")
    print("5. The system will auto-calibrate using standard card dimensions")
    print("6. Press 'm' to start measuring!")

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
            cv2.imshow('Credit Card Hand Tracking with Measurement', annotated_image_bgr)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('c'), ord('m'), ord('1'), ord('2'), ord('r'), ord('3'), ord('4')]:
            handle_key_press(key)
        


    cap.release()
    cv2.destroyAllWindows()
