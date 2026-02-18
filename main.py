#basic initialization(toooooooooo much librariess i am so tireedddd)(this is acteully so annoying i have no time to organize code so everything in rondom place and beacuse of that erro keep happening aaaaaaaaaaah)
import joblib
import cv2
import csv
import os
import keras
import mediapipe as mp
from math import dist
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from collections import deque
import time
import winsound
import pyautogui 
#idk what this do but when i run teh thing it says to put this so yeaaaaah
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#ask user which mode they want
MODE = input("Which mode Driver_Safety or Mouse (very case sensetive)")
#AI yap loading AI stuff
model = tf.keras.models.load_model("gaze_direction_model.keras")
label_classes = np.load("gaze_label_classes.npy",allow_pickle= True)
scaler_blink_mean = np.load("blink_ear_scaler_mean.npy", allow_pickle= True) #Pickle is Python's way of saving complex data (pickle realy???)
scaler_blink_scale = np.load("blink_ear_scaler_scale.npy",allow_pickle= True) 
scaler_blink = StandardScaler()
scaler_blink.mean_ = scaler_blink_mean
scaler_blink.scale_ = scaler_blink_scale
scaler_blink.var_ = scaler_blink_scale ** 2
scaler_blink.n_features_in_ = 5
model_blink = tf.keras.models.load_model("blink_ear_aggregate_model.keras")
blink_classes = np.load("blink_ear_label_classes.npy", allow_pickle= True)

#safe gured if model does not work
predicted_label = "idk"
predicted_label_blink = "why would i kow"
#stuff for alert
alert_cooldown = 0  
ALERT_COOLDOWN_FRAMES = 60  
#deqe
EAR_WINDOW_SIZE = 100
EAR_history = deque(maxlen=EAR_WINDOW_SIZE)
MAX_BLINK = 100
blink_timestamps = deque(maxlen=MAX_BLINK)
BLINK_DURATION_WINDOW = 100
blink_durations_history = deque(maxlen=BLINK_DURATION_WINDOW)
longest_eye_close = 0
current_eye_close = 0
# Track how long user has been in alerting state
not_center_start_time = None  
tired_start_time = None  
ALERT_DELAY_SECONDS = 5  
# Load scaler parameters and recreate scaler
scaler_mean = np.load("gaze_scaler_mean.npy", allow_pickle= True)
scaler_scale = np.load("gaze_scaler_scale.npy", allow_pickle= True)
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale
scaler.var_ = scaler_scale ** 2  
scaler.n_features_in_ = 4        
webcam = cv2.VideoCapture(0)
fps = webcam.get(cv2.CAP_PROP_FPS)
#saving data to train ai later
csv_file_path = 'datafoeAI1.csv'
csv_header = [
    'Frame', 'Left_EAR', 'Right_EAR', 'Avg_EAR', 'Blinks', 'Blink_Duration',
    'Yaw', 'Pitch', 'Roll', 'Horizontal_Gaze', 'Vertical_Gaze',
    'iris_x_norm', 'iris_y_norm', 'eye_width', 'eye_height', 'Label',
    'Blink_Rate', 'Avg_Blink_Duration', 'EAR_Mean', 'EAR_Variance', 'Longest_Eye_Closure'
]
# Mouse control settings(you ahv eto do this for the libraries)
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
pyautogui.FAILSAFE = False  
pyautogui.PAUSE = 0  
SMOOTHING_FACTOR = 0.15  
prev_mouse_x = SCREEN_WIDTH // 2
prev_mouse_y = SCREEN_HEIGHT // 2

write_header = not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0
csv_file = open(csv_file_path, 'a', newline='')
csv_writer = csv.writer(csv_file)
if write_header:
    csv_writer.writerow(csv_header)
total_frame = 0

#just for safity so nothing breaks when there is fps eror(idk if this really needed but i will put it anywyas i mean i have a good pc i know some of you fellas are broke and dont have that much money lol)
if fps == 0 or fps < 10 or fps > 120:
    fps = 30


#setting some varaibles cuz me cool(This initializes the FaceMesh detector.)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
stop = False
closed_frames = 0
blinks = 0
blinking = False
HORIZONTAL_HISTORY = deque(maxlen=15)  # last 15 horizontal gaze values
VERTICAL_HISTORY = deque(maxlen=15)    # last 15 vertical gaze values

#landmark LOCATIONS for eye tracking
leftEye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
rightEye = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

#landmark LOCATIONS for iris tracking
leftIris = [468, 469, 470, 471, 472]
rightIris = [473, 474, 475, 476, 477]
left_eye_boundaries = [33, 133, 159, 160, 161, 145, 144, 153]
Right_eye_boundaries = [362, 263, 386, 387, 388, 374, 373, 380]

#landmark LOCATIONS for headpose tracking
head_pose_landmarks_2d_index = [1, 152, 33, 263, 61, 291 ]
head_pose_landmarks_3d_cordinates = [(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)]

def dist_2d(p1, p2):
    #basiclly gets the 3d points and make it 2d for calculations in the future
    return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

def getEyeMetrics(eyePoints):
    #using the EAR formula(it basicly detects how closed your eyes are )
    A = dist(eyePoints[12], eyePoints[4])
    B = dist(eyePoints[11], eyePoints[3])
    C = dist(eyePoints[0], eyePoints[8])
    metric = (A + B) / (2.0 * C)
    
    return metric

#blink detection cuz why not
def detect_Blink(EAR,fps):
#honestly it gives eror with out global so i just put global no idea why that is
    global blinking, closed_frames, blinks, blink_timestamps
    global current_eye_close, longest_eye_close
    global blink_durations_history
    
    if EAR <=0.31:
        if not blinking:
            blinking = True
            current_eye_close = 0.0
        closed_frames = closed_frames+1
        current_eye_close = closed_frames / fps
    else:
        if current_eye_close > longest_eye_close:
            longest_eye_close = current_eye_close
        current_eye_close = 0  
        
        if blinking == True:
            blink_duration_sec = closed_frames / fps

            if blink_duration_sec >= 0.05:
                blinks = blinks+1
                #for AI 2
                blink_timestamps.append(time.time())
                blink_durations_history.append(blink_duration_sec) 
            closed_frames = 0
            blinking = False


    blink_duration = closed_frames / fps if blinking else 0
    return blinks, blink_duration
#calculating blink rate
def calculate_blink_rate(blink_timestamps, window_seconds=10):
    current_time = time.time()
    
    # Keep only blinks inside the window
    while blink_timestamps and blink_timestamps[0] < current_time - window_seconds:
        blink_timestamps.popleft()
    return len(blink_timestamps) / window_seconds  
#why not
def average_blink_duration(blink_durations_history):
    if len(blink_durations_history) == 0:
        return 0
    return np.mean(blink_durations_history)

def head_Pose_Estimation(camera_matrix, points, In_real_life_points, dist_coeffs ):
    
    sucess, rotation_vector, translation_vector = cv2.solvePnP(In_real_life_points, points, camera_matrix, dist_coeffs)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Calculate Euler angles from rotation matrix #Euler angles are a set of three angles that describe the orientation of a rigid body in 3D space by defining a sequence of three successive rotations around the object's principal axes (yap of euler)
    val = -rotation_matrix[2, 0]
    if val > 1.0: val = 1.0
    if val < -1.0: val = -1.0
    pitch = np.arcsin(val)
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    # Convert radians to degrees (omg me know trig me so cool)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)

    return yaw, pitch, roll
#gaze detection wouhhhh
def gaze_detection(lefteye_points, Righteye_points, leftiris_points, rightiris_points, left_eye_boundaries_points, Right_eye_boundaries_points, debug=False):
    epsilon = 1e-5  # tiny number to avoid zero division
    #the yap of the century is aobut to begin get ready(this is torture disguised as equations)
    left_eye_width = dist_2d(lefteye_points[0], lefteye_points[1]) + epsilon
    right_eye_width = dist_2d(Righteye_points[0], Righteye_points[1]) + epsilon
    # Average bottom and top points of eyes for vertical metrics
    average_right_eye_bottom = np.mean([Righteye_points[5], Righteye_points[6], Righteye_points[7]], axis=0)
    average_left_eye_bottom = np.mean([lefteye_points[5], lefteye_points[6], lefteye_points[7]], axis=0)
    average_right_eye_top = np.mean([Righteye_points[2], Righteye_points[3], Righteye_points[4]], axis=0)
    average_left_eye_top = np.mean([lefteye_points[2], lefteye_points[3], lefteye_points[4]], axis=0)
    left_eye_height = dist_2d(average_left_eye_bottom, average_left_eye_top) + epsilon
    right_eye_height = dist_2d(average_right_eye_bottom, average_right_eye_top) + epsilon
    # Eye center y
    left_eye_center_y = (average_left_eye_top[1] + average_left_eye_bottom[1]) / 2
    right_eye_center_y = (average_right_eye_top[1] + average_right_eye_bottom[1]) / 2
    # Iris centers
    left_iris_center = np.mean(np.array(leftiris_points)[:, :2], axis=0)
    right_iris_center = np.mean(np.array(rightiris_points)[:, :2], axis=0)
    # Horizontal gaze position
    left_horizontal_pos = (left_iris_center[0] - lefteye_points[0][0]) / left_eye_width
    right_horizontal_pos = (right_iris_center[0] - Righteye_points[0][0]) / right_eye_width
    # Vertical gaze position normalized relative to eye center
    left_vertical_pos = (average_left_eye_bottom[1] - left_iris_center[1]) / left_eye_height
    right_vertical_pos = (average_right_eye_bottom[1] - right_iris_center[1]) / right_eye_height
    # Average horizontal and vertical gaze positions
    horizontal_gaze = (left_horizontal_pos + right_horizontal_pos) / 2
    vertical_gaze = (left_vertical_pos + right_vertical_pos) / 2
    # Normalize left eye iris position for features
    left_iris_x_norm = (left_iris_center[0] - left_eye_boundaries_points[0][0]) / left_eye_width
    left_iris_y_norm = (left_iris_center[1] - left_eye_center_y) / left_eye_height
    # Normalize right eye iris position for features
    right_iris_x_norm = (right_iris_center[0] - Right_eye_boundaries_points[0][0]) / right_eye_width
    right_iris_y_norm = (right_iris_center[1] - right_eye_center_y) / right_eye_height
    # Average normalized iris positions for both eyes
    iris_x_norm = (left_iris_x_norm + right_iris_x_norm) / 2
    iris_y_norm = (left_iris_y_norm + right_iris_y_norm) / 2
    # Use left eye width and height for feature consisten cy
    eye_width = left_eye_width
    eye_height = left_eye_height
    # Clip horizontal gaze between 0 and 1 (sometimes can go out of range)
    horizontal_gaze = np.clip(horizontal_gaze, 0, 1)
    vertical_gaze = np.clip(vertical_gaze, 0, 1)
    horizontal_gaze_raw = np.clip(horizontal_gaze, 0, 1)
    vertical_gaze_raw = np.clip(vertical_gaze, 0, 1)
    # Append to history for smoothing
    HORIZONTAL_HISTORY.append(horizontal_gaze)
    VERTICAL_HISTORY.append(vertical_gaze)
    # Smoothed gaze values
    horizontal_gaze_smooth = sum(HORIZONTAL_HISTORY) / len(HORIZONTAL_HISTORY)
    vertical_gaze_smooth = sum(VERTICAL_HISTORY) / len(VERTICAL_HISTORY) 
    #debugging tesxt prob gonna deleate later
    #if debug:
        #print(f"Left iris center: {left_iris_center}, Right iris center: {right_iris_center}")
        #print(f"Left eye width: {left_eye_width:.3f}, height: {left_eye_height:.3f}")
        #print(f"Right eye width: {right_eye_width:.3f}, height: {right_eye_height:.3f}")
        #print(f"Left eye center y: {left_eye_center_y:.3f}, Right eye center y: {right_eye_center_y:.3f}")
        #print(f"Horizontal gaze raw: {horizontal_gaze:.3f}, smoothed: {horizontal_gaze_smooth:.3f}")
        #print(f"Vertical gaze raw: {vertical_gaze:.3f}, smoothed: {vertical_gaze_smooth:.3f}")

    return horizontal_gaze_smooth, vertical_gaze_smooth, iris_x_norm, iris_y_norm, eye_width, eye_height

def map_gaze_to_screen(horizontal_gaze, vertical_gaze, smoothing=SMOOTHING_FACTOR):
    global prev_mouse_x, prev_mouse_y
    
    # Flip horizontal so looking left moves cursor left(invert stuff)
    target_x = int((1 - horizontal_gaze) * SCREEN_WIDTH)
    target_y = int(vertical_gaze * SCREEN_HEIGHT)
    
    # Smooth the movement(so more stable)
    smooth_x = int(prev_mouse_x * (1 - smoothing) + target_x * smoothing)
    smooth_y = int(prev_mouse_y * (1 - smoothing) + target_y * smoothing)
    
    # Update previous position
    prev_mouse_x = smooth_x
    prev_mouse_y = smooth_y
    
    return smooth_x, smooth_y

def getPoints(table): #this thing is so frickin compicatedddddd
    return [landmarks_list[i] for i in table]

#hoenstly idk why i need this but you need it(it is setup part dont blame me)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
) as face_mesh:

    # getting frames forever
    while stop == False:
        key = cv2.waitKey(1) & 0xFF
        there, frame = webcam.read()  # reads camera every frame
        # default values for when no faces is detected cuz it cause eror for some reason (figuring out how to debug this took a whole frickin day)
        label = None  
        #media pipe data type conversion
        if there == True:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)  # overlays landmarks over the frame (yap)

                    height, width, _ = frame.shape
                    # camera matrix def - (Given 3D points of a face (nose, chin, eye corners, etc.) and their 2D positions in the camera image, how is the head rotated and positioned in 3D space? that is why we need this honestly hurts my brain)
                    focal_length = width  # how zoomed in the camera is
                    center = (width / 2, height / 2) 
                    camera_matrix = np.array([  #
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]
                    ], dtype="double")

                    dist_coeffs = np.zeros((4, 1))  # assume no distorsion
                    landmarks_list = []
                    for lm in face_landmarks.landmark:
                        x_px = int(lm.x * width)
                        y_px = int(lm.y * height)
                        z = lm.z
                        landmarks_list.append((x_px, y_px, z))
                    head_pose_landmarks_2d_index_points = getPoints(head_pose_landmarks_2d_index)

                    # ACTUAL landmarks for eye tracking
                    left_eye_points = getPoints(leftEye)
                    right_eye_points = getPoints(rightEye)

                    # ACTUAL landmarks for iris tracking
                    left_iris_points = getPoints(leftIris)
                    right_iris_points = getPoints(rightIris)
                    left_eye_boundaries_points = getPoints(left_eye_boundaries)
                    Right_eye_boundaries_points = getPoints(Right_eye_boundaries)

                    # getting the EAR
                    left_EAR = getEyeMetrics(left_eye_points)
                    right_EAR = getEyeMetrics(right_eye_points)

                    # GETTING TO USE THE BLINK FUNCTION WOUAHHH
                    average_EAR = (left_EAR + right_EAR) / 2
                    blinks, blink_duration = detect_Blink(average_EAR, fps)

                    # setting up list for Ai 2 wouhhhh
                    EAR_history.append(average_EAR)

                    # Extract only x,y coordinates
                    image_points = np.array([(p[0], p[1]) for p in head_pose_landmarks_2d_index_points], dtype=np.float64)

                    # Convert 3D model points to numpy array for reasons to use it cuz AI dummy
                    model_points = np.array(head_pose_landmarks_3d_cordinates, dtype=np.float64)
                    #call functions
                    yaw, pitch, roll = head_Pose_Estimation(camera_matrix, image_points, model_points, dist_coeffs)
                    horizontal_gaze, vertical_gaze, iris_x_norm, iris_y_norm, eye_width, eye_height = gaze_detection(left_eye_points, right_eye_points, left_iris_points, right_iris_points,left_eye_boundaries_points, Right_eye_boundaries_points, debug=False)

                    
                    # getting values from the lists we made
                    ear_mean = np.mean(EAR_history) if len(EAR_history) > 0 else 0
                    ear_variance = np.var(EAR_history) if len(EAR_history) > 0 else 0

                    # Calculate blink rate and average blink
                    blink_rate = calculate_blink_rate(blink_timestamps)
                    avg_blink_duration = average_blink_duration(blink_durations_history)

                    #displaying It for testing (and cuz cool
                    cv2.putText(frame, f'Left EARRR: {left_EAR:.2f}', (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'Right EARRR: {right_EAR:.2f}', (30, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'Blinksss: {blinks}', (30, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'Blink Durationnn: {blink_duration:.2f}s', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'Yawwwwww: {yaw:.2f}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'Pitchccccc: {pitch:.2f}', (30, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'Rolll: {roll:.2f}', (30, 210), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'Gaze Horizontalll: {horizontal_gaze:.2f}', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f'Gaze Verticalll: {vertical_gaze:.2f}', (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, "to train lable the frames by where yo are looking at", (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f'Iris X Norm: {iris_x_norm:.2f} Iris Y Norm: {iris_y_norm:.2f}', (10, height - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Prepare features for gaze model input
                    features = np.array([[iris_x_norm, iris_y_norm, eye_width, eye_height]])

                    # Scale features 
                    features_scaled = scaler.transform(features)

                    # Predict class probabilitiesn)
                    predicted_label = "idk"
                    if total_frame % 3 == 0:
                        pred_probs = model.predict(features_scaled, verbose=0)
                        predicted_class_index = np.argmax(pred_probs)
                        predicted_label = label_classes[predicted_class_index]

                    cv2.putText(frame, f'Predicted Gaze: {predicted_label}', (30, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Prepare features as numpy array for blink/drowsiness modeutl inp(this si acteully ununderstandable you need the athings in a array for the stupodo ai to understand)
                    features_blink = np.array([[blink_rate, avg_blink_duration, ear_mean, ear_variance, longest_eye_close]])

                    # Scale features(not make the whole program crash)
                    try:
                        features_blink_scaled = scaler_blink.transform(features_blink)
                    except Exception as e:
                        print("WARNING: scaler_blink.transform failed:", e)
                        features_blink_scaled = features_blink

                    # Predict probabilities for drowsiness
                    predicted_label_blink = "Unknown"
                    if model_blink is not None:
                        print(f"DEBUG: features_blink_scaled = {features_blink_scaled}")
                        print(f"DEBUG: features_blink_scaled shape = {features_blink_scaled.shape}")
                        
                        pred_probs_blink = model_blink.predict(features_blink_scaled, verbose=0)  # Add verbose=0 to suppress output this how python works dont ask me why
                        
                        predicted_class_index_blink = np.argmax(pred_probs_blink)
                        predicted_label_blink = blink_classes[predicted_class_index_blink] 
                    else:
                         #fallback math stuff (just in case AI not work)
                        if avg_blink_duration > 0.5 or longest_eye_close > 1.0:
                            predicted_label_blink = "Very Tired"
                        elif blink_rate < 0.05:
                            predicted_label_blink = "Tired"
                        else:
                            predicted_label_blink = "Normal"

                    # Display prediction
                    cv2.putText(frame, f'Drowsiness: {predicted_label_blink}', (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # key press to set label
                    if key == ord('l') or key == ord('L'):
                        label = "Left"
                    elif key == ord('r') or key == ord('R'):
                        label = "Right"
                    elif key == ord('u') or key == ord('U'):
                        label = "Up"
                    elif key == ord('d') or key == ord('D'):
                        label = "Down"
                    elif key == ord('c') or key == ord('C'):
                        label = "Center"
                    elif key == ord('n') or key == ord('N'):
                        label = "Normal"
                    elif key == ord('t') or key == ord('T'):
                        label = "Tired"
                    elif key == ord('v') or key == ord('V'):
                        label = "Very Tired"
                    
                    # Prepare the full frame data row including label  for AI 1 and 2
                    frame_data = [
                        total_frame, left_EAR, right_EAR, average_EAR, blinks,
                        blink_duration, yaw, pitch, roll, horizontal_gaze, vertical_gaze,
                        iris_x_norm, iris_y_norm, eye_width, eye_height, label if label else "",
                        blink_rate, avg_blink_duration, ear_mean, ear_variance, longest_eye_close
                    ]
                    if label is not None and label != "":
                        csv_writer.writerow(frame_data)
                        csv_file.flush()
                        print(f"Saved frame {total_frame} with label {label}")
                    total_frame += 1
                    #now we do the alarm stuff
                    if MODE == "Driver_Safety":
                        current_time = time.time()
                        
                        if predicted_label != "Center":
                            if not_center_start_time is None:
                                not_center_start_time = current_time
                            
                            time_looking_away = current_time - not_center_start_time
                            if time_looking_away >= ALERT_DELAY_SECONDS:
                                if alert_cooldown <= 0:
                                    winsound.Beep(1000, 300)
                                    print(f"⚠️ ALERT: Look at the road! (Looking {predicted_label} for {time_looking_away:.1f}s)")
                                    alert_cooldown = ALERT_COOLDOWN_FRAMES
                                
                                # Show alert text on screen
                                cv2.putText(frame, f"⚠️ LOOK AT THE ROAD! ({time_looking_away:.1f}s)", 
                                        (width//2 - 200, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        else:
                            # Reset timer when looking at center
                            not_center_start_time = None
                        
                        if predicted_label_blink == "Very Tired":
                            if tired_start_time is None:
                                tired_start_time = current_time
                            
                            # Check if they've been tired for 5+ seconds
                            time_tired = current_time - tired_start_time
                            if time_tired >= ALERT_DELAY_SECONDS:
                                if alert_cooldown <= 0:
                                    winsound.Beep(2000, 500)
                                    print(f"⚠️ ALERT: TOO TIRED! ({time_tired:.1f}s)")
                                    alert_cooldown = ALERT_COOLDOWN_FRAMES
                                
                                # Show alert text on screen
                                cv2.putText(frame, f"⚠️ TOO TIRED - pull over ({time_tired:.1f}s)", 
                                        (width//2 - 250, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        else:
                            tired_start_time = None
                        
                        # Update cooldown counter
                        if alert_cooldown > 0:
                            alert_cooldown -= 1
                    elif MODE == "Mouse":
                        # Map gaze to screen coordinates
                        mouse_x, mouse_y = map_gaze_to_screen(horizontal_gaze, vertical_gaze)
                        
                        # Move the mouse
                        pyautogui.moveTo(mouse_x, mouse_y)
                        
                        # Blink to click 
                        if predicted_label_blink == "Blink":  
                            pyautogui.click()
                            cv2.putText(frame, "CLICK!", (width//2 - 50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        
                        # Display mouse coordinates
                        cv2.putText(frame, f'Mouse: ({mouse_x}, {mouse_y})', (30, 330),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Draw crosshair showing where you're looking
                        cv2.circle(frame, (int(horizontal_gaze * width), int(vertical_gaze * height)), 
                                10, (0, 255, 255), 2)

            cv2.imshow("I hate science fair from the bottom of my heart never doing this again in my life", frame)  # show the frame (with landmarks)

        

        # setting a stop thingie
        if key == ord("p"):
            stop = True
        elif there == False:
            print("there is no webcam idiot")
#finishhhhhhhhhhhhhhhhhh
csv_file.close()
webcam.release()
cv2.destroyAllWindows()




