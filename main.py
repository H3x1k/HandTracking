import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller, Button
import pyautogui
import threading
import time

mouse = Controller()
width, height = pyautogui.size()

# Shared variables for threading
target_mouse_pos = [width * 0.5, height * 0.5]
lock = threading.Lock()

SMOOTHING = 0.3 

def mouse_mover():
    last_pos = pyautogui.position()
    while True:
        with lock:
            tx, ty = target_mouse_pos
        if hand_present:
            # Interpolate for smooth movement
            cx, cy = pyautogui.position()
            nx = int(cx + (tx - cx) * SMOOTHING)
            ny = int(cy + (ty - cy) * SMOOTHING)
            try:
                mouse.position = (nx, ny)
            except Exception:
                pass
        time.sleep(0.01)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.7)

index_pinch = False
middle_pinch = False
ring_pinch = False
hand_present = False
last_index_position = 0
sensitivity = 6
threshold = 1
pinch_threshold = 0.05

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def get_palm_normal(hand_landmarks):
    # Get 3D coordinates of the palm base
    p0 = np.array([hand_landmarks.landmark[0].x,
                   hand_landmarks.landmark[0].y,
                   hand_landmarks.landmark[0].z])
    p1 = np.array([hand_landmarks.landmark[5].x,
                   hand_landmarks.landmark[5].y,
                   hand_landmarks.landmark[5].z])
    p2 = np.array([hand_landmarks.landmark[17].x,
                   hand_landmarks.landmark[17].y,
                   hand_landmarks.landmark[17].z])

    # Two vectors on the palm
    v1 = p1 - p0
    v2 = p2 - p0

    # Compute the normal vector (cross product)
    normal = np.cross(v1, v2)
    
    # Normalize it
    normal = normal / np.sqrt(np.sum(normal**2))

    return normal
def distance(x1, x2, y1, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)
def joint_distance(j1, j2):
    return np.sqrt((j2.x-j1.x)**2 + (j2.y-j1.y)**2 + (j2.z-j1.z)**2)
def is_index_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    return joint_distance(thumb_tip, index_tip) < pinch_threshold
def is_middle_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    middle_tip = hand_landmarks.landmark[12]
    return joint_distance(thumb_tip, middle_tip) < pinch_threshold
def is_ring_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    ring_tip = hand_landmarks.landmark[16]
    return joint_distance(thumb_tip, ring_tip) < pinch_threshold


# Start the mouse mover thread
threading.Thread(target=mouse_mover, daemon=True).start()

while True:
    success, img = cap.read()
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks[0].landmark) == 21:
            hand_present = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                norm = get_palm_normal(hand_landmarks)
                if norm[2] < -0.85:
                    if index_pinch:
                        if is_index_pinch(hand_landmarks):
                            index_x, index_y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
                            last_index_x, last_index_y = last_index_position.x, last_index_position.y
                            # Update the target mouse position for the thread
                            with lock:
                                target_mouse_pos[0] += int((last_index_x - index_x) * w)
                                target_mouse_pos[1] += int((index_y - last_index_y) * h)
                            last_index_position = hand_landmarks.landmark[8]
                        else:
                            index_pinch = False
                    else:
                        if is_index_pinch(hand_landmarks):
                            index_pinch = True
                            last_index_position = hand_landmarks.landmark[8]
                        else:
                            index_pinch = False

                    if is_middle_pinch(hand_landmarks):
                        if not middle_pinch:
                            mouse.press(Button.left)
                            middle_pinch = True
                    else:
                        if middle_pinch:
                            mouse.release(Button.left)
                            middle_pinch = False

                    if is_ring_pinch(hand_landmarks):
                        if not ring_pinch:
                            mouse.press(Button.right)
                            ring_pinch = True
                    else:
                        if ring_pinch:
                            mouse.release(Button.right)
                            ring_pinch = False
    else:
        hand_present = False

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()