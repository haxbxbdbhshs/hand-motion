from runtime import camera_session, parse_args
import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


CAMERA_INDEX = 0
OUTPUT_FPS = 30
SHOW_PREVIEW = False
MIRROR_WEBCAM = True

mp_hands = mp.solutions.hands
hands = None

mp_draw = mp.solutions.drawing_utils
cap = None
frame_index = 0
active_region_points = None
effect_locked = False
pinch_gesture_was_active = False
frame_gesture_detected_stable = False
MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")

gesture_options = None
gesture_recognizer = None

THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP.value
INDEX_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP.value
MIDDLE_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value
FRAME_GESTURE_DISTANCE = 50
LOCK_GESTURE_DISTANCE = 40
FOUR_FINGER_GESTURE_DISTANCE = 50
MIN_EFFECT_SQUARE_SIZE = 80


def is_frame_gesture(left_thumb, left_index, right_thumb, right_index):
    if (
        left_thumb is None or
        left_index is None or
        right_thumb is None or
        right_index is None
    ):
        return False

    distance_1 = np.linalg.norm(right_thumb - left_index)
    distance_2 = np.linalg.norm(left_thumb - right_index)

    return (
        distance_1 < FRAME_GESTURE_DISTANCE and
        distance_2 < FRAME_GESTURE_DISTANCE
    )


def is_lock_gesture(left_thumb, left_index, right_thumb, right_index):
    if (
        left_thumb is None or
        left_index is None or
        right_thumb is None or
        right_index is None
    ):
        return False

    left_pinch_distance = np.linalg.norm(left_thumb - left_index)
    right_pinch_distance = np.linalg.norm(right_thumb - right_index)

    return (
        left_pinch_distance < LOCK_GESTURE_DISTANCE and
        right_pinch_distance < LOCK_GESTURE_DISTANCE
    )


def is_four_finger_gesture(left_index, left_middle, right_index, right_middle):
    if (
        left_index is None or
        left_middle is None or
        right_index is None or
        right_middle is None
    ):
        return False

    index_distance = np.linalg.norm(left_index - right_index)
    middle_distance = np.linalg.norm(left_middle - right_middle)

    return (
        index_distance < FOUR_FINGER_GESTURE_DISTANCE and
        middle_distance < FOUR_FINGER_GESTURE_DISTANCE
    )


def landmark_to_point(landmark, width, height):
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return np.array([x, y], dtype=np.int32)


def order_points_clockwise(points):
    points = np.array(points, dtype=np.int32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def make_rectangle_from_points(points, width, height):
    points = np.array(points, dtype=np.int32)

    min_x = points[:, 0].min()
    max_x = points[:, 0].max()
    min_y = points[:, 1].min()
    max_y = points[:, 1].max()

    left = max(min_x, 0)
    right = min(max_x, width - 1)
    top = max(min_y, 0)
    bottom = min(max_y, height - 1)

    if right - left < MIN_EFFECT_SQUARE_SIZE:
        center_x = (left + right) // 2
        left = max(center_x - MIN_EFFECT_SQUARE_SIZE // 2, 0)
        right = min(left + MIN_EFFECT_SQUARE_SIZE, width - 1)

    if bottom - top < MIN_EFFECT_SQUARE_SIZE:
        center_y = (top + bottom) // 2
        top = max(center_y - MIN_EFFECT_SQUARE_SIZE // 2, 0)
        bottom = min(top + MIN_EFFECT_SQUARE_SIZE, height - 1)

    return np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
    ], dtype=np.int32)


def make_breakcore_region(frame, points, frame_index):
    polygon = order_points_clockwise(points)
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    threshold = 120 + int(45 * np.sin(frame_index * 0.37))
    _, black_white = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    pulse_is_on = (frame_index // 6) % 4 == 0
    if pulse_is_on:
        black_white = cv2.bitwise_not(black_white)

    edges = cv2.Canny(gray, 80, 180)
    black_white[edges > 0] = 255 if pulse_is_on else 0

    y_grid, x_grid = np.indices((height, width))
    hue = ((x_grid * 0.35 + y_grid * 0.75 + frame_index * 7) % 180).astype(np.uint8)
    saturation = np.full((height, width), 255, dtype=np.uint8)
    value = np.where(black_white > 0, 255, 45).astype(np.uint8)
    hsv = cv2.merge([hue, saturation, value])
    effect = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    blue, green, red = cv2.split(effect)
    blue = np.roll(blue, -8, axis=1)
    red = np.roll(red, 8, axis=1)
    effect = cv2.merge([blue, green, red])

    edge_color = np.array([255, 255, 255] if pulse_is_on else [255, 0, 255], dtype=np.uint8)
    effect[edges > 0] = edge_color

    rng = np.random.default_rng(frame_index)

    for y in range(0, frame.shape[0], 18):
        if rng.random() < 0.45:
            stripe_height = int(rng.integers(3, 10))
            shift = int(rng.integers(-28, 29))
            stripe = np.roll(effect[y:y + stripe_height], shift, axis=1)
            stripe_color = rng.integers(0, 256, size=3, dtype=np.uint8)
            color_layer = np.full_like(stripe, stripe_color)
            effect[y:y + stripe_height] = cv2.addWeighted(stripe, 0.65, color_layer, 0.35, 0)

    effect[::4] = (effect[::4] * 0.35).astype(np.uint8)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    result = frame.copy()
    result[mask == 255] = effect[mask == 255]

    border_color = (255, 255, 255) if pulse_is_on else (0, 255, 255)
    cv2.polylines(result, [polygon], isClosed=True, color=border_color, thickness=2)
    return result


def make_four_finger_region(frame, points, frame_index):
    polygon = order_points_clockwise(points)
    height, width = frame.shape[:2]

    y_grid, x_grid = np.indices((height, width))
    waves = (
        np.sin((x_grid + frame_index * 9) * 0.045) +
        np.cos((y_grid - frame_index * 6) * 0.06)
    )
    value = ((waves + 2) / 4 * 255).astype(np.uint8)

    hsv = cv2.merge([
        ((value.astype(np.int32) // 2 + frame_index * 5) % 180).astype(np.uint8),
        np.full((height, width), 255, dtype=np.uint8),
        value
    ])
    effect = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    result = frame.copy()
    result[mask == 255] = cv2.addWeighted(
        frame[mask == 255],
        0.25,
        effect[mask == 255],
        0.75,
        0
    )
    cv2.polylines(result, [polygon], isClosed=True, color=(255, 255, 255), thickness=3)
    return result


def has_reset_gesture(gesture_result):
    for hand_gestures in gesture_result.gestures:
        if not hand_gestures:
            continue

        best_gesture = hand_gestures[0]
        if best_gesture.category_name == "Open_Palm" and best_gesture.score > 0.65:
            return True

    return False


def process_frame(frame, frame_index):
    global active_region_points
    global effect_locked
    global pinch_gesture_was_active
    global frame_gesture_detected_stable

    if MIRROR_WEBCAM:
        frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    gesture_result = gesture_recognizer.recognize_for_video(
        mp_image,
        frame_index * 33
    )

    fingertip_points = []
    left_thumb = None
    left_index = None
    left_middle = None
    right_thumb = None
    right_index = None
    right_middle = None

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            thumb_tip = landmark_to_point(hand_landmarks.landmark[THUMB_TIP], width, height)
            index_tip = landmark_to_point(hand_landmarks.landmark[INDEX_TIP], width, height)
            middle_tip = landmark_to_point(hand_landmarks.landmark[MIDDLE_TIP], width, height)
            label = handedness.classification[0].label
            if label == 'Left':
                left_thumb = thumb_tip
                left_index = index_tip
                left_middle = middle_tip
            else:
                right_thumb = thumb_tip
                right_index = index_tip
                right_middle = middle_tip
            fingertip_points.extend([thumb_tip, index_tip])

    frame_gesture_detected = is_frame_gesture(
        left_thumb,
        left_index,
        right_thumb,
        right_index
    )
    lock_gesture_detected = is_lock_gesture(
        left_thumb,
        left_index,
        right_thumb,
        right_index
    )
    four_finger_gesture_detected = is_four_finger_gesture(
        left_index,
        left_middle,
        right_index,
        right_middle
    )
    reset_gesture_detected = has_reset_gesture(gesture_result)

    if reset_gesture_detected:
        reset_effect_state()

    if (frame_gesture_detected_stable or frame_gesture_detected) and not effect_locked and len(fingertip_points) == 4:
        active_region_points = make_rectangle_from_points(fingertip_points, width, height)
        frame_gesture_detected_stable = True

    pinch_gesture_started = lock_gesture_detected and not pinch_gesture_was_active

    if pinch_gesture_started:
        if effect_locked:
            effect_locked = False
        elif active_region_points is not None:
            effect_locked = True

    pinch_gesture_was_active = lock_gesture_detected
    effect_is_active = active_region_points is not None

    if effect_is_active and four_finger_gesture_detected:
        frame = make_four_finger_region(frame, active_region_points, frame_index)
        status = "FOUR FINGER EFFECT"
        status_color = (255, 255, 255)
    elif effect_is_active:
        frame = make_breakcore_region(frame, active_region_points, frame_index)
        status = "BREAKCORE MODE: LOCKED" if effect_locked else "BREAKCORE MODE: TRACKING"
        status_color = (0, 255, 0)
    else:
        status = "Make a finger frame to start effect"
        status_color = (0, 0, 255)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    for index, point in enumerate(fingertip_points):
        color = (255, 0, 255) if index % 2 == 0 else (0, 255, 0)
        cv2.circle(frame, tuple(point), 8, color, -1)

    cv2.putText(
        frame,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA
    )

    return frame


def reset_effect_state():
    global active_region_points
    global effect_locked
    global pinch_gesture_was_active
    global frame_gesture_detected_stable

    active_region_points = None
    effect_locked = False
    pinch_gesture_was_active = False
    frame_gesture_detected_stable = False


def run_virtual_camera():
    global frame_index

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read from webcam.")

    height, width = first_frame.shape[:2]

    with pyvirtualcam.Camera(width=width, height=height, fps=OUTPUT_FPS) as virtual_cam:
        print(f"Virtual camera is running: {virtual_cam.device}")
        frame = first_frame

        while True:
            frame_index += 1
            output_frame = process_frame(frame, frame_index)

            virtual_cam.send(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
            virtual_cam.sleep_until_next_frame()

            if SHOW_PREVIEW:
                cv2.imshow("Breakcore Virtual Camera", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    reset_effect_state()
                if key == 27:  # ESC
                    break

            ret, frame = cap.read()
            if not ret:
                break



def main():
    global cap, hands, gesture_recognizer, SHOW_PREVIEW
    args = parse_args(virtual=True)
    SHOW_PREVIEW = not args.no_preview
    with camera_session(args.camera, args.model) as resources:
        cap, hands, gesture_recognizer = resources
        run_virtual_camera()


if __name__ == "__main__":
    main()
