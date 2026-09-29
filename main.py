from runtime import camera_session, parse_args
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


mp_hands = mp.solutions.hands
hands = None

mp_draw = mp.solutions.drawing_utils
cap = None
frame_index = 0
active_region_points = None
active_four_finger_regions = []
four_finger_draw_trail = deque(maxlen=220)
effect_locked = False
pinch_gesture_was_active = False
reset_gesture_was_active = False
frame_gesture_detected_stable = False
four_finger_drawing_active = False
four_finger_region_closed = False
four_finger_gesture_ready = True
four_finger_drawing_start_time = 0
gesture_block_until_time = 0
reset_gesture_next_allowed_time = 0
left_index_trail = deque(maxlen=45)
right_index_trail = deque(maxlen=45)
MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

gesture_options = None

gesture_recognizer = None

THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP.value
INDEX_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP.value
MIDDLE_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value
FRAME_GESTURE_DISTANCE = 50
LOCK_GESTURE_DISTANCE = 40
FOUR_FINGER_GESTURE_DISTANCE = 100
MIN_EFFECT_SQUARE_SIZE = 80
TRAIL_MIN_POINTS = 2
TRAIL_REGION_MIN_POINTS = 8
TRAIL_SMOOTHING_WINDOW = 5
TRAIL_CLOSE_DISTANCE = 70
TRAIL_CLOSE_IGNORE_RECENT_POINTS = 12
TRAIL_MIN_REGION_AREA = 1000
GEOMETRIC_APPROX_EPSILON_RATIO = 0.055
NEON_BLUE = (255, 90, 0)
NEON_CORE = (255, 190, 80)
FOUR_FINGER_DRAW_DELAY_SECONDS = 0.8
GESTURE_RESET_COOLDOWN_SECONDS = 0.8
debug_four_finger_distances = None


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
    global debug_four_finger_distances

    if (
        left_index is None or
        left_middle is None or
        right_index is None or
        right_middle is None
    ):
        debug_four_finger_distances = None
        return False

    index_distance = np.linalg.norm(left_index - right_index)
    middle_distance = np.linalg.norm(left_middle - right_middle)
    debug_four_finger_distances = (index_distance, middle_distance)

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

    hue = ((value.astype(np.int32) // 2 + frame_index * 5) % 180).astype(np.uint8)

    hsv = cv2.merge([
        hue,
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


def smooth_trail_points(trail, window_size=TRAIL_SMOOTHING_WINDOW):
    points = np.array(trail, dtype=np.float32)
    if len(points) < window_size:
        return points.astype(np.int32)

    smoothed_points = []
    half_window = window_size // 2
    for index in range(len(points)):
        start = max(0, index - half_window)
        end = min(len(points), index + half_window + 1)
        smoothed_points.append(points[start:end].mean(axis=0))

    return np.array(smoothed_points, dtype=np.int32)


def make_polygon_from_trails(left_trail, right_trail):
    if len(left_trail) < TRAIL_REGION_MIN_POINTS or len(right_trail) < TRAIL_REGION_MIN_POINTS:
        return None

    left_points = smooth_trail_points(left_trail)
    right_points = smooth_trail_points(right_trail)
    return np.vstack([left_points, right_points[::-1]]).astype(np.int32)


def make_polygon_from_trail(trail):
    if len(trail) < TRAIL_REGION_MIN_POINTS:
        return None

    return smooth_trail_points(trail).astype(np.int32)


def make_geometric_polygon(polygon):
    if polygon is None or len(polygon) < 3:
        return None

    contour = cv2.convexHull(np.array(polygon, dtype=np.int32)).reshape(-1, 1, 2)
    area = cv2.contourArea(contour)
    if area < TRAIL_MIN_REGION_AREA:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(
        contour,
        perimeter * GEOMETRIC_APPROX_EPSILON_RATIO,
        True
    )
    vertex_count = len(approx)

    if vertex_count == 3:
        return approx.reshape(-1, 2).astype(np.int32)

    if vertex_count == 4:
        rect = cv2.minAreaRect(contour)
        return cv2.boxPoints(rect).astype(np.int32)

    if len(contour) >= 5:
        center, axes, angle = cv2.fitEllipse(contour)
        ellipse_center = (int(center[0]), int(center[1]))
        ellipse_axes = (
            max(1, int(axes[0] / 2)),
            max(1, int(axes[1] / 2))
        )
        return cv2.ellipse2Poly(
            ellipse_center,
            ellipse_axes,
            int(angle),
            0,
            360,
            8
        ).astype(np.int32)

    center, radius = cv2.minEnclosingCircle(contour)
    circle_center = (int(center[0]), int(center[1]))
    circle_radius = max(1, int(radius))
    return cv2.ellipse2Poly(
        circle_center,
        (circle_radius, circle_radius),
        0,
        0,
        360,
        8
    ).astype(np.int32)


def make_four_finger_region_from_trails(frame, left_trail, right_trail, frame_index):
    polygon = make_polygon_from_trails(left_trail, right_trail)
    if polygon is None:
        return frame

    return make_four_finger_regions(frame, [polygon], frame_index)


def make_four_finger_regions(frame, polygons, frame_index):
    if not polygons:
        return frame

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, polygons, 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.bitwise_and(edges, mask)

    thin_edges = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8), iterations=1)
    glow_edges = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=2)
    glow = cv2.GaussianBlur(glow_edges, (0, 0), 7)
    glow = cv2.normalize(glow, None, 0, 255, cv2.NORM_MINMAX)

    dark_frame = (frame * 0.18).astype(np.uint8)
    blue_glow = np.zeros_like(frame)
    blue_glow[:, :, 0] = glow
    blue_glow[:, :, 1] = (glow * 0.28).astype(np.uint8)
    blue_glow[mask == 0] = 0

    result = frame.copy()
    result[mask == 255] = dark_frame[mask == 255]
    result = cv2.addWeighted(result, 1.0, blue_glow, 1.35, 0)
    result[thin_edges > 0] = NEON_CORE

    for polygon in polygons:
        cv2.polylines(result, [polygon], isClosed=True, color=NEON_BLUE, thickness=2)
        cv2.polylines(result, [polygon], isClosed=True, color=NEON_CORE, thickness=1)

    return result


def clear_index_trails():
    left_index_trail.clear()
    right_index_trail.clear()


def clear_four_finger_trail():
    four_finger_draw_trail.clear()


def add_index_trail_points(left_index, right_index):
    if left_index is not None:
        left_index_trail.append(tuple(left_index))
    if right_index is not None:
        right_index_trail.append(tuple(right_index))


def add_four_finger_trail_point(right_index):
    if right_index is not None:
        four_finger_draw_trail.append(tuple(right_index))


def is_four_finger_trail_closed():
    if len(four_finger_draw_trail) < TRAIL_REGION_MIN_POINTS:
        return False

    return make_closed_trail_polygon() is not None


def make_closed_trail_polygon():
    if len(four_finger_draw_trail) < TRAIL_REGION_MIN_POINTS + TRAIL_CLOSE_IGNORE_RECENT_POINTS:
        return None

    points = list(four_finger_draw_trail)
    old_points = points[:-TRAIL_CLOSE_IGNORE_RECENT_POINTS]
    end = np.array(points[-1])
    distances = np.linalg.norm(np.array(old_points) - end, axis=1)
    close_index = int(np.argmin(distances))

    if distances[close_index] >= TRAIL_CLOSE_DISTANCE:
        return None

    closed_segment = points[close_index:]
    polygon = make_polygon_from_trail(closed_segment)
    if polygon is None:
        return None

    area = cv2.contourArea(cv2.convexHull(polygon))
    if area < TRAIL_MIN_REGION_AREA:
        return None

    return polygon


def draw_index_trail(frame, trail, color):
    if len(trail) < TRAIL_MIN_POINTS:
        return

    points = list(trail)
    for index in range(1, len(points)):
        intensity = index / len(points)
        faded_color = tuple(int(channel * intensity) for channel in color)
        thickness = max(1, int(6 * intensity))
        cv2.line(frame, points[index - 1], points[index], faded_color, thickness, cv2.LINE_AA)

    cv2.circle(frame, points[-1], 9, color, -1, cv2.LINE_AA)


def draw_index_trails(frame):
    draw_index_trail(frame, left_index_trail, (255, 0, 255))
    draw_index_trail(frame, right_index_trail, (255, 255, 0))


def has_reset_gesture(gesture_result):
    for hand_gestures in gesture_result.gestures:
        if not hand_gestures:
            continue

        best_gesture = hand_gestures[0]

        if (
            best_gesture.category_name == "Open_Palm" and
            best_gesture.score > 0.65
        ):
            return True

    return False



def main():
    global active_region_points, cap, effect_locked, four_finger_drawing_active, four_finger_drawing_start_time, four_finger_gesture_ready, four_finger_region_closed, frame_gesture_detected_stable, frame_index, gesture_block_until_time, gesture_recognizer, hands, pinch_gesture_was_active, reset_gesture_next_allowed_time, reset_gesture_was_active
    args = parse_args()
    with camera_session(args.camera, args.model) as resources:
        cap, hands, gesture_recognizer = resources
        while True:
            current_time = time.monotonic()
            frame_index += 1
            ret, frame = cap.read()
            if not ret:
                break

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

            reset_gesture_detected = has_reset_gesture(gesture_result)
            reset_gesture_started = (
                reset_gesture_detected and
                current_time >= reset_gesture_next_allowed_time
            )
            reset_gesture_was_active = reset_gesture_detected

            if reset_gesture_started:
                active_region_points = None
                active_four_finger_regions.clear()
                effect_locked = False
                pinch_gesture_was_active = False
                reset_gesture_was_active = True
                frame_gesture_detected_stable = False
                four_finger_drawing_active = False
                four_finger_region_closed = False
                four_finger_gesture_ready = True
                four_finger_drawing_start_time = 0
                gesture_block_until_time = current_time + GESTURE_RESET_COOLDOWN_SECONDS
                reset_gesture_next_allowed_time = current_time + GESTURE_RESET_COOLDOWN_SECONDS
                clear_index_trails()
                clear_four_finger_trail()

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

            gestures_are_blocked = current_time < gesture_block_until_time

            if not four_finger_gesture_detected:
                four_finger_gesture_ready = True

            if (
                not gestures_are_blocked and
                (frame_gesture_detected_stable or frame_gesture_detected) and
                not effect_locked and
                len(fingertip_points) == 4
            ):
                active_region_points = make_rectangle_from_points(fingertip_points, width, height)
                frame_gesture_detected_stable = True

            four_finger_gesture_started = (
                four_finger_gesture_detected and
                four_finger_gesture_ready and
                not four_finger_drawing_active and
                not gestures_are_blocked
            )

            if four_finger_gesture_started:
                four_finger_gesture_ready = False
                four_finger_region_closed = False
                four_finger_drawing_active = True
                four_finger_drawing_start_time = current_time + FOUR_FINGER_DRAW_DELAY_SECONDS
                clear_index_trails()
                clear_four_finger_trail()

            four_finger_delay_is_over = current_time >= four_finger_drawing_start_time

            if four_finger_drawing_active and not four_finger_region_closed and four_finger_delay_is_over:
                add_four_finger_trail_point(right_index)
                polygon = make_closed_trail_polygon()
                if polygon is not None:
                    geometric_polygon = make_geometric_polygon(polygon)
                    if geometric_polygon is not None:
                        active_four_finger_regions.append(geometric_polygon)
                        four_finger_region_closed = False
                        four_finger_drawing_active = False
                        four_finger_gesture_ready = False
                        four_finger_drawing_start_time = 0
                        gesture_block_until_time = current_time + GESTURE_RESET_COOLDOWN_SECONDS
                        clear_index_trails()
                        clear_four_finger_trail()

            pinch_gesture_started = lock_gesture_detected and not pinch_gesture_was_active

            if pinch_gesture_started and active_region_points is not None and not gestures_are_blocked:
                if effect_locked:
                    effect_locked = False
                else:
                    effect_locked = True

            pinch_gesture_was_active = lock_gesture_detected

            effect_is_active = active_region_points is not None
            four_finger_effect_is_active = len(active_four_finger_regions) > 0

            if four_finger_drawing_active:
                add_index_trail_points(None, right_index)
            else:
                clear_index_trails()

            if effect_is_active:
                frame = make_breakcore_region(frame, active_region_points, frame_index)

            if four_finger_effect_is_active:
                frame = make_four_finger_regions(frame, active_four_finger_regions, frame_index)

            if four_finger_drawing_active:
                if four_finger_delay_is_over:
                    status = "DRAW CLOSED AREA WITH RIGHT INDEX"
                else:
                    seconds_left = max(0, four_finger_drawing_start_time - current_time)
                    status = f"GET READY: {seconds_left:.1f}s"
                status_color = (255, 255, 255)
            elif effect_is_active:
                if effect_locked:
                    status = "BREAKCORE MODE: LOCKED"
                else:
                    status = "BREAKCORE MODE: TRACKING"
                status_color = (0, 255, 0)
            elif four_finger_effect_is_active:
                status = f"FOUR FINGER SECTORS: {len(active_four_finger_regions)}"
                status_color = (255, 255, 255)
            else:
                status = "Make a finger frame to start effect"
                status_color = (0, 0, 255)

            draw_index_trails(frame)

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

            if debug_four_finger_distances is not None:
                index_distance, middle_distance = debug_four_finger_distances
                cv2.putText(
                    frame,
                    f"4F index:{index_distance:.0f} middle:{middle_distance:.0f} limit:{FOUR_FINGER_GESTURE_DISTANCE}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            cv2.imshow("Hand Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                effect_locked = False
                active_region_points = None
                active_four_finger_regions.clear()
                pinch_gesture_was_active = False
                reset_gesture_was_active = False
                frame_gesture_detected_stable = False
                four_finger_drawing_active = False
                four_finger_region_closed = False
                four_finger_gesture_ready = True
                four_finger_drawing_start_time = 0
                gesture_block_until_time = 0
                reset_gesture_next_allowed_time = 0
                clear_index_trails()
                clear_four_finger_trail()
            if key == 27:  # ESC
                break


if __name__ == "__main__":
    main()
