"""Explicit acquisition and cleanup of camera and MediaPipe resources."""
import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path

import cv2
import mediapipe as mp

MODEL_PATH = Path(__file__).with_name('gesture_recognizer.task')


def parse_args(virtual=False):
    parser = argparse.ArgumentParser(description='Gesture-controlled webcam effects')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--model', type=Path, default=MODEL_PATH, help='MediaPipe .task model')
    if virtual:
        parser.add_argument('--no-preview', action='store_true', help='Hide preview; stop with Ctrl+C')
    return parser.parse_args()


@contextmanager
def camera_session(camera_index, model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError('Model missing. Run: python download_model.py')
    with ExitStack() as stack:
        stack.callback(cv2.destroyAllWindows)
        hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        stack.callback(hands.close)
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2)
        recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)
        stack.callback(recognizer.close)
        cap = cv2.VideoCapture(camera_index)
        stack.callback(cap.release)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera {camera_index}. Close other camera apps or use --camera 1.')
        yield cap, hands, recognizer
