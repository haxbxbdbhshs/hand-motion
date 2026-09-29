"""Download the official MediaPipe model, verifying its SHA-256 checksum."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
SHA256 = '97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482'
DESTINATION = Path(__file__).with_name('gesture_recognizer.task')


def main():
    if DESTINATION.exists():
        if hashlib.sha256(DESTINATION.read_bytes()).hexdigest() == SHA256:
            print('Model already installed and verified.')
            return
        raise RuntimeError('Existing model differs from the official model; move it aside before downloading.')
    with urlopen(URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError('Model checksum mismatch; no file was written.')
    DESTINATION.write_bytes(data)
    print(f'Model installed: {DESTINATION}')


if __name__ == '__main__':
    main()
