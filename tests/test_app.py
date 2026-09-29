import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import runtime


class AppTests(unittest.TestCase):
    def test_import_does_not_acquire_camera_or_models(self):
        with patch.object(cv2, 'VideoCapture') as camera, \
             patch.object(runtime.mp.solutions.hands, 'Hands') as hands, \
             patch.object(runtime.mp.tasks.vision.GestureRecognizer, 'create_from_options') as recognizer:
            import main
            importlib.reload(main)
            camera.assert_not_called()
            hands.assert_not_called()
            recognizer.assert_not_called()

    def test_effect_preserves_frame_and_pixels_outside_region(self):
        import main
        frame = np.full((120, 160, 3), 100, dtype=np.uint8)
        original = frame.copy()
        polygon = np.array([[40, 30], [110, 30], [110, 90], [40, 90]], dtype=np.int32)
        result = main.make_breakcore_region(frame, polygon, 10000)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, frame.dtype)
        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(result[:20], original[:20])
        self.assertTrue(np.any(result[40:80, 50:100] != original[40:80, 50:100]))

    def test_missing_model_does_not_open_camera(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(cv2, 'VideoCapture') as camera:
            with self.assertRaises(FileNotFoundError):
                with runtime.camera_session(0, Path(folder) / 'missing.task'):
                    pass
            camera.assert_not_called()

    def test_failed_camera_releases_native_resources(self):
        cap, hands, recognizer = MagicMock(), MagicMock(), MagicMock()
        cap.isOpened.return_value = False
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'model.task'
            model.write_bytes(b'test')
            with patch.object(cv2, 'VideoCapture', return_value=cap), \
                 patch.object(cv2, 'destroyAllWindows'), \
                 patch.object(runtime.mp.solutions.hands, 'Hands', return_value=hands), \
                 patch.object(runtime.mp.tasks.vision.GestureRecognizer, 'create_from_options', return_value=recognizer):
                with self.assertRaises(RuntimeError):
                    with runtime.camera_session(0, model):
                        pass
            cap.release.assert_called_once()
            hands.close.assert_called_once()
            recognizer.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
