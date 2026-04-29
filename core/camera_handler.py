import cv2
import numpy as np
import sys
import time

try:
    from pypylon import pylon

    BASLER_AVAILABLE = True
except ImportError:
    BASLER_AVAILABLE = False


class CameraHandler:
    def __init__(self, camera_type="Basler"):
        self.camera_type = camera_type
        self.webcam_capture = None
        self.basler_camera = None

    def start(self):
        """Initializes and opens the selected camera hardware."""
        if self.camera_type == "Webcam":
            # On Windows, MSMF is default but highly buggy. We MUST force DSHOW.
            backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY

            # Try index 0 first
            self.webcam_capture = cv2.VideoCapture(0, backend)

            if not self.webcam_capture.isOpened():
                # If 0 fails, try index 1 (common on some laptops)
                self.webcam_capture = cv2.VideoCapture(1, backend)

            if not self.webcam_capture.isOpened():
                raise Exception("Could not connect to the webcam.\n\n"
                                "1. Ensure no other app (Zoom, Teams, Camera App) is using it.\n"
                                "2. Check Windows Settings -> Privacy -> Camera.")

            # Force a safe, standard resolution to prevent stream corruption
            self.webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Warm-up sequence: Pull 5 frames to clear any corrupt initial buffers
            valid_frame_found = False
            for _ in range(5):
                ret, frame = self.webcam_capture.read()
                if ret and frame is not None and frame.sum() > 0:
                    valid_frame_found = True
                    break
                time.sleep(0.1)

            if not valid_frame_found:
                self.webcam_capture.release()
                self.webcam_capture = None
                raise Exception("Camera connected, but the video stream is empty (black screen).\n"
                                "Windows is likely blocking Python from accessing the camera feed.")

        elif self.camera_type == "Basler":
            if not BASLER_AVAILABLE:
                raise Exception("The 'pypylon' library is not installed.")

            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if len(devices) == 0:
                raise Exception("No Basler camera found. Is it plugged in?")

            self.basler_camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            self.basler_camera.Open()

            try:
                self.basler_camera.ExposureAuto.SetValue('Continuous')
            except:
                pass

            self.basler_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        else:
            raise Exception(f"Unknown camera type selected: {self.camera_type}")

    def stop(self):
        """Safely shuts down the camera hardware."""
        if self.camera_type == "Webcam" and self.webcam_capture is not None:
            self.webcam_capture.release()
            self.webcam_capture = None

        elif self.camera_type == "Basler" and self.basler_camera is not None:
            if self.basler_camera.IsGrabbing():
                self.basler_camera.StopGrabbing()
            self.basler_camera.Close()
            self.basler_camera = None

    def get_frame(self):
        """Grabs a single frame and ensures it is formatted as a BGR numpy array."""
        if self.camera_type == "Webcam" and self.webcam_capture is not None:
            ret, frame = self.webcam_capture.read()
            # Double check that the frame isn't corrupted before sending it to GUI
            if ret and frame is not None and frame.sum() > 0:
                return True, frame
            return False, None

        elif self.camera_type == "Basler" and self.basler_camera is not None and self.basler_camera.IsGrabbing():
            try:
                grabResult = self.basler_camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    img_array = grabResult.Array
                    if len(img_array.shape) == 2:
                        frame = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                    else:
                        frame = img_array
                    grabResult.Release()
                    return True, frame
                grabResult.Release()
            except Exception as e:
                print(f"Basler grab error: {e}")

        return False, None