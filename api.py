import cv2

class CameraAPI:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

    def read_frame(self):
        if self.cap is None:
            raise Exception("Camera not opened")
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to read frame")
        return frame

    def release(self):
        if self.cap:
            self.cap.release()

    def set_resolution(self, width, height):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
api=CameraAPI()
# api.read_frame()
api.open()
# api.set_resolution(640,480)
