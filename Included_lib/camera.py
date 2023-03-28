import cv2


class Camera:
    def __init__(self, camera_settings):
        self.camera = None
        self.settings = camera_settings
        self.name = self.settings['Name']
        self.max_resolution = self.settings['MaxResolution']
        self.actual_resolution = self.settings['Resolution']

    def set(self, nr_camera=0):
        self.camera = cv2.VideoCapture(nr_camera)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.actual_resolution['Width'])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.actual_resolution['Height'])

    def make_photo(self):
        ret, frame = self.camera.read()
        if ret:
            return frame
        else:
            message = 'error' + '#' + 'brak zdjecia'
            print(message)
            return message

    def release(self):
        self.camera.release()
