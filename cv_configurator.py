from cv_lite import CvCameraLite


class CvCameraConfigurator(CvCameraLite):
    def __init__(self, hostname: str):
        super().__init__(hostname)
