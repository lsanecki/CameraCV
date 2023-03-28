from Included_lib.json_settings_support import JsonSettingSupport
import os
from Included_lib.camera import Camera


class BarcodeReader:
    def __init__(self):
        self.camera = None
        self.camera_settings = None
        self.path_camera_settings = None
        self.prepare_path_camera_settings()
        self.config_camera()

    def load_camera_settings(self):
        self.camera_settings = JsonSettingSupport.load_file(self.path_camera_settings)

    def config_camera(self):
        self.load_camera_settings()
        self.camera = Camera(self.camera_settings['Camera'])

    def prepare_path_camera_settings(self):
        _current_directory = os.getcwd()
        _setting_directory = "Settings"
        _setting_file = "CameraConfig.json"
        self.path_camera_settings = f"{_current_directory}/{_setting_directory}/{_setting_file}"
