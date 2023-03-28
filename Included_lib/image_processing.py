import cv2
from time import time
import socket


def crop_image(image, size):
    """
    format size {"X_start": 0, "X_stop":300, "Y_start": 0, "Y_stop": 400}
    :param image:
    :param size:
    :return:
    """

    try:
        return image[size['Y_start']:size['Y_stop'], size['X_start']:size['X_stop']]
    except TypeError as error:
        return "Error#crop_image#{}".format(error)


def convert_bgr_to_rgb(image):
    try:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as error:
        return "Error#convert_bgr_to_rgb#{}".format(error)


def save_image(image, path_image):
    try:
        cv2.imwrite(path_image, image)
    except Exception as error:
        return "Error#save_image#{}".format(error)


def generate_unique_name():
    now = time()
    now = str(int(now * 1000))
    return now


def save_image_to_file_server(image):
    try:
        file_name = generate_unique_name()
        file_name = '/home/pi/NAS/' + str(socket.gethostname()) + '_' + str(file_name) + '.webp'
        cv2.imwrite(file_name, image, [cv2.IMWRITE_WEBP_QUALITY, 50])
    except Exception as error:
        return "Error#save_image_to_file_server#{}".format(error)
