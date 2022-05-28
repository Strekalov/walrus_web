import numpy as np
import cv2


def draw_boxes(image: np.ndarray, box: list, draw_variant:str):
    x, y, w, h = box
    
    center_point = x + int(w/2), y + int(h/2)
    if draw_variant == 'Рамки':
        top_left_point, top_right_point = (x, y), (x+ w, y+ h)
        cv2.rectangle(image, top_left_point, top_right_point, (255, 0, 255), thickness=10)
    elif draw_variant == 'Точки':
        cv2.circle(image, center_point, 20, (255, 0, 255), -1)
    
    
def bytes_to_numpy(image: bytes):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image

