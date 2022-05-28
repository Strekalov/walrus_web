import numpy as np
import cv2


def draw_boxes(image: np.ndarray, box: list):
    top_left_point, top_right_point = (box[0], box[1]), (box[0]+ box[2], box[1]+ box[3])
    cv2.rectangle(image, top_left_point, top_right_point, (255, 0, 255), thickness=10)
    
    
def bytes_to_numpy(image: bytes):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image

