import os
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import box, Polygon
from math import ceil, floor

yolo_size = 1280
tr = 0.4

INPUT_IMG = 'PATH2DATA/images'
INPUT_JSON = 'PATH2DATA/markup'


OUTPUT_PATH_IMAGES = 'PATH2SAVE/yolo/images'
OUTPUT_PATH_LABELS = 'PATH2SAVE/yolo/labels'



def is_in_crop(label, x1, y1, x2, y2):
    # Проверяем отношение площади ббокса, находящейся внутри вырезанной части
    # изображения, ко всей площади ббокса. И сравниваем с пороговым значением.
    minx, miny, maxx, maxy = label[5:]
    a = box(minx, miny, maxx, maxy)
    b = box(x1, y1, x2, y2)
    if a.area < 0.001:
        return False
    return b.intersection(a).area / a.area > tr


def xywh2xyxy(x, y, w, h, img_h, img_w):
    # (x центр, y центр, ширина, высота) ->
    # -> (минимум по x, минимум по y, максимум по x, максимум по y)
    x1 = int((x - (w / 2)) * img_w)
    x2 = int((x + (w / 2)) * img_w)
    y1 = int((y - (h / 2)) * img_h)
    y2 = int((y + (h / 2)) * img_h)
    return x1, y1, x2, y2


def xyxy2xywh(x1, y1, x2, y2, img_h, img_w):
    # (минимум по x, минимум по y, максимум по x, максимум по y) ->
    # -> (x центр, y центр, ширина, высота)
    w = (x2 - x1)
    h = (y2 - y1)
    x = (x1 + w / 2.0) / img_w
    y = (y1 + h / 2.0) / img_h

    w = w / img_w
    h = h / img_h
    return round(x, 6), round(y, 6), round(w, 6), round(h, 6)


def get_new_label(label, x0, y0, img_h, img_w):
    # Вырезаем у ббоксов части выходящие за границы
    class_id, x, y, w, h = label[:5]
    x = (x - x0) / img_w
    y = (y - y0) / img_h
    w = w / img_w
    h = h / img_h

    x1, y1, x2, y2 = xywh2xyxy(x, y, w, h, img_h, img_w)
    x1, y1, x2, y2 = max(x1, 1), max(y1, 1), min(x2, img_w - 1), min(y2,
                                                                     img_h - 1)
    x, y, w, h = xyxy2xywh(x1, y1, x2, y2, img_h, img_w)
    return class_id, x, y, w, h


def save_label(label, x1, y1, new_w, new_h, image_name, i, j):
    # Сохраняем. Использовали mode='a', если файл уже существует,
    # то ответ записываем в новую строку
    class_id, x, y, w, h = get_new_label(label, x1, y1, new_h, new_w)
    txt_name = f'w-{i}_h-{j}__' + image_name[:-4] + '.txt'
    with open(os.path.join(OUTPUT_PATH_LABELS, txt_name), 'a') as f:
        print(f'{class_id} {x} {y} {w} {h}', file=f)


def json2yolo(data):
    labels = []
    for elem in data:
        x1, y1, w, h, _ = elem['bbox']
        x, y = int(x1 + w / 2), int(y1 + h / 2)
        x2 = x1 + w
        y2 = y1 + h

        labels.append(list(
            map(int, [0, x, y, w, h, x1, y1, x2, y2])))
    return labels


def read_json(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data


#
def create_polygon(box, x1, y1):
    # Создаем объект плигон. Нужно для подсчета вхождения ббокса в кроп.
    # Также для умного закрашивания
    x, y, w, h = box
    x = x - x1
    y = y - y1
    x1, y1, x2, y2 = xywh2xyxy(x, y, w, h, 1, 1)
    polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return polygon


imgs = list(filter(lambda x: x[-3:] == 'jpg', os.listdir(INPUT_IMG)))

for image_name in imgs:
    image_path = os.path.join(INPUT_IMG, image_name)
    image = cv2.imread(image_path.__str__())
    image_h, image_w, _ = image.shape
    json_path = os.path.join(INPUT_JSON, image_name[:-3] + 'json')
    json_data = read_json(json_path)

    labels = json2yolo(json_data)

    c_h, c_w = ceil(image_h / yolo_size), ceil(image_w / yolo_size)
    for i in range(c_h):
        for j in range(c_w):
            if j == 0:
                x1, x2 = 0, yolo_size
            elif j == c_w - 1:
                x1, x2 = image_w - yolo_size, image_w
            else:
                crop_center = (j + 0.5) * (image_w / c_w)
                x1, x2 = int(crop_center - yolo_size / 2), int(crop_center + yolo_size / 2)
            if i == 0:
                y1, y2 = 0, yolo_size
            elif i == c_h - 1:
                y1, y2 = image_h - yolo_size, image_h
            else:
                crop_center = (i + 0.5) * (image_h / c_h)
                y1, y2 = int(crop_center - yolo_size / 2), int(crop_center + yolo_size / 2)


            x1 = x1 if x1 > 0 else 0
            y1 = y1 if y1 > 0 else 0
            x2 = x2 if x2 < image_w else image_w
            y2 = y2 if y2 < image_h else image_h

            crop_img = image[y1:y2, x1:x2]
            for label in labels:
                if is_in_crop(label, x1, y1, x2, y2):
                    save_label(label, x1, y1, x2 - x1, y2 - y1, image_name, i, j)
            out_path = os.path.join(OUTPUT_PATH_IMAGES, f'w-{i}_h-{j}__' + image_name)
            cv2.imwrite(out_path, crop_img)



