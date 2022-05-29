import numpy as np
import cv2
import io
import zipfile
import csv
import streamlit as st


def boxes_to_points(boxes: list):
    result = []
    for box in boxes:
        x, y, w, h = box
        center_point = x + int(w / 2), y + int(h / 2)
        result.append(center_point)
    return result


def draw_boxes(image: np.ndarray, box: list, draw_variant: str):
    x, y, w, h = box

    center_point = x + int(w / 2), y + int(h / 2)
    if draw_variant == "Рамки":
        top_left_point, top_right_point = (x, y), (x + w, y + h)
        cv2.rectangle(
            image, top_left_point, top_right_point, (75, 75, 255), thickness=10
        )
    elif draw_variant == "Точки":
        cv2.circle(image, center_point, 10, (75, 75, 255), -1)


def bytes_to_numpy(image: bytes):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image


def make_csv_report_count(data: dict):
    """отчёт по числу моржей в файлах"""
    csv_buffer = io.StringIO()
    fieldnames = ["filename", "walrus_count"]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    for el in data:
        writer.writerow({"filename": el["filename"]+'.jpg', "walrus_count": len(el["boxes"])})

    return csv_buffer


def make_csv_report_coords(data):

    fieldnames = ["x", "y"]
    result = []
    for image_data in data:
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for el in image_data["points"]:
            writer.writerow({"x": el[0], "y": el[1]})
        result.append((image_data["filename"] + ".csv", csv_buffer))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, data in result:
            zip_file.writestr(file_name, data.getvalue())
    return zip_buffer
