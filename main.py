import streamlit as st
from walrus_web.rendering import render_title, render_draw_select
from walrus_web.utils import (
    bytes_to_numpy,
    draw_boxes,
    boxes_to_points,
    make_csv_report_coords,
    make_csv_report_count,
)
from walrus_yolo.model import ObjectDetector
import numpy as np
import cv2
import time
import base64
import os

st.experimental_memo.clear()
result_title = st.empty()

@st.experimental_singleton
def get_detector():

    detector = ObjectDetector(
        "./walrus_yolo/models/yolov5x6_29.05.2022.onnx",
        image_size=(64 * 81, 64 * 81),
        device="gpu",
    )
    return detector

detector = get_detector()

@st.cache(allow_output_mutation=True)
@st.experimental_memo
def get_walrus_boxes(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector(image=image)
    return results


@st.cache(allow_output_mutation=True)
@st.experimental_memo
def detect_object(image: np.ndarray):
    yolo_result = get_walrus_boxes(image)
    boxes = [el["box"] for el in yolo_result]
    result = {}

    result["points"] = boxes_to_points(boxes)
    result["boxes"] = boxes
    result["walruses_count"] = len(boxes)

    time.sleep(2)
    return result


def count_walruses(uploaded_images: list):
    my_bar = st.progress(0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(" ")
    with col2:
        proccess_image = st.empty()
    with col3:
        st.write(" ")

    count_images = len(uploaded_images)

    all_walruses_count = 0
    coords_report_data = []
    all_walruses_data = []

    if count_images > 0:
        with col2:
            with open("assets/walrus_progress.gif", "rb") as file:
                gif = file.read()
                data_url = base64.b64encode(gif).decode("utf-8")
                proccess_image = st.markdown(
                    f'<img src="data:image/gif;base64,{data_url}">',
                    unsafe_allow_html=True,
                )

    with st.expander("Разверните для просмотра деталей"):
        for i, image in enumerate(uploaded_images):
            filename, extension = os.path.splitext(image.name)
            opencv_image = bytes_to_numpy(image)

            if isinstance(opencv_image, np.ndarray):
                with st.spinner(f"Считаю моржей на {i+1} изображении ..."):

                    walruses_data = detect_object(opencv_image)
                    walruses_data["filename"] = filename
                    all_walruses_data.append(walruses_data)
                    image_with_boxes = opencv_image.copy()
                    walrus_count = walruses_data["walruses_count"]

                    for box in walruses_data["boxes"]:
                        draw_boxes(image_with_boxes, box, draw_variant=draw_variant)
                    all_walruses_count += walrus_count
                st.image(
                    image_with_boxes,
                    channels="BGR",
                    caption=f"На изображении найдено {walrus_count} моржей!",
                )

            else:
                st.error("С изображением что-то не так, проверьте и загрузите заново!")

            my_bar.progress((i + 1) / count_images)

        if count_images > 0:
            with col1:
                st.write(" ")
            with col2:
                with open("assets/walrus_title.svg", "r") as file:
                    svg = file.read()
                    proccess_image.empty()
                    proccess_image = st.image(svg, width=128, use_column_width="always")

            with col3:
                st.write(" ")
    if all_walruses_count > 0:
        result_title = st.success(
            f"На загруженных изображениях найдено {all_walruses_count} моржей!"
        )
        with col1:
            download_button1 = st.empty()
            download_button2 = st.empty()
            report_1 = make_csv_report_count(all_walruses_data)
            report_2 = make_csv_report_coords(all_walruses_data)
        
        if report_1:
            download_button1 = st.download_button(
                    label="Скачать отчёт по количеству моржей в csv",
                    data=report_1.getvalue(),
                    file_name="count_report.csv",
                    mime="text/csv",
                )
            download_button2 = st.download_button(
                    label="Скачать отчёт по координатам моржей в zip",
                    data=report_2.getvalue(),
                    file_name="coords_report.zip",
                )
            st.stop()
    st.experimental_memo.clear()  
    


render_title()
draw_variant = render_draw_select()


def main():
    # st.balloons()

    imges_list = upload_image_ui()
    count_walruses(imges_list)


def upload_image_ui():
    uploaded_images = st.file_uploader(
        "Пожалуйста, выберите файлы с изображениями:",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Нажмите, чтобы загрузить фото с моржами",
    )
    return uploaded_images


if __name__ == "__main__":
    main()
