import streamlit as st
from walrus_web.rendering import render_title, render_element_in_center
from walrus_web.utils import bytes_to_numpy, draw_boxes
import numpy as np
import cv2
import time


result_title = st.empty()


def get_walrus_boxes(image):
    "заменить на реальную Йоло!"
    return [{"box": [200, 200, 200, 200]},{"box": [400, 600, 500, 500]}]
    


def detect_object(image: np.ndarray):
    walrus_count = 10
    yolo_result = get_walrus_boxes(image)
    boxes = [el['box'] for el in yolo_result]
    walrus_count = len(boxes)
    
    return boxes, walrus_count


def count_walruses(uploaded_images: list):
    my_bar = st.progress(0)
    count_images = len(uploaded_images)
    
    all_walruses_count = 0
    with st.expander("Разверните для просмотра деталей"):
        for i, image in enumerate(uploaded_images):

            opencv_image = bytes_to_numpy(image)
            
            if isinstance(opencv_image , np.ndarray):
                with st.spinner(f'Считаю моржей на {i+1} изображении ...'):
                    
                    boxes, valrus_count = detect_object(opencv_image)
                    image_with_boxes = opencv_image.copy()
                    for box in boxes:
                        draw_boxes(image_with_boxes, box)
                    all_walruses_count += valrus_count
                st.image(image_with_boxes,channels='BGR', caption=f"На изображении {valrus_count} моржей!")

            else:
                st.error("С изображением что-то не так, проверьте и загрузите заново!")
            
            my_bar.progress((i+1)/count_images)
    if all_walruses_count > 0:
        result_title = st.success(f"На загруженных изображениях найдено {all_walruses_count} моржей!")


def main():
    # st.balloons()
    render_title()
    imges_list = upload_image_ui()
    count_walruses(imges_list)
    

def upload_image_ui():
    uploaded_images = st.file_uploader("Пожалуйста, выберите файлы с изображениями:", 
                                      type=["png", "jpg", "jpeg"],
                                      accept_multiple_files=True,
                                      help="Нажмите, чтобы загрузить фото с моржами")
    return uploaded_images
    
        
if __name__ == '__main__':
    main()
