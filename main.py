import streamlit as st
import numpy as np
import cv2
import time


result_title = st.empty()


def get_walrus_boxes(image):
    "заменить на реальную Йоло!"
    return [{"box": [200, 200, 200, 200]}]


def render_svg():
    
    """Renders the given svg string."""
    with open("svg.svg", "r") as file:
        svg = file.read()
    st.image(svg, width=128, use_column_width='always')
    

def render_title():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        render_svg()
    with col3:
        st.write(' ')
    st.title("Подсчёт моржей на изображении!")


def bytes_to_numpy(image: bytes):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image


def detect_object(image: np.ndarray):
    walrus_count = 10
    yolo_result = get_walrus_boxes(image)
    boxes = [el['box'] for el in yolo_result]
    walrus_count = len(boxes)
    
    return boxes, walrus_count


def draw_boxes(image: np.ndarray, box: list):
    top_left_point, top_right_point = (box[0], box[1]), (box[0]+ box[2], box[1]+ box[3])
    cv2.rectangle(image, top_left_point, top_right_point, (0, 0, 255), thickness=10)



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
                st.image(image_with_boxes, width=512 ,channels='BGR', caption=f"На изображении {valrus_count} моржей!")
                st.write(f"На изображении {valrus_count} моржей!")
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
