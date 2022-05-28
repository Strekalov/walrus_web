import streamlit as st


def render_svg(svg_path:str):
    """Renders the given svg string."""
    with open(svg_path, "r") as file:
        svg = file.read()
    st.image(svg, width=128, use_column_width='always')
    

def render_element_in_center(st_element):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st_element
    with col3:
        st.write(' ')
    
def render_title():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        render_svg("assets/svg.svg")
    with col3:
        st.write(' ')
    st.title("Подсчёт моржей на изображении!")