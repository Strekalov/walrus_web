import streamlit as st


def render_svg(svg_path: str):
    """Renders the given svg string."""
    with open(svg_path, "r") as file:
        svg = file.read()
    st.image(svg, width=128, use_column_width="always")


def render_element_in_center(st_element):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(" ")
    with col2:
        st_element
    with col3:
        st.write(" ")


def render_title():

    st.title("Подсчёт моржей на изображении!")


def render_draw_select():
    with open("assets/walrus_sidebar.svg", "r") as file:
        svg = file.read()
    st.sidebar.image(svg, width=128, use_column_width="always")
    draw_variant = st.sidebar.radio(
        "Выбери как выделить моржей на изображении:", ("Точки", "Рамки", "Никак")
    )

    return draw_variant
