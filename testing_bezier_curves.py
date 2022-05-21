from typing import Tuple

import streamlit as st
from streamlit.components.v1 import html

import pysvelte as ps

def compute_bezier_coords(top: Tuple[int, int], bot: Tuple[int, int], opacity: float = 1):
    y_offset = (bot[1] - top[1]) / 2
    
    string = f'<svg width="190" height="160"> <path d="M {top[0]} {top[1]}'
    string += f' C {top[0]} {top[1] + y_offset}, {bot[0]} {bot[1] - y_offset}'
    string += f', {bot[0]} {bot[1]}" stroke="black" fill="transparent" opacity="{opacity}"/></svg>'
    
    return string

st.title("Bezier Curves")

top = (50, 80)
ROW_DIFF = 50
bot = (150, 80 + ROW_DIFF)

bezier_string = compute_bezier_coords(top, bot)

circle_string = ps.CircleHover().html_page_str() + bezier_string
html(circle_string, height=500, width=500)