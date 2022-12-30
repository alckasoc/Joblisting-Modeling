# Imports.
import streamlit as st

# Page config.
st.set_page_config(
    page_title="The Data Science Process",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Haha, you found this!",
    }
)

# Title.
st.title("The Data Science Process")

# Table of contents.

st.markdown("""
**Table of Contents**: <br>

<ul>
    <li>1. <a href=\"#1-introduction\">Introduction</a></li>
    <li>2. <a href=\"#2-cleaning\">Cleaning</a></li>
    <li>3. <a href=\"#3-eda\">Exploratory Data Analysis (EDA)</a>
        <ul>
            <li>3.1. <a href=\"#3-1-a-univariate-look\">A Univariate Look</a></li>
            <li>3.2. <a href=\"#3-2-a-multivariate-dive\">A Multivariate Dive</a></li>
            <li>3.3. <a href=\"#3-3-a-free-form-exploration\">A Free-form Exploration</a></li>
        </ul>
    </li>
    <li>4. <a href=\"#4-conclusion\">Conclusion</a></li>
    <li>5. <a href=\"#5-author-info\">Author Info</a></li>
</ul>
""", unsafe_allow_html=True)
