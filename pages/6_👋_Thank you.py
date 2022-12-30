# Imports.
import numpy as np
from PIL import Image
import streamlit as st

# Page config.
st.set_page_config(
    page_title="Thank you!",
    page_icon="🙏",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Haha, you found this!",
    }
)

# Title.
st.title("Thank You!")

st.markdown("""
    I began this project wanting to pursue a more comprehensive take on projects. I wanted a series 
    of projects that would tackle every aspect of the data science pipeline. At the time (2021), I only had
    a bit of predictive modeling and data preprocessing under my belt. I wanted to explore more into this data science realm.
    This lengthy project did just that. It taught me an awful lot, both technically and non-technically speaking.
    The next steps for me is starting a deep learning project! I've done plenty of deep learning in the past, but 
    I don't think any of them were as thorough as this one. I'll have to mirror this project's depth! 

    This page is a short thank you note and so I'd like to thank all the references in both
    my notebook and in `resources.md`. Without those guides, articles, and demos, I wouldn't have been
    able to do this project. I'd also like to express some appreciation for these fantastic libraries!
    And by libraries I mean all the tools I've used up to this point. Though I've ran into countless
    bugs, these libraries were thorough, easy to understand, and had great documentation. And thank you 
    [Catherine](https://github.com/crasgaitis) for your help with the visuals and banners. You made this 
    project all the more memorable. ❤️ 

    And thank *you* for reading this! I hope this project was an absolute joy for you to look through.
""")

image = np.asarray(Image.open("./img/waving_hand.png"))
st.image(image)
