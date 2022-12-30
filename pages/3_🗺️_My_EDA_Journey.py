# Imports.
import streamlit as st

# Page config.
st.set_page_config(
    page_title="My EDA Journey",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Haha, you found this!",
    }
)

# Title.
st.title("❗My EDA Journey❗")

st.markdown(
"""
This short blog post is about my journey through this project! It will overlap a slight bit 
with my README.
"""
)

st.markdown(
"""
## Intro

This project took about 3 months. I started it May of 2021 (last year from when
I'm typing this). Initially, I scraped the data, designed my EDA approach, and
began my exploration. But I put this project on hold when I started doing Kaggle! Now,
this year, September, I continued on it again. And it was a blast.

## Difficulties

There were *lots* of difficulties. On the technical side, I did a TON of googling
to stylize each and every plot. I remember 3 plots initially took about 4-5 hours to make!
Over the course of this project, my skills improved and the graphs I generated grew
more visually appealing. Another technical difficult was plotly. Though similar to seaborn
and matplotlib, this new graphing library challenged my existing knowledge. The heatmaps
I created in plotly took 3 hours! It was difficult to adapt the matrix of data to a plotly heatmap.
I also faced difficulty creating this app. Dataframes and images were a lot harder to center than I had 
originally imagined. I must've spent at least 2 hours googling and re-googling and trying different techniques
to center the dataframes/images. On the non-technical side, prior to this, I had
never done a standalone EDA project. Approaching, organizing, and presenting my findings through this EDA
posed the initial challenge. It was a new process for me. And as such, I spent a bit of time beforehand
planning and designing all the components of this project (e.g. the presentation, the notebook, cleaning & EDA, the 
EDA process, keeping a spreadsheet, defining an objective, etc). Regardless of how difficult each of the aforementioned topics
were, I soon arrived at my solution. 

## Takeaways

Along the way, I learned a multitude of 
technical skills. I learned various streamlit modules for formatting webpages and embedding media.
I explored statistics and auto-EDA! I did a fair bit of research into the general EDA process.
I learned and worked with plotly and scipy. At a higher level, I learned how to organize an EDA project.
I learned how to weave all the separately built out components together to form this project. 
This project was long and demanding but fruitful. I enjoyed every struggle and every victory.
I haven't done many projects the past year and, in a larger sense, this project has
reinvigorated my joy for creating projects, for learning new things, and for creating something amazing!

Thank you for your time.

Vincent
"""
)