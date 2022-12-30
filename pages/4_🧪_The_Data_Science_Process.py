# Imports.
import numpy as np
from PIL import Image
import streamlit as st

# Page config.
st.set_page_config(
    page_title="The Data Science Process",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Haha, you found this!",
    }
)

# Banner.
col1, col2, col3 = st.columns([2, 5, 1])
with col2:
    image = np.asarray(Image.open("./diagrams/pipeline_diagram-removebg-preview.png"))
    st.image(image)

# Title.
st.title("🚨The Data Science Process🚨")

# Table of contents.

st.markdown("""
**Table of Contents**: <br>

<ul>
    <li>1. <a href=\"#1-introduction\">Introduction</a></li>
    <li>2. <a href=\"#2-webscraping\">Webscraping</a></li>
    <li>3. <a href=\"#3-cleaning-eda\">Cleaning & Exploratory Data Analysis (EDA)</a></li>
    <li>4. <a href=\"#4-predictive-modeling\">Predictive Modeling</a></li>
    <li>5. <a href=\"#5-deployment\">Deployment</a></li>
    <li>6. <a href=\"#6-conclusion\">Conclusion</a></li>
</ul>
""", unsafe_allow_html=True)

# 1. Introduction.
st.markdown("## 1. Introduction")

st.markdown("""
    This page will detail my entire end-to-end data science project on Glassdoor.com's data scientist job listings. I've broken down this project into 
    its components, each one representing a separate project in itself. I started this entire series
    of projects in May of 2021. At the time, I had grown comfortable building ML pipelines but never
    venturing into the data science lifecycle. This project was inspired by a lot of *other* 
    projects I came across at the time. I wanted a project useful for others and rich with new libraries and
    cool tools to learn. And so I decided to analyze data scientist job listings. I exercised
    my data science skills on data science and along the way, I learned a myriad of tools and developed
    this amazing project. Without further ado, let's begin.
""")

# 2. Webscraping.
st.markdown("## 2. Webscraping")

image = np.asarray(Image.open("./img/banner/webscraper_banner.PNG"))
st.image(image)

col1, col2, col3 = st.columns([1, 1, 0.5])
with col2:
    st.markdown("### 🎉[Project Here](https://github.com/alckasoc/Joblisting-Webscraper)🎉")

st.markdown("""
    I began my webscraping project in 2021 right when I began this series of projects. I intended
    for the webscraper to be the first of many projects. In the full data science lifecycle, I began with this project
    because it's data ingestion and collection. This project took about a month! I learned an
    assortment of webscraping tools plus quite a few design principles. I also picked up on a few  software engineering conventions
    and tools like `poetry`, how to style a GitHub repo, formatters for code readability like black or flake, and much more.
    This project taught me not just how to collect and ingest data for data science, but also how to employ
    standard principles to organize projects and code. 
""")

col1, col2, col3 = st.columns([1, 1, 0.5])

with col1:
    image = np.asarray(Image.open("./diagrams/webscraper_filter_pipeline.png"))
    st.image(image)
with col2:
    image = np.asarray(Image.open("./diagrams/webscraper_filters_diagram.png"))
    st.image(image)
with col3:
    image = np.asarray(Image.open("./diagrams/webscraper_structure_diagram.png"))
    st.image(image)

st.caption("<figcaption style='text-align: center'>Figure 1. Filter Pipeline (left) | Organizing Filters (middle) | Code Structure (right).</figcaption>", unsafe_allow_html=True)
st.write("\n\n\n")

st.markdown("""
    Above in Figure 1, I spent quite a bit of time planning my project and all throughout the process
    I strove to document and organize my project into these diagrams! The left figure is a simple diagram
    of how I organized my functions to tackle the many filters on Glassdoor's job listing page. The middle 
    figure is a similiar diagram to show how I categorized the many filters and the right figure is a UML
    diagram of my code structure. 
    
    I ran into quite a few difficulties mainly dealing with either code design and structure or 
    navigating around Glassdoor's confusing job listing webpage. I remember simply finding a way
    to select a salary estimate range through my webscraper took days! These numerous challenges were met with 
    more meticulous planning and preparation work. Besides the technical skills, this project taught me
    the importance of planning before beginning.

    Though sadly the webscraper now does not work, the trades I learned from that project
    were invaluable!
""")

# 3. Cleaning & EDA.
st.markdown("## 3. Cleaning & EDA")

# Banner.
image = np.asarray(Image.open("./img/banner/banner_cropped.png"))
st.image(image)
image = np.asarray(Image.open("./img/banner/EDA_my_app_banner.png"))
st.image(image)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("### 🎉[Project Here](https://github.com/alckasoc/Joblisting-Cleaning-EDA)🎉 & \
        🎉[App Here](https://joblisting-cleaning-and-eda.streamlit.app/)🎉")

st.markdown("""
    After finishing my webscraper project, I ran it for a few iterations and amassed a bit over 2000 data science 
    job listings. I began this section of this mega project late May into the beginning of June before I 
    pursued Kaggle. This project remained unfinished for a whole year until October of 2022! That's when I picked
    it up again and finished around early December of 2022. This project undoubtedly developed my exploratory analysis
    skills by ten-fold! I sharpened my data cleaning skills, developed my existing skills with `numpy`, `pandas`, and `matplotlib`, and 
    learned a diverse range of technical tools including `plotly`, `seaborn`, `scipy`, and a bunch of auto-EDA tools.
""")

col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    image = np.asarray(Image.open("./diagrams/joblisting_EDA_cleaning_pipeline.png"))
    st.image(image)

st.caption("<figcaption style='text-align: center'>Figure 2. Cleaning pipeline.</figcaption>", unsafe_allow_html=True)
st.write("\n\n\n")

st.markdown("""
    This project faced many difficulties much like the last. The data was extremely messy. There were inconsistent NaNs,
    incorrect data types, irrelevant columns, data entry errors, misplaced values, string values that needed
    parsing and much more. The cleaning half of this project took weeks! I went through the dataset 
    multiple times before even beginning to construct my preprocessing pipeline. In fact, the cleaning
    process was so complex I split the cleaning pipeline components from the EDA notebook! Of all the
    datasets I've worked with, the one I generated myself required the most amount of care and preparation.
    I lost count of *just* how much I Googled to figure out the best ways to clean the data! 
    Though rigorous, the cleaning paid off in the EDA section of this project. Refer to Figure 2 for some
    of the nuances I tackled in the cleaning process! And if you want a more thorough breakdown, refer to 
    the `EDA Report`.
""")

col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    image = np.asarray(Image.open("./diagrams/EDA Procedure.png"))
    st.image(image)

st.caption("<figcaption style='text-align: center'>Figure 3. EDA Procedure.</figcaption>", unsafe_allow_html=True)
st.write("\n\n\n")

st.markdown("""
    Above in Figure 3 is my EDA procedure. I've modularized my process into components
    and detailed the various tools used at each step of the EDA process. for this side of the 
    EDA & Cleaning project, besides the numerous bugs and problems in the code, I struggled with
    *design*. That is, designing the process I chose, designing the graphs, and structuring all the 
    other components to accompany the master notebook were difficult. And so after a few days of deliberation, I dismantled my process into 
    the following multiple categories:
    univariate/multivariate, graphical/non-graphical, and question-driven. In addition to the technical tools learned,
    I learned a lot of new graphs! For instance, though unrevealing, I implemented a *sankey plot* to describe the 
    feature flows of two categorical variables. Fun fact, this plot took me 3 days to make! If you want
    a more un-summarized version of this EDA procedure, refer to the `EDA Report` again.
""")

# 4. Predictive Modeling.
st.markdown("## 4. Predictive Modeling")

image = np.asarray(Image.open("./img/banner/banner.png"))
st.image(image)

col1, col2, col3 = st.columns([1, 1, 0.5])
with col2:
    st.markdown("### 🎉[Project Here](https://github.com/alckasoc/Joblisting-Modeling)🎉")

st.markdown("""
    This project began early December and I'm now wrapping it up before 2023, the new year!
    The predictive modeling project of this series was more manageable as I've had a good deal
    of experience working with the tools needed. That being said, for this project specifically,
    I aimed to learn as many new things as possible. I learned about AutoML tools that perform
    automatic feature engineering, hyperparameter tuning and the like. I learned about hyperparameter
    tuning tools like `raytune` and `optuna`. I also did quite a bit of digging into ensembling methods,
    hyperparameter tuning methods, feature engineering, importance, and selection methods! All of which
    I do touch on in my project! This predictive modeling project is, without a doubt, my most comprehensive
    and thorough one yet. Fun fact, the banner for this project is the accumulation of images and diagrams from all the other 
    projects and this one!
""")

col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    image = np.asarray(Image.open("./diagrams/ML pipeline.png"))
    st.image(image)

st.caption("<figcaption style='text-align: center'>Figure 4. Generic ML Pipeline.</figcaption>", unsafe_allow_html=True)
st.write("\n\n\n")

st.markdown("""
    In Figure 4 above, I illustrate a generic ML pipeline. In my predictive modeling project, I
    thoroughly investigate each component of this pipeline. I'll walk through this pipeline and describe
    my steps for each component. 
""")

col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    image = np.asarray(Image.open("./diagrams/joblisting_modeling_Experiments_Pipeline.png"))
    st.image(image)

st.caption("<figcaption style='text-align: center'>Figure 5. Experiments Pipeline.</figcaption>", unsafe_allow_html=True)
st.write("\n\n\n")


st.markdown("""
    Let's walk through the ML pipeline in Figure 4 with a supplementary diagram, Figure 5, and some commentary.

    Having saved the cleaned and explored data from the previous project and split the data into train, test, and validation, 
    I implemented, at first, a simple
    preprocessing pipeline. This one applied one-hot encoding to the categorical features and 
    z-normalization or standardization to the numerical features. In addition to this base pipeline,
    I added 2 custom transformers: `CustomRemover` and `CatGrouper`. `CustomRemover` removes unnecessary
    columns and `CatGrouper` groups the rare categorical values for categorical features into 1 class called
    \w"Other\".

    After preprocessing, I dove into a baseline with linear regression and a
    rough test of different model performances. I tested decision
    tree, random forest, extra tree, SVM, XGBoost, CatBoost, LightGBM, and a shallow neural network
    with 5-fold cross validation. And I also explained a little bit behind each algorithm!
    From this list, I chose to move forward with random forest, XGBoost, CatBoost, and LightGBM.

    Then, in CV Experiments, I used EvalML to automate, tune, and engineer an ML pipeline. Further,
    I manually engineered a set of features based off of the numerical, categorical, and also string columns
    of the dataset! More information on this can be found in Figure 5, specifically under the 
    \"Feature Engineering Pipeline\". I added 2 \"output heads\" to this base feature pipeline.
    One head would preprocess with one-hot encoding and z-normalization much like my original preprocessing pipeline.
    The second head would preprocess with hash encoding and robust scaler from sklearn. In my design,
    I had hoped the concatenation of different preprocessing methods plus the engineered features would help!
    I tested these engineered features with the my empirically best model, XGBoost. Unfortunately, of the engineered
    features I tested, none of them were promising. I performed dimensionality reduction on the engineered
    data to save training time as I continued testing the features. I also tested an auto feature engineering tool but
    it didn't prove any better. Because none of these results were promising,
    I opted for the original, simple preprocessing pipeline to move forward with.

    After the feature engineering and DR section, I focused on hyperparameter tuning my best models:
    random forest, CatBoost, LightGBM, and XGBoost. I utilized a variety of tuning libraries like:
    hyperopt, optuna, and raytune. 

    Then came the ensembling stage! I dug a bit into different ensembling methods and opted to test
    2 methods: average ensemble and rank average ensemble. Prior to testing ensembling methods, I had
    trained a 5-fold XGBoost, LightGBM, CatBoost, and random forest. I ensembled these 20 total models
    with the XGBoost EvalML found to be most promising. Surprisingly, EvalML's XGBoost performed the 
    second best with an MAE of 13-14 while my tested XGBoost performed the worst at around 26. CatBoost performed the best with a 5-fold
    MAE of roughly 12-13. I decided to go with rank averaging because it bolstered significant improvement in performance
    compared to a simple average ensemble. 

    The next step was deployment!
""")

# 5. Deployment.
st.markdown("## 5. Deployment")

image = np.asarray(Image.open("./img/banner/train_your_own_model_banner_cropped.png"))
st.image(image)

st.markdown("""
    *You're seeing the deployment!* For deployment, I went with `streamlit` over gradio knowing, one, I'm more familiar with `streamlit` and, 
    second, I planned to incorporate some front-end aspects to the app. I knew `streamlit` had this
    capability. And what you see now is the website layout I decided to go with! I incorporated report-style
    aspects, reflection sections, and also interactive pages. Fun fact, I had the banners/backgrounds for the first three pages
    of this app to be seasonal colors!
""")

# 6. Conclusion.
st.markdown("## 6. Conclusion")

st.markdown("""
    That just about wraps up this entire project! This project took well over 100 hours, countless
    late nights googling, and about 3 or so months to complete. It was an amazing journey and I've written
    about this all throughout this app! Thanks for reading. 👋
""")