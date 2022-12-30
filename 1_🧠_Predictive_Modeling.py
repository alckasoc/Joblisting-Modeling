# Imports.
import os
import base64
import joblib
from glob import glob
from PIL import Image

# Specific imports.
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# Custom imports.
from utils import CustomRemover, CatGrouper, get_pred_ranked_avg

# Helper style functions.
# Ref: https://discuss.streamlit.io/t/how-do-i-use-a-background-image-on-streamlit/5067/6.
def set_bg_hack(main_bg, block_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

    st.markdown(
      f"""
      <style>
      [data-testid="stHeader"] {{
            background-color: rgba(0, 0, 0, 0);
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )

    st.markdown(
      f"""
      <style>
      [data-testid="stVerticalBlock"] {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(block_bg, "rb").read()).decode()});
            padding-bottom: 20px;
            padding-left: 20px;
            padding-right: 20px;
            border-radius: 10px;
            # margin: auto;
            width: 745px;
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )

    st.markdown(
      f"""
      <style>
      .css-o3een2 {{
            background-color: rgba(240,242,246, 0);
            opacity: 0;
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )


def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )

# Page config.
st.set_page_config(
    page_title="Can we Predict Salary From Company Statistics?",
    page_icon=":brain:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Haha, you found this!",
    }
)

# Change background.
# "./img/banner/banner.png"
# "./img/background/green_red_hex_blurred_lightened_cropped1.png"
# "./img/background/white_bg.png"
set_bg_hack("./img/background/green_red_hex_blurred_lightened_cropped1.png",
            "./img/background/white_bg.png")

# Load the data.
@st.cache
def load_data(path):
    return pd.read_csv(path)

data_path = "./input/joblisting_cleaned_withtext.csv"
df = load_data(data_path)

# pipeline & model paths.
pipeline_path = "./pipelines/preprocessing_pipeline.pkl"

save_root_dir = "./models"
catboost_p = os.path.join(save_root_dir, "catboost")
lightgbm_p = os.path.join(save_root_dir, "lightgbm")
random_forest_p = os.path.join(save_root_dir, "random_forest")
xgboost_p = os.path.join(save_root_dir, "xgboost")

catboost_paths = glob(os.path.join(catboost_p, "*.pkl"))
lightgbm_paths = glob(os.path.join(lightgbm_p, "*.pkl"))
random_forest_paths = glob(os.path.join(random_forest_p, "*.pkl"))
xgboost_paths = glob(os.path.join(xgboost_p, "*.pkl"))
xgboost_evalml_path = os.path.join(save_root_dir, "xgboost_evalml.pkl")

# Load models.
rankings = [2, 4, 1, 0, 3]  # In the same order as the model.keys().
rankings_norm = np.array(rankings) / np.sum(rankings)

def load_models(paths):
    models = []
    for p in paths:
        with open(p, "rb") as file:
            models.append(joblib.load(file))
    return models

catboost_models = load_models(catboost_paths)
lightgbm_models = load_models(lightgbm_paths)
random_forest_models = load_models(random_forest_paths)
xgboost_models = load_models(xgboost_paths)
xgboost_evalml_model = load_models([xgboost_evalml_path])

models = {
    "catboost": catboost_models,
    "lightgbm": lightgbm_models,
    "random_forest": random_forest_models,
    "xgboost": xgboost_models,
    "xgboost_evalml": xgboost_evalml_model
}

# Load preprocessing pipeline.
@st.cache(allow_output_mutation=True)
def load_pipeline(path):
    with open(path, "rb") as f:
        pipeline = joblib.load(f)
    return pipeline

pipeline = load_pipeline(pipeline_path)
pipeline.fit(pd.read_csv("./input/X_train_val.csv"))

# Banner.
image = np.asarray(Image.open("./img/banner/banner.png"))
st.image(image)

# Main prediction page.
st.title("The most alluring job this century...")

st.write("Can we predict Data Scientist salaries in California from company statistics?")
st.caption("By Vincent Tu")
st.write("\n\n\n\n\n")

st.markdown("For more information on the process behind this, check out `The Data Science Process` section.")

st.info("Check out the repo: https://github.com/alckasoc/Joblisting-Modeling !", icon="🎉")

# User input features.
company = st.text_input("Company (note, this is not a learned parameter)", "My Company")  # String.
rating_range = [round(i, 2) for i in list(np.arange(0, 5.1, 0.1))]
rating = st.select_slider("Rating", options=rating_range, value=4)  # np.float64.
job_title = st.text_input("Job Title (note, this is not a learned parameter)", "Data Scientist")  # String.
headquarters = st.selectbox("Headquarters", df.headquarters.unique())
job_type = st.selectbox("Job Type", df["job type"].unique())  # String.
size = st.selectbox("Company Size", df["size"].unique())  # String.
company_type = st.selectbox("Company Type", df["type"].unique())  # String.
company_sector = st.selectbox("Company Sector", df["sector"].unique())  # String.
revenue = st.selectbox("Revenue", df["revenue"].unique())  # String.
job_desc = st.text_area("Job Description (note, this is not a learned parameter)", "Our Company is...")  # String.

submit = st.button("Submit")

if submit:
    # Preprocessing.
    user_input = pd.DataFrame([[company, rating, job_title, headquarters, -1,
                            job_type, size, company_type, company_sector,
                            revenue, job_desc]], columns=df.columns)
    user_input = user_input.drop(columns=["salary estimate", "job title", "job description"])
    user_input_prep = pipeline.transform(user_input).toarray()

    # Prediction.
    user_output = get_pred_ranked_avg(models, rankings_norm, user_input_prep, y_test=None)[0]
    user_output = "$" + str(round(user_output, 2)) + "k"
    st.success(f"Based on your provided info, this position has an estimated salary of: {user_output}!")
    st.balloons()


