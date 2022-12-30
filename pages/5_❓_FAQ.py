# Imports.
import streamlit as st

# Page config.
st.set_page_config(
    page_title="FAQ",
    page_icon="❓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Haha, you found this!",
    }
)

# Title.
st.title("🤔Frequently Asked Questions (FAQ)🤔")

st.write("Well, there are no frequently asked questions. But I'll anticipate a few questions.")

# FAQ.
expand_faq1 = st.expander('''Where's the code?''')
with expand_faq1:
    st.write("""
        Webscraper: https://github.com/alckasoc/Joblisting-Webscraper

        Cleaning & EDA: https://github.com/alckasoc/Joblisting-Cleaning-EDA

        Modeling & Deployment: https://github.com/alckasoc/Joblisting-Modeling
    """)

expand_faq2 = st.expander('''Do you have any resources on the topics you learned?''')
with expand_faq2:
    st.markdown("""
        I'm glad you asked! Check my [Modeling & Deployment](https://github.com/alckasoc/Joblisting-Modeling) project.
        Inside should be a `resources.md`.

        And just for your convenience: https://github.com/alckasoc/Joblisting-Modeling/blob/main/resources.md.
    """)

expand_faq3 = st.expander('''How did you conclude the engineered features to be insignificant?''')
with expand_faq3:
    st.markdown("""
        I tested first with XGBoost's built-in feature importance attribute. Then, afterwards I 
        tested with the LOFO feature importance method. You can check out the library here: https://pypi.org/project/lofo-importance/.
    """)

expand_faq4 = st.expander('''Does the webscraper still work?''')
with expand_faq4:
    st.markdown("""
        Sadly, no. :( There's an issue with initializing the filters. I'm sure with some thorough simplification
        the webscraper will work, but it won't have as much functionality as it had before. As I specified
        in my webscraper project, I won't be maintaining the webscraper. For the sake of streamlining the entire
        data science pipeline from end to end (ingestion to prediction), I looked into other webscrapers
        and Glassdoor's API, but to no avail.
    """)

expand_faq5 = st.expander('''What does your inference pipeline look like?''')
with expand_faq5:
    st.markdown("""
        I have a fitted preprocessing pipeline from training. This pipeline transforms user input
        specified in the `Predictive Modeling` section. This pipeline one-hot encodes the categorical features
        and z-normalizes the numerical features. No feature engineering is done. This preprocessed data is then 
        fed into all 21 models: 5-fold XGBoost, LightGBM, CatBoost, Random Forest, and EvalML's generated XGBoost estimator.
        The predictions are then weighted and summed by each 5-fold mini-ensemble's overall performance on the *test* set (which is a bit biased on my part).
        The ensembling method is rank average. The output of this 21-model ensemble is then printed to the user! 
    """)

expand_faq5 = st.expander('''Is this data fit for modeling?''')
with expand_faq5:
    st.markdown("""
        Hard question. I'd say it is *somewhat* fit for modeling. The main issues with this dataset 
        are specified in my notebook `modeling.ipynb`. But to summarize, the dataset only has about 1700 job listings
        for data scientist positions. The sample size is not ideal. The number of diverse features is also not ideal. All
        features are categorical except for rating and salary estimate (the predicted variable). Additionally, 
        the dataset underwent extensive cleaning. And I also believe collecting more features would've definitely helped
        with modeling. Further, in the case that I would collect data now (assuming the webscraper still worked), the data would be 
        different. Data scientist job listings from December of 2022 would vary greatly from data obtained in May of 2021. There's 
        a huge gap! For a project like this, it'd be even better to continuously gather data, and maintain and retrain the model(s).
        Despite the slight issues, I still found this to be a a very rewarding and comprehensive project! I'm excited for the 
        projects that lay ahead to match or better this project's extensiveness. 
    """)