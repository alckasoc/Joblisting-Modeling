# Imports.
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

import xgboost as xgb
from xgboost import XGBRegressor
import catboost as cb
from catboost import CatBoostRegressor
import lightgbm as lgbm
from lightgbm import LGBMRegressor

# Global variables.
seed = 42
n_folds = 5
scoring = "neg_mean_absolute_error"
criterion = mean_absolute_error
my_score = -13.86  # MAE.
key_cnt = 0

# Helper function.
def train_and_test(model, X_train_val, y_train_val, X_test, y_test):
    # Cross validation training.
    scores = cross_val_score(model, X_train_val, y_train_val,
                        cv=n_folds, scoring=scoring)
    
    string = ""
    for fold, score in zip(range(n_folds), scores):
        string += f"Fold-{fold}: {score}\n\n"
    string += f"\nMean Score: {np.mean(scores)}"
    st.info(string)

    # Inference.
    model.fit(X_train_val, y_train_val)
    y_test_pred = model.predict(X_test)
    score = -criterion(y_test, y_test_pred)
    st.success(f"Your model's MAE test score is: {score:.3f}!")

    # If you beat my score!
    if score > my_score: 
        st.balloons()
        st.success(f"Congratulations! You beat my score!")


# Page config.
st.set_page_config(
    page_title="Train Your Own Model!",
    page_icon="👉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Haha, you found this!",
    }
)

# Load the data.
@st.cache
def load_y_data(y_train_val_p, y_test_p):
    return pd.read_csv(y_train_val_p), pd.read_csv(y_test_p)

@st.cache
def load_X_data(X_train_val_p, X_test_p):
    with open(X_train_val_p, "rb") as f:
        X_train_val =  np.load(f)
    with open(X_test_p, "rb") as f:
        X_test =  np.load(f)
    return X_train_val, X_test

X_train_val_p = "./input/X_train_val_prep.npy"
X_test_p = "./input/X_test_prep.npy"
y_train_val_p = "./input/y_train_val.csv"
y_test_p = "./input/y_test.csv"
X_train_val, X_test = load_X_data(X_train_val_p, X_test_p)
y_train_val, y_test = load_y_data(y_train_val_p, y_test_p)
y_train_val = y_train_val.drop(columns=["Unnamed: 0"]).values.ravel()
y_test = y_test.drop(columns=["Unnamed: 0"]).values.ravel()

# Banner.
image = np.asarray(Image.open("./img/banner/train_your_own_model_banner_cropped.png"))
st.image(image)

# Title.
st.title("😲Try Predictive Modeling Yourself!😲")

# Introduction.
st.write("""
    Welcome! This page is dedicated to giving you, the user, the power to model this dataset! 
""")

st.markdown("""
    **A little background**:
    You'll be training 4 different models. So what are these models you're training?\n\n
""")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    image = np.asarray(Image.open("./img/decision_tree.png"))
    st.image(image)
with col2:
    image = np.asarray(Image.open("./img/random_forest.png"))
    st.image(image)
with col3:
    image = np.asarray(Image.open("./img/xgboost.png"))
    st.image(image)

st.caption("<figcaption style='text-align: center'>Figure 1. Decision Tree (left) | Random Forest (middle) | Boosting (right).</figcaption>", unsafe_allow_html=True)
st.write("\n\n\n")

st.markdown("""
    The supervised models you'll be training today will all be built upon **decision trees** (left of Figure 1).
    These \"decision\" trees consist of nodes and arcs that dictate, given an input, what decision, whether that
    be classification or regression, should be made.\n\n

    A **random forest** is a collection of these *trees* (middle of Figure 1). The majority vote from all the trees dictates
    the output of the random forest ensemble model. The random forest model is one of the 4 you'll
    be training!\n\n

    A **boosted forest** is a collection of these trees, but they are arranged in a special way. Namely,
    The trees are organized sequentially and each tree \"learns\" from the mistakes of the previous tree (right of Figure 1).
    In the end, once all these trees are done training, they make for a very robust ensemble model! `CatBoost`,
    `XGBoost`, and `LightGBM` are the 3 staple libraries for implementing cutting edge boosting methods. These
    will be the other 3 models you'll be training!\n\n

    Let's get to training models. 😎
""")

st.markdown("""
    **Details**:
    * model selection & manual hyperparameter are available to you!
    * hyperparameter tuning options are based off of the sweeps I've done in `modeling.ipynb`
    * preprocessing pipeline is fixed; `X` and `y` for train+val and test are already preprocessed and fixed
    * singular model with specified hparams trained with 5-fold cross validation exactly the same as my process
    * singular model with specified hparams tested on same test set as my process
    * user inference is done with a single model, my pipeline describes a 21-model rank-average ensemble (though it is still very possible to beat this ensemble!)
""")
st.markdown("""
    **Note**: The options I provide aren't comprehensive, but they are a strong
    subset of all possible customizations to get you started with tinkering! 
""")
st.write(f"❗Can you beat my score of {my_score} MAE?")

# Tabs.
tab_names = ["Random Forest", "CatBoost", "LightGBM", "XGBoost"]
rf_tab, cb_tab, lgbm_tab, xgb_tab = st.tabs(tab_names)

# Random Forest tab.
with rf_tab:
    st.markdown("""For more information on random forest regressors, check out 
        [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn-ensemble-randomforestregressor).
    """)

    max_depth_range = list(range(10, 1200))
    min_samples_leaf_range = [round(i, 2) for i in list(np.arange(0.1, 0.61, 0.01))]
    min_samples_split_range = [round(i, 2) for i in list(np.arange(0.1, 1.01, 0.01))]
    n_estimators_range = list(range(1, 501))

    max_depth = st.select_slider("Max Depth", options=max_depth_range, value=100)
    max_features = st.selectbox('Max Features', ('auto', 'sqrt', 'log2', None))
    min_samples_leaf = st.select_slider("Min Samples Leaf", options=min_samples_leaf_range, value=0.1)
    min_samples_split = st.select_slider("Min Samples Split", options=min_samples_split_range, value=0.1)
    n_estimators = st.select_slider('N Estimators', options=n_estimators_range, value=1)
    bootstrap = st.selectbox('Bootstrap', (True, False))

    train = st.button("Train & Test!", key=key_cnt)
    key_cnt += 1

    if train: 
        # Instantiate model.
        model = RandomForestRegressor(
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            criterion="absolute_error", random_state=seed
        )
        with st.spinner('Training...'):
            train_and_test(model, X_train_val, y_train_val, X_test, y_test)

# CatBoost tab.
with cb_tab:
    st.markdown("""For more information on CatBoost regressors, check out 
        [CatBoostRegressor](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor).
    """)
    
    max_depth_range = list(range(1, 11))
    lr_range = [round(i, 2) for i in list(np.arange(0.01, 0.31, 0.01))]
    n_estimators_range = list(range(100, 501))
    border_count_range = list(range(5, 201))
    l2_leaf_reg_range = list(range(1, 101))

    max_depth = st.select_slider("Max Depth", options=max_depth_range, value=1)
    lr = st.select_slider("Learning Rate", options=lr_range, value=0.01)
    n_estimators = st.select_slider('N Estimators', options=n_estimators_range, value=100)
    border_count = st.select_slider('Border Count', options=border_count_range, value=5)
    l2_leaf_reg = st.select_slider('L2 Leaf Regularization', options=l2_leaf_reg_range, value=1)

    train = st.button("Train & Test!", key=key_cnt)
    key_cnt += 1

    if train: 
        # Instantiate model.
        model = CatBoostRegressor(
            max_depth=max_depth,
            learning_rate=lr,
            n_estimators=n_estimators,
            border_count=border_count,
            l2_leaf_reg=l2_leaf_reg, random_state=seed
        )
        with st.spinner('Training...'):
            train_and_test(model, X_train_val, y_train_val, X_test, y_test)

# LightGBM tab.
with lgbm_tab:
    st.markdown("""For more information on LightGBM regressors, check out 
        [LightGBM.LGBMRegressor](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html).
    """)

    lr_range = [round(i, 2) for i in list(np.arange(0.01, 0.31, 0.01))]
    n_estimators_range = list(range(100, 10001))
    num_leaves_range = list(range(20, 3001))
    max_depth_range = list(range(1, 14))
    min_data_in_leaf_range = list(range(200, 10001))
    lambda_l1_range = list(range(0, 101))
    lambda_l2_range = list(range(0, 101))
    min_gain_to_split_range = list(range(0, 17))
    bagging_fraction_range = [round(i, 2) for i in list(np.arange(0.2, 0.95, 0.01))]
    feature_fraction_range = [round(i, 2) for i in list(np.arange(0.2, 0.95, 0.01))]

    lr = st.select_slider("Learning Rate", options=lr_range, value=0.01, key=key_cnt)
    key_cnt += 1
    n_estimators = st.select_slider('N Estimators', options=n_estimators_range, value=100)
    num_leaves = st.select_slider('Num Leaves', options=num_leaves_range, value=20)
    max_depth = st.select_slider("Max Depth", options=max_depth_range, value=1)
    min_data_in_leaf = st.select_slider("Min Data In Leaf Range", options=min_data_in_leaf_range, value=200)
    lambda_l1 = st.select_slider("Lambda L1", options=lambda_l1_range, value=0)
    lambda_l2 = st.select_slider("Lambda L2", options=lambda_l2_range, value=0)
    min_gain_to_split = st.select_slider("Min Gain to Split", options=min_gain_to_split_range, value=0)
    bagging_fraction = st.select_slider("Bagging Fraction", options=bagging_fraction_range, value=0.2)
    bagging_freq = 1
    feature_fraction = st.select_slider("Feature Fraction", options=feature_fraction_range, value=0.2)

    train = st.button("Train & Test!", key=key_cnt)
    key_cnt += 1

    if train:
        # Instantiate model.
        model = LGBMRegressor(
            learning_rate=lr,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_data_in_leaf=min_data_in_leaf,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            min_gain_to_split=min_gain_to_split,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            border_count=border_count,
            feature_fraction=feature_fraction, random_state=seed
        )
        with st.spinner('Training...'):
            train_and_test(model, X_train_val, y_train_val, X_test, y_test)

# XGBoost tab.
with xgb_tab:
    st.markdown("""For more information on XGBoost regressors, check out 
        [xgboost.XGBRegressor](https://xgboost.readthedocs.io/en/stable/python/module-xgboost.sklearn).
    """)

    reg_lambda_range = [round(i, 3) for i in list(np.arange(1e-3, 10.001, 0.001))]
    reg_alpha_range = [round(i, 3) for i in list(np.arange(1e-3, 10.001, 0.001))]
    colsample_bytree_range = [round(i, 2) for i in list(np.arange(0.1, 1.1, 0.1))]
    subsample_range = [round(i, 2) for i in list(np.arange(0.1, 1.1, 0.1))]
    lr_range = [round(i, 3) for i in list(np.arange(0.008, 0.021, 0.001))]
    n_estimators_range = list(range(100, 10001))
    max_depth_range = list(range(1, 21))
    min_child_weight_range = list(range(1, 301))

    reg_lambda = st.select_slider("Lambda", options=reg_lambda_range, value=1e-3)
    reg_alpha = st.select_slider("Alpha", options=reg_alpha_range, value=1e-3)
    colsample_bytree = st.select_slider("Col Sample By Tree", options=colsample_bytree_range, value=0.1)
    subsample = st.select_slider("Subsample Fraction", options=subsample_range, value=0.1)
    lr = st.select_slider("Learning Rate", options=lr_range, value=0.008, key=key_cnt)
    key_cnt += 1
    n_estimators = st.select_slider('N Estimators', options=n_estimators_range, value=100, key=key_cnt)
    key_cnt += 1
    max_depth = st.select_slider("Max Depth", options=max_depth_range, value=1)
    min_child_weight = st.select_slider("Min Child Weight", options=min_child_weight_range, value=1)
    
    train = st.button("Train & Test!", key=key_cnt)
    key_cnt += 1

    if train: 
        # Instantiate model.
        model = XGBRegressor(
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            learning_rate=lr,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            eval_metric="mae", random_state=seed
        )
        with st.spinner('Training...'):
            train_and_test(model, X_train_val, y_train_val, X_test, y_test)