[<img src="https://github.com/alckasoc/Joblisting-Modeling/blob/main/img/banner/banner.png?raw=true">](https://joblisting-modeling.streamlit.app/)
<figcaption style='text-align: center'>Click above for the app :)!</figcaption>
<br>

# Joblisting-Modeling 

An ML project on the joblistings scraped from the Joblisting Webscraper project and explored in Joblisting Cleaning EDA.

## Table of Contents

- [Motivation](https://github.com/alckasoc/Joblisting-Cleaning-EDA/blob/main/README.md#motivation)
- [Structure](https://github.com/alckasoc/Joblisting-Cleaning-EDA/blob/main/README.md#structure)
- [Dataset](https://github.com/alckasoc/Joblisting-Cleaning-EDA/blob/main/README.md#dataset)
- [Difficulties](https://github.com/alckasoc/Joblisting-Cleaning-EDA/blob/main/README.md#difficulties)
- [What I Learned](https://github.com/alckasoc/Joblisting-Cleaning-EDA/blob/main/README.md#what-i-learned)
- [References](https://github.com/alckasoc/Joblisting-Cleaning-EDA/blob/main/README.md#references)
- [Author Info](https://github.com/alckasoc/Joblisting-Cleaning-EDA/blob/main/README.md#author-info)
- [Thank you](https://github.com/alckasoc/Joblisting-Cleaning-EDA/blob/main/README.md#thank-you)
 
## Motivation

In short, I'm motivated by a want to learn more about the data science pipeline and a desire to venture into the unknown! I've done countless scikit-learn projects, but I pursued this one because I wanted a project that would be more in-depth.

## Structure

![](https://github.com/alckasoc/Joblisting-Modeling/blob/main/diagrams/pipeline_diagram.png?raw=true)\
Figure 1. Data science lifecycle. 
<br/><br/>

This project is part of a *larger* project! This is only 1 step in that larger project. To check out the other projects in this series:
1. [Joblisting-Webscraper](https://github.com/alckasoc/Joblisting-Webscraper)
2. [Joblisting-Cleaning-EDA](https://github.com/alckasoc/Joblisting-Cleaning-EDA)
3. [Joblisting-Modeling](https://github.com/alckasoc/Joblisting-Modeling)

About the structure of this repo:
* `csv` stores the CSVs I generated from the previous project
* `diagrams` stores the diagrams from the past 2 projects and this project
* `img` stores images from the past 2 projects and this project
* `input` stores the dataset I scraped, split and preprocessed data
* `pages` stores the subpage for my app
* `pipelines` stores the pipelines I tested in `modeling.ipynb`
* `1_🧠_Predictive_Modeling.py` is the main page of my streamlit app
* `modeling.ipynb` is the source code
* `resources.md` is a list of all the resources I reference throughout this project
* `utils.py` contains a few helper functions for my app

**Note**: the package versions listed in `requirements.txt` and imported in my code may not be the exact versions. However, the versioning here is less important. I've listed all used libraries.

## Dataset

A little about the dataset: the data was webscraped from Glassdoor.com's job listings for data science jobs. I used my own webscraper for it! That can be found here: https://github.com/alckasoc/Joblisting-Webscraper. The dataset is small and can be found in this repo under `input`. As an alternative, I've also stored this on Kaggle publicly: https://www.kaggle.com/datasets/vincenttu/glassdoor-joblisting.

## Difficulties

Note, I talk more about this in the app! I faced a *ton* of difficulties going into this project. For one, prior to this, I've only ever made simple projects modeling tidy data in fixed environments without too much depth. Venturing into this unknown meant a lot of searching and reading and learning! Along the way, I ran into countless problems code-wise and model-wise. 

## What I Learned

Note, I talk more about this in the app! I learned more about each step of the machine learning pipeline. I've never gone this in-depth in any of the subject matters whether that be feature engineering or hyperparameter tuning. This project I aimed to flush out each and every aspect to the best of my ability. I learned tools like `optuna`, `raytune`, `hyperopt` for hyperparameter tuning. I learned various feature engineering methods and libraries like `lofo`. I learned a bit about AutoML through tools like `autofeat` and `EvalML`. More importantly, I learned about the experimentation process and how crucial it is to have a strong cross validation framework for testing what works and what doesn't work. This is something every Kaggler knows!

## References

For information on references and resources, refer to `resources.md`.

## Author Info

Contact me:

Gmail: tuvincent0106@gmail.com\
Linkedin: [Vincent Tu](https://www.linkedin.com/in/vincent-tu-422b18208/)\
Kaggle: [vincenttu](https://www.kaggle.com/vincenttu)

## Thank you

Psst, I've written another thank you note in my app (check it out). I'd just like to reiterate again that I'm grateful for the tools, documentation, and articles available to me. They have been a great help in my project and without them, this entire project would've been much like any other one I've made! And, thank you, again, [Catherine](https://github.com/crasgaitis) for your help with the visuals and banners! This project is incomplete without you. ❤️

Lastly, thank you for viewing! 