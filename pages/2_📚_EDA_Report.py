# Imports.
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from st_aggrid import AgGrid

# Page config.
st.set_page_config(
    page_title="Diving into Data Science Jobs",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Haha, you found this!",
    }
)

# Banner.
image = np.asarray(Image.open("./img/banner/banner_cropped.png"))
st.image(image)

# Title.
st.title("🔎Diving into Data Science Jobs🔍")

# Table of contents.
st.markdown("\
**Table of Contents**: <br>\
\
<ul>\
    <li>1. <a href=\"#1-introduction\">Introduction</a></li>\
    <li>2. <a href=\"#2-cleaning\">Cleaning</a></li>\
    <li>3. <a href=\"#3-eda\">Exploratory Data Analysis (EDA)</a>\
        <ul>\
            <li>3.1. <a href=\"#3-1-a-univariate-look\">A Univariate Look</a></li>\
            <li>3.2. <a href=\"#3-2-a-multivariate-dive\">A Multivariate Dive</a></li>\
            <li>3.3. <a href=\"#3-3-a-free-form-exploration\">A Free-form Exploration</a></li>\
        </ul>\
    </li>\
    <li>4. <a href=\"#4-conclusion\">Conclusion</a></li>\
</ul>\
", unsafe_allow_html=True)

# 1. Introduction.
st.markdown("## 1. Introduction")

st.markdown("Repo: [Joblisting-Cleaning-EDA](https://github.com/alckasoc/Joblisting-Cleaning-EDA)")

st.write(
"""
Hello! I'm Vincent and, as of this project, I'm a sophomore at UCSD studying Computer Science
(and data science + AI in my free time). This is my in-depth EDA project into the data science job market!
I hope you enjoy it.
""")

st.write("\
    This project tackled 2 tasks within the data science lifecycle: cleaning & exploring data.\
    A table of contents is provided to direct you. I've organized the sections in this EDA\
    report the same way I organized my project process. After this introduction,\
    I cover my cleaning process and the reasoning behind my process. Then, I cover the\
    the EDA which is broken down into 3 sections: univariate, multivariate, free-form (not all aspects of my `eda.ipynb` are included).\
    Finally, I wrap up this report with a conclusion. I'm really happy with how this\
    project turned out. I hope you enjoy it!\
    \
")

st.write("\
    PS: As a small note, (columns ≡ features ≡ variables) and (rows ≡ instances ≡ joblistings).\
")

# 2. Cleaning.
st.markdown("## 2. Cleaning")

st.markdown("\
    Data was webscraped from my [Joblisting Webscraper](https://github.com/alckasoc/Joblisting-Webscraper) project.\
    I scraped data on 3 separate occasions (these were all scraped in close time proximity to one another in roughly\
    March of 2021 when I began this project). I accumulated a little over two thousand instances.\
    Being real-world data, this dataset was extremely unclean! Some features were lumped together (company and rating).\
    There were lots of NaNs. Joblistings on Glassdoor have multiple optional input fields and\
    scraping those often yielded a good amount of NaNs. Some string features were representing\
    numerical values (like revenue and salary estimate). Different columns also had different ways\
    of representing a NaN. Some represented as \"-1\" and some as \"unknown\".\
")

col1, col2, col3 = st.columns([0.25, 2, 0.5])

image = np.asarray(Image.open("./diagrams/joblisting_EDA_cleaning_pipeline.png").convert("RGB"))
with col1: st.write(" ")
with col3: st.write(" ")
with col2: 
    st.image(image)
st.caption("<figcaption style='text-align: center'>Figure 1. Cleaning Pipeline.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    From the figure, I divided my cleaning pipeline into multiple modularized components (all of which\
    can be easily expanded upon) to handle each feature.\
    In addition to what's already evident from the figure, I decided to remove rows with five\
    or more NaNs because these joblistings were unsalvageable!\
    I also realized that the \"headquarters\" column had a mix up with the job title and the\
    \"company\" column had a few non-company name strings.\
    The \"revenue\" column had a huge amount of unknowns (NaNs) (~25\% of the data), but I decided to keep\
    the column because non-reporting may be interesting to look into.\
    Finally, I noticed quite a few joblistings were from the same company but with a different\
    spelling for the company name. I included a grouper function to fix this.\
")

# 3. EDA
st.markdown("## 3. EDA")

st.write("The following sections concern exploratory data analysis! Both univariate and\
    multivariate sections are divided into non-graphical and graphical.\
")

# 3.1. A Univariate Look.
st.markdown("### 3.1. A Univariate Look")

df = pd.read_csv("./csv/describe.csv")
st.dataframe(df, use_container_width=True)
st.caption("<figcaption style='text-align: left'>Figure 2. Descriptive statistics about rating and salary estimate (in thousands).</figcaption>", unsafe_allow_html=True)
st.write("")

st.write("\
    Average rating is 4.1 with a minimum at 2.1 and a maximum of 5.0. Average salary\
    estimate is around \$133k to \$135k with the minimum at \$24.5k and the maximum at \$209k.\
")

col1, col2, col3 = st.columns([1, 2, 1])

image = np.asarray(Image.open("./img/prop.PNG").convert("RGB"))
with col1: st.write(" ")
with col3: st.write(" ")
with col2:
    st.image(image)
st.caption("<figcaption style='text-align: center'>Figure 3. Proportions of unique companies w.r.t joblistings.</figcaption>", unsafe_allow_html=True)
st.write("")

st.write("\
    Roughly 32\% of the companies in this dataset post 1 joblisting and over 90\%\
    of companies post less than 10 joblistings. Facebook had an abnormally large amount\
    (over 100)! Ratings hovered between 3.5 to 4.5 and most headquarter locations were in\
    San Francisco and Silicon Valley. Most joblistings look for data scientists with working\
    experience (most of the joblistings are for full-time positions). Most joblistings come from\
    public and private companies (not non-profits or schools/universities or contract or government jobs)\
    specifically from the information technology (IT) sector.\
")

col1, col2, col3 = st.columns([1, 10, 1])

image = np.asarray(Image.open("./img/top_30_company_count.png").convert("RGB"))
with col1: st.write(" ")
with col3: st.write(" ")
with col2:
    st.image(image)
st.caption("<figcaption style='text-align: center'>Figure 4. Top 30 Company most frequent companies and their counts.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    Again, as a visual representation, the above bar chart shows *just* how staggering\
    and aggressive Facebook/Meta's hiring behavior is!\
    Interestingly, a few other companies are hiring aggressively.\
    Of course, the larger IT companies will hire more employees, but I did not expect\
    other companies less well-known to be hiring just as much! Who knew Walmart\
    would be so high on this list! I expected most data science joblistings to be\
    from the IT world especially with how much big data they accumulate every month.\
")

col1, col2, col3 = st.columns([1, 2, 1])

image = np.asarray(Image.open("./img/rating_dist.png").convert("RGB"))
with col1: st.write(" ")
with col3: st.write(" ")
with col2:
    st.image(image)
st.caption("<figcaption style='text-align: center'>Figure 5. Distribution of ratings.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    This graph further reinforces the notion that ratings fell most commonly into\
    the 3.5 to 4.5 range. I'd say that this range is expected. Most companies\
    , in order to retain their employees, must operate with a certain level\
    of principle and the ratings reflect this. Most jobs are situated in companies with\
    tolerable work environments. This distribution is left-skewed though, meaning\
    the majority of ratings lean to the right and outliers are seen further to the left.\
    The mean and median for this unimodal distribution seem extremely similar so there aren't many outliers.\
    Do note the ratings fall into buckets (hence the gaps between bars).\
")

col1, col2, col3 = st.columns([1, 2, 1])

image = np.asarray(Image.open("./img/hq_bar_count.png").convert("RGB"))
with col1: st.write(" ")
with col3: st.write(" ")
with col2:
    st.image(image)
st.caption("<figcaption style='text-align: center'>Figure 6. Headquarter counts.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    Unsurprisingly, when joblistings are sourced in California an overwhelming\
    majority of them turn out in Silicon Valley and the San Francisco Bay Area!\
    Though Silicon Valley is known for its tech-y environment and thus its hiring behavior,\
    I never knew San Francisco was the epicenter of all this (or at least from what I'm\
    gleaning from this specific dataset)! There's also a good amount of joblistings from\
    Menlo Park, CA. What companies might reside there?\
")

col1, col2, col3 = st.columns([1, 2, 1])

image = np.asarray(Image.open("./img/salary_est_dist.png").convert("RGB"))
with col1: st.write(" ")
with col3: st.write(" ")
with col2:
    st.image(image)
st.caption("<figcaption style='text-align: center'>Figure 7. Salary estimate.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    From this distribution plot, it seems like most data scientist salaries lie between\
    \$125k to \$160k. It looks unimodal with a slight left skew. The mean and median\
    are close implying there aren't too many outliers. I expect the center of this\
    distribution to be where most common data scientists are (usually an entry or one with\
    a year of experience in the industry already). Again, we note the *extremely* low\
    salary sitting at the \"25\" tick.\
")

col1, col2, col3 = st.columns([1, 2, 1])

image = np.asarray(Image.open("./img/size_bar.png").convert("RGB"))
with col1: st.write(" ")
with col3: st.write(" ")
with col2:
    st.image(image)
st.caption("<figcaption style='text-align: center'>Figure 8. Company Size Counts.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    As expected, a lot of these joblistings come from *large* companies (large in number\
    of employees). What's also fascinating is that it looks like companies of other sizes\
    tend to also hire around the same amount (from our sample). This may not be true in\
    the entire population of data scientist joblistings, but it's interesting to think\
    *just* how large the disparity in hiring is between smaller companies and much larger ones.\
")

col1, col2 = st.columns([1, 1])

image = np.asarray(Image.open("./img/allunique_type_count.png").convert("RGB"))
with col1: 
    st.image(image)

image = np.asarray(Image.open("./img/top_15_sector_count.png").convert("RGB"))
with col2:
    st.image(image)
st.caption("<figcaption style='text-align: center'>Figure 9. Type (left) & Sector (right) Counts.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    Excuse the font on the right graph! It was a fun little experiment playing with different fonts.\
    Most data scientist joblistings come from public and private companies with most companies\
    being from the information technology sector (as mentioned earlier). I highlighted these\
    graphs for another reason: I learned quite a bit about data science *domains*.\
    I knew data scientists usually specialized in a sector, but I had no idea there were\
    this many applications! It seems that though technologically backed, data science\
    has found a great deal of relevance outside of the tech, statistics, and coding bubble.\
    What fascinated me here was biotech & pharmaceuticals. Though incomparable to IT,\
    there was still a huge chunk of joblistings for this sector. I can imagine with\
    growing reliance on software to practice biology, a need for data scientists would grow\
    as they would be the seasoned veterans in wrangling and extracting from this large\
    pool of work!\
")

# 3.2. A Multivariate Dive.
st.markdown("### 3.2. A Multivariate Dive")

df = pd.read_csv("./csv/size_by_rating_desc_stats_ind.csv")
AgGrid(df, use_container_width=True)
df = pd.read_csv("./csv/size_by_salary_desc_stats_ind.csv")
AgGrid(df, use_container_width=True)
st.caption("<figcaption style='text-align: center'>Figure 9. Summary statistics \
    on company size vs rating (top) and company size vs salary estimate (bottom).</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    (Figure 9 top) As expected, a lot of these joblistings come from *large* companies (large in number\
    of employees). What's also fascinating is that it looks like companies of other sizes\
    tend to also hire around the same amount (from our sample). This may not be true in\
    the entire population of data scientist joblistings, but it's interesting to think\
    *just* how large the disparity in hiring is between smaller companies and much larger ones.\
")

st.markdown("\
    (Figure 9 bottom) What I found interesting here is that salary estimate is consistent for both\
    mean and median across all company sizes! Generally, it follows the formula that\
    bigger tech companies pay the best (and that holds true here). Though true, other companies undoubtedly\
    offer competitive salaries. To hire talent, you must be willing to accommodate!\
")

with open("./img/size_by_revenue_of_salary_est.html",'r') as f: 
    html_data = f.read()

st.components.v1.html(html_data, height=500)
st.caption("<figcaption style='text-align: center'>Figure 10. Size by Revenue of Average Salary Estimate.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    From this heatmap, it makes sense that the salary estimates would fall along the main\
    diagonal: larger companies generate more revenue. Ignoring the \"unknown / non-applicable\"\
    row, we see that generally as we move from bottom left to top right, we see brighter colors.\
    This aligns with the idea that data scientists are more well-paid at larger, more successful companies.\
    It's interesting to see the bright yellow spot for companies of size 5001 to 10000 employees\
    with an estimated revenue range of 5 to 10 billion (usd).\
")


with open("./img/sector_by_revenue_of_rating.html",'r') as f: 
    html_data = f.read()

st.components.v1.html(html_data, height=500)
st.caption("<figcaption style='text-align: center'>Figure 11. Sector by Revenue of Average Rating.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    In my notebook `eda.ipynb`, I said this looked like sideways tetris! Some interesting findings:\
    most companies don't report revenue, larger companies tend to report revenue (possibly as leverage for joblistings),\
    and information technology sector-related companies are consistent in reporting revenue and generally\
    seem to have higher ratings. Though, of course, we don't have an equal sample size for each\
    combination of revenue and sector categories.\
")

col1, col2 = st.columns([1, 1])

image = np.asarray(Image.open("./img/autoviz/bar_plot_1_1.png").convert("RGB"))
with col1:
    st.image(image)
image = np.asarray(Image.open("./img/autoviz/bar_plot_1_2.png").convert("RGB"))
with col2: 
    st.image(image)

col1, col2, col3 = st.columns([1, 3, 1])

image = np.asarray(Image.open("./img/autoviz/bar_plot_6_1.png").convert("RGB"))
with col1: st.write(" ")
with col2:
    st.image(image)
with col3: st.write(" ")
st.caption("<figcaption style='text-align: center'>Figure 12. Auto-EDA findings.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    I ran the dataset through a few auto-EDA libraries (autoviz, sweetviz, pandas-profiling, and dtale).\
    The above graphs are a small subset of bar plots generated from autoviz.\
    A few interesting tidbits: internships have on average the highest rating despite\
    having an extremely small sample size in this dataset, contract jobs on average\
    have the highest salary estimate (I'm surprised!), and Sausalito, CA has\
    the highest average rating out of all headquarters in this dataset.\
    A few interesting questions that could be pursued in future exploratory analyses\
    could be: Where is Sausalito, CA and what type of environment is it? What makes it a\
    a good place to work in (if it actually is good)?\
")

col1, col2, col3 = st.columns([1, 3, 1])

image = np.asarray(Image.open("./img/dtale/company_count_by_size_wordcloud.jpeg").convert("RGB"))
with col1: st.write(" ")
with col2:
    st.image(image)
with col3: st.write(" ")
st.caption("<figcaption style='text-align: center'>Figure 13. Company word clouds based on size.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    Just for fun, I generated a word cloud in dtale for unique companies based on company size!\
    Though I recognize a healthy amount of these companies, I never knew their actual estimated\
    size. I think grouping company names into word clouds stratified by size puts into perspective\
    just how large (the obvious) a company is, and, in turn, how successful a company might be.\
    For instance, I never knew Levi Strauss & Co. was such a large company! In the tech industry,\
    tech companies, especially the larger ones, are usually under the limelight. But it's safe\
    to say that there are very large non-IT companies that also utilize a good deal of data science.\
")

# 3.3. A Free-form Exploration.
st.markdown("### 3.3. A Free-form Exploration")

col1, col2, col3 = st.columns([1, 3, 1])

image = np.asarray(Image.open("./img/job_desc_wordcloud.png").convert("RGB"))
with col1: st.write(" ")
with col2:
    st.image(image)
with col3: st.write(" ")
st.caption("<figcaption style='text-align: center'>Figure 14. Most frequent job description words.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    I made another word cloud for most frequent words in job descriptions for data scientist\
    positions. Teamwork, collaboration, management, problem solving, working experience, customer-centric approach,\
    and especially machine learning are all very integral to a data scientist's toolkit.\
    This word cloud is very interesting. It reflects the multi-discipline nature of data scientists.\
    Data scientists aren't just programmers, or statisticians, or engineers, or researchers.\
    They are ultimately a fusion of multiple different paths with their core purpose of leveraging\
    data to understand and tackle business problems. They are an ensemble of many different specializations!\
")

with open("./img/sankey_jobtitle_jobdesc.html",'r') as f: 
    html_data = f.read()

st.components.v1.html(html_data, height=500)
st.caption("<figcaption style='text-align: center'>Figure 15. Sankey plot between job positions and job skills.</figcaption>", unsafe_allow_html=True)
st.write("")

st.markdown("\
    And again, out of curiosity, I wanted to investigate how a certain selection of\
    skills were distributed across different job positions. A plot fitting for this was the\
    Sankey plot. Sadly, the plots I created seemed pretty uniform (and showed a lack of sufficient sample size).\
    Though even then, I suppose it is useful to know that all these paths within the data science\
    career require some subset of the specified skills. Both the soft and hard skills mentioned\
    are crucial for any data science position.\
")

col1, col2 = st.columns([1, 2])

image = np.asarray(Image.open("./img/stats_tests_conditions.png").convert("RGB"))
with col2:
    st.image(image)
with col1:
    st.markdown("\
        With our previous investigations into salary and rating, I wanted to also further reinforce\
        where the salary and rating were \"concentrated\". In essence, I conduct a CI to find a confident finite\
        range for the mean rating/salary estimate in the population (not our sample).\
        I conduct a t-test to find reason to believe that our sample is different from the general population.")

    st.markdown("\
        Thus, I conducted a 95\% CI and a one-sample t-test for sample\
        mean with an alpha of 5\% for both numerical features. All assumptions/conditions were checked.\
        From the 95\% CI, we can say we are 95\% confident the mean population rating\
        is between 4.02 and 4.06 and the mean population salary estimate is between \$131.9k and \$134.3k.\
        The margin of error, or the uncertainty, of both CI results are small (notice the range for both CIs are small).\
        Though not concrete evidence, we can *expect* the population to have mean ratings and salary estimates\
        for data scientist positions on Glassdoor.com to be generally within this range.\
        Our p-value for both categories were 0.999 and 1, respectively. Again,\
        there is reason to believe that our sample is indicative (to some degree)\
        of the population mean for both categories.\
    ")

st.write("")

st.markdown("\
    Lastly, I concluded this EDA with a few probing questions.\
")

st.markdown(
"""
- Which company offers the highest salary? Highest rating? How about the lowest for each?
Indeed offers the highest salary at \$209k and Oak Valley Hospital offers the lowest at \$24.5k.
Wing has the lowest rating at 2.1 and Grammarly Inc. had the highest at 5.0.

- Which companies in our dataset have the highest average salary estimate?
Fanatics and Databricks offer the highest salaries though these are for principal data scientist positions.

- We know Facebook (Meta) to have over 100 joblistings in our dataset. They are all for data science positions. But are they all for the same one?
Interestingly, no! They were hiring a diverse army of data scientists for emerging business, product analytics, and marketing.
In retrospect (since this dataset was collected in 2021), I now understand that this huge hiring surge was
for the metaverse!

- What does Ursus specialize in and why, out of all companies in our dataset, do they have the 2nd highest number of joblistings?
I dug a little into this and saw that Ursus, a company specializing in connecting talented 
candidates with companies, witnessed a lot of growth the previous year which could explain all the joblistings they have
in my dataset!
"""
)

# 4. Conclusion.
st.markdown("## 4. Conclusion")

st.markdown(
"""
Wow. If you're reading this, you made it all the way to the end. This report took ages! But it
was very rewarding. I learned a great deal about Streamlit. I learned how to embed different 
types of media, how to make multipage apps, and how to organize and format the page with containers.
Apart from that, this EDA has certainly taught me a great deal about the data science job market. 
I hope it taught you just as much as it has taught me. Thank you for your time!
"""
)