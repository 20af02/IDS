# %% [markdown]
# # Background

# %% [markdown]
# ## Introduction to Data Science - DS UA 112 Capstone Project
# The purpose of this capstone project is to tie everything we learned in this class
# together. This might be challenging in the short term, but is consistently rated as being
# extremely valuable in the long run. The cover story is that you are working as a Data
# Scientist for an auction house that works with a major art gallery. You want to better
# understand this space with the help of Data Science. Historically, this domain was
# dominated by art historians, but is increasingly informed by data. This is where you – a
# budding Data Scientist – come in. Can you provide the value that justifies your rather
# high salary?
#
# __Mission command preamble:__ As in general, we won’t tell you *how* to do something. That
# is up to you and your creative problem solving skills. However, we will pose the questions
# that you should answer by interrogating the data. Importantly, we do expect you to do this
# work yourself, so it reflects your intellectual contribution – not that of third parties.
# By doing this assignment, you certify that it indeed reflects your individual intellectual
# work.
#
# __Format:__ The project consist of your answers to 10 (equally-weighed, grade-wise.
# questions. Each answer *must* include some text (describing both what you did and what you
# found, i.e. the answer to the question), a figure that illustrates the findings and some
# numbers(e.g. test statistics, confidence intervals, p-values or the like.. Please save it
# as a pdf document. This document should be 4-6 pages long (arbitrary font size and
# margins). About half a page per question is reasonable. In addition, open your document
# with a brief statement as to how you handled preprocessing (e.g. dimension reduction, data
# cleaning and data transformations), as this will apply to all answers. Include your name.
#
# __Academic integrity:__ You are expected to do this project by yourself, individually, so
# that we are able to determine a grade for you personally. There are enough degrees of
# freedom (e.g. how to clean the data, what variables to compare, aesthetic choices in the
# figures, etc.) that no two reports will be alike. We’ll be on the lookout for suspicious
# similarities, so please refrain from collaborating. To prevent cheating (please don’t do
# this – it is easily detected), it is very important that you – at the beginning of the
# code file – seed the random number generator with your N-number. That way, the correct
# answers will be keyed to your own solution (as this matters, e.g. for the specific train/
# test split or bootstrapping). As N-numbers are unique, this will also protect your work
# from plagiarism. __Failure to seed the RNG in this way will also result in the loss of
# grade points.__
#
#
# __Deliverables:__ Upload two files to the Brightspace portal by the due date in the
# sittyba:
#
# * A pdf (the “project report”) that contains your answers to the 10 questions, as well as
# an introductory paragraph about preprocessing.
#
# * A .py file with the code that performed the data analysis and created the figures.
#
# We do wish you all the best in executing on these instructions. We aimed at an optimal
# balance between specificity and implementation leeway, while still allowing us to grade
# the projects in a consistent and fair manner.
#
# Everything should be doable from what was covered in this course. If you take this project
# seriously and do a quality job, you can easily use it as an item in your DS portfolio.
# Former students told us that they secured internships and even jobs by well executed
# capstone projects that impressed recruiters and interviewers.
# ___

# %% [markdown]
# ## Dataset Description
# Description of dataset: This dataset consists of data from 300 users and 91 art pieces.
# The art pieces are described in the file `theArt.csv`. Here is how this file is structured:
#
# **Rows**
# * 1st row: Headers (e.g. number, title, etc.)
# * Rows 2-92: Information about the 91 individual art piece
#
# **Columns**
# * Column 1: “Number” (the ID number of the art piece)
# * Column 2: Artist
# * Column 3: Title
# * Column 4: Artistic style
# * Column 5: Year of completion
# * Column 6: Type code – 1 = classical art, 2 = modern art, 3 = non-human art
# * Column 7: computer or animal code – 0 = human, 1 = computer generated art, 2 = animal art
# * Column 8: Intentionally created? 0 = no, 1 = yes
#
# You can also take a look at the actual art by looking at the files in the `artPieces`
# folder on Brightspace.
#
# The user data is contained in the file `theData.csv`. Here is how this file is structured:
#
# **Rows**
# * Rows 1-300: Responses from individual users
#
# **Columns**
# * Columns 1-91: Preference ratings (liking) of the 91 art pieces.
#
# The column number in this file corresponds to the number of the art piece in column 1 of “theArt.csv” file described
# above. For instance, ratings of art piece 27 (“the woman at the window”) is in column 27. Numbers represent
# preference ratings from 1 (“hate it”) to 7 (“love it”).
#
# * Columns 92-182: “Energy” ratings of the same 91 art pieces (in the same order as the
# preference ratings above).
#
# Numbers represent ratings from 1 (“it calms me down a lot”) to
# 7 (“it agitates me a lot”).
#
# * Columns 183-194: “Dark” personality traits.
#
# Numbers represent how much a user agrees with a statement, from 1 (strongly disagree) to 5 (strongly agree). Here are
# the 12 statements, in column order:
#
# 1. I tend to manipulate others to get my way
# 2. I have used deceit or lied to get my way
# 3. I have used flattery to get my way
# 4. I tend to exploit others towards my own end
# 5. I tend to lack remorse
# 6. I tend to be unconcerned with the morality of my actions
# 7. I can be callous or insensitive
# 8. I tend to be cynical
# 9. I tend to want others to admire me
# 10. I tend to want others to pay attention to me
# 11. I tend to seek prestige and status
# 12. I tend to expect favors from others
#
# * Columns 195-205: Action preferences. Numbers represent how much a user agrees with a
# statement, from 1 (strongly disagree) to 5 (strongly agree). Here are the 11 actions, in
# column order:
#
# 1. I like to play board games
# 2. I like to play role playing (e.g. D&D) games
# 3. I like to play video games
# 4. I like to do yoga
# 5. I like to meditate
# 6. I like to take walks in the forest
# 7. I like to take walks on the beach
# 8. I like to hike
# 9. I like to ski
# 10. I like to do paintball
# 11. I like amusement parks
#
#
# * Columns 206-215: Self-image/self-esteem. Numbers represent how much a user agrees with a
# statement. Note that if a statement has “reverse polarity”, e.g. statement 2 “at times I
# feel like I am no good at all”, it has already been re-coded/inverted by the professor
# such that higher numbers represent higher self-esteem. Here are the 10 items, in column
# order:
#
# 1. On the whole, I am satisfied with myself
# 2. At times I think I am no good at all
# 3. I feel that I have a number of good qualities
# 4. I am able to do things as well as most other people
# 5. I feel I do not have much to be proud of
# 6. I certainly feel useless at times
# 7. I feel that I'm a person of worth, at least on an equal plane with others
# 8. I wish I could have more respect for myself
# 9. All in all, I am inclined to feel that I am a failure
# 10. I take a positive attitude toward myself
#
# * Column 216: User age
# * Column 217: User gender (1 = male, 2 = female, 3 = non-binary)
# * Column 218: Political orientation (1 = progressive, 2 = liberal, 3 = moderate, 4 =
# conservative, 5 = libertarian, 6 = independent)
#
# * Column 219: Art education (The higher the number, the more: 0 = none, 3 = years of art
# education)
#
# * Column 220: General sophistication (The higher, the more: 0 = not going to the opera, etc.
# 3 = doing everything – opera, art galleries, etc.)
#
# * Column 221: Being somewhat of an artist myself? (0 = no, 1 = sort of, 2 = yes, I see
# myself as an artist)
#
# Note that we did most of the data munging and coding for you already but you still need to
# handle missing data in some way (e.g. row-wise removal, element-wise removal, imputation).
# Extreme values might also have to be handled.
# ___

# %% [markdown]
# ## Questions Management Would Like You to Answer
# 1. Is classical art more well liked than modern art?
# 2. Is there a difference in the preference ratings for modern art vs. non-human (animals and computers) generated
# art?
# 3. Do women give higher art preference ratings than men?
# 4. Is there a difference in the preference ratings of users with some art background (some art education) vs. none?
# 5. Build a regression model to predict art preference ratings from energy ratings only. Make sure to use
# cross-validation methods to avoid overfitting and characterize how well your model predicts art preference ratings.
#
# 6. Build a regression model to predict art preference ratings from energy ratings and demographic information. Make
# sure to use cross-validation methods to avoid overfitting and comment on how well your model predicts relative to the
# “energy ratings only” model.
#
# 7. Considering the 2D space of average preference ratings vs. average energy rating (that contains the 91 art pieces
# as elements), how many clusters can you – algorithmically - identify in this space? Make sure to comment on the
# identity of the clusters – do they correspond to particular types of art?
#
# 8. Considering only the first principal component of the self-image ratings as inputs to a regression model – how
# well can you predict art preference ratings from that factor alone?
#
# 9. Consider the first 3 principal components of the “dark personality” traits – use these as inputs to a regression
# model to predict art preference ratings. Which of these components significantly predict art preference ratings?
# Comment on the likely identity of these factors (e.g. narcissism, manipulativeness, callousness, etc.).
#
# 10. Can you determine the political orientation of the users (to simplify things and avoid gross class imbalance
# issues, you can consider just 2 classes: “left” (progressive & liberal) vs. “non-left” (everyone else)) from all the
# other information available, using any classification model of your choice? Make sure to comment on the
# classification quality of this model.
#
# __Extra credit:__ Tell us something interesting about this dataset that is not trivial and not already part of
# an answer (implied or explicitly) to these enumerated questions.
#
# __Hints:__
# * Beware of off-by-one errors. This document and the csv data files index from 1, but Python indexes from 0. Make
# sure to keep track of this.
# * In order to answer some of these questions, you might have to apply a dimension reduction method first. For
# instance, “dark personality traits” and “self-image” are characterized by 10-12 variables each. Similarly, you might
# have to reduce variables to their summary statistics.
#
# * In order to do some analyses, you will have to clean the data first, either by removing or imputing missing data
# (either is fine, but explain and justify what you did)
#
# * If you encounter skewed data, you might want to transform the data first, e.g. by z-scoring
# * To clarify: When talking about “principal components” above, we mean the transformed data, rotated into the new
# coordinate system by the PCA.
#
# * Avoid overfitting with cross-validation methods.
#
# * How well your model predicts can be assessed with RMSE or $R^2$ for regression models, or AUC for classification
# models.
#
# * You can use conventional choices of alpha (e.g. 0.05) or confidence intervals (e.g. 95%) throughout
# ___

# %% [markdown]
# # Coding Portion

# %% [markdown]
# ## Initial Setup

# %% [markdown]
# ### Imports
#

# %%
from sklearn.model_selection import StratifiedKFold
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import requests
import random
import warnings

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix, roc_auc_score, roc_curve, auc, silhouette_samples, r2_score, silhouette_score, explained_variance_score

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import stats
from scipy import stats as st


from mlxtend.plotting import plot_decision_regions
from sklearn.manifold import TSNE

import xgboost as xgb
import os
from tqdm import tqdm

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# %% [markdown]
# ### Seeding the random number generator with my student ID

# %%
N_NUMBER = 14254767
random.seed(N_NUMBER)

# %% [markdown]
# ### Loading the data & Helper Functions

# %%
user_ratings = pd.read_csv('Data/theData.csv', sep=',',
                           header=None, engine='python')
art_ratings = pd.read_csv('Data/theArt.csv', sep=',')


def select_df(df, columns):
    return df[columns].dropna()


# Helper function to join, drop na, then separate ratings from two indicies
def join_drop_na_separate_ratings(df, indicies1, indicies2):
    # Combine both indicies
    indicies = np.concatenate((indicies1, indicies2))

    # Get ratings for both indicies
    ratings = select_df(df, indicies)

    # Drop NA rows for both indicies
    ratings_no_na = ratings.dropna()

    # Select ratings with index1
    ratings1 = select_df(ratings_no_na, indicies1)

    # Select ratings with index2
    ratings2 = select_df(ratings_no_na, indicies2)

    return ratings1, ratings2


# %%
# Plotting functions

def plot_ratings_histogram_sbs(ratings, title, id):
    plt.subplot(1, 2, id)
    plt.hist(ratings.values.flatten(), bins=7,
             density=False, histtype='bar', ec='black')
    plt.title(title)
    plt.xlabel('Rating')
    plt.ylabel('Count')


def plot_ratings_histogram(title1, title2, ratings1, ratings2):
    common_str = "\n(#users = {}, #art pieces = {})"
    plot_ratings_histogram_sbs(
        ratings1, title1 + common_str.format(ratings1.shape[0], ratings1.shape[1]), 1)
    plot_ratings_histogram_sbs(
        ratings2, title2 + common_str.format(ratings2.shape[0], ratings2.shape[1]), 2)
    plt.tight_layout()
    plt.show()


def plot_ratings_histogram_wl(title, ratings1, label1, ratings2, label2):
    x = [ratings1.values.flatten(
    ), ratings2.values.flatten()]
    colors = ['blue', 'red']

    plt.plot(figsize=(10, 5))
    plt.hist(x, density=True, alpha=0.75)

    # Ensure plots side by side middle matches with tick

    # Include kde plot
    g_kde_modern = st.gaussian_kde(x[0])
    g_kde_classical = st.gaussian_kde(x[1])
    # linespace for kde
    x_kde = np.linspace(0, 7, 1000)

    plt.plot(x_kde, g_kde_modern(x_kde), color='blue', alpha=0.5)
    plt.plot(x_kde, g_kde_classical(x_kde), color='red', alpha=0.5)

    plt.title(title)
    plt.xlabel('Rating')

    plt.ylabel('Frequency')
    plt.legend([label1, label2,
                label1, label2])
    plt.tight_layout()

    plt.tight_layout()
    plt.show()


# %% [markdown]
# #### 3d Plotting Function

# %%
def plot_3d(ax, x_test, y_test, y_pred, num_range, user_num):

    num_ = [x for x in range(0, num_range)]
    ax.scatter(num_, x_test[:num_range, user_num],
               y_test[:num_range, user_num], c='r', marker='o', label='Actual')

    ax.scatter(num_, x_test[:num_range, user_num],
               y_pred[:num_range, user_num], c='b', marker='o', label='Predicted')

    ax.set_xlabel('Art Piece Number')

    ax.set_ylabel('Energy Rating')

    ax.set_zlabel('Preference Rating')

    # Draw lines between the predicted and actual values

    for i in range(0, num_range):
        ax.plot([num_[i], num_[i]], [x_test[i, user_num], x_test[i, user_num]], [
            y_test[i, user_num], y_pred[i, user_num]], c='g', label='residual' if i == 0 else "")
    ax.legend(loc='upper right')
    ax.set_title('User {}'.format(user_num))


def plot_3d_prf(x_test, y_test, y_pred, rmse, tmp_str, num_range=90, user_num=0):
    fig = plt.figure(figsize=(30, 15))
    fig.suptitle('Actual vs. Predicted Preference Ratings for {} Art Pieces in test dataset.\n'.format(
        num_range)+tmp_str+'\nRMSE={:.3f}'.format(rmse))

    axes = []
    for i in range(2):
        for j in range(4):
            # Create an Axes3D object for each subplot
            ax = fig.add_subplot(2, 4, i*4 + j + 1, projection='3d')
            axes.append(ax)

    for i in range(2):
        for j in range(4):
            plot_3d(axes[i*4 + j], x_test, y_test,
                    y_pred, num_range, i*4+j)
    plt.tight_layout()
    plt.show()


# %%
# Plot importances for extreme gradient boosting
def plot_importances(importances, num_x, title):
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.bar(range(num_x), importances[indices])
    plt.xticks(range(num_x), indices+1, rotation=90, fontsize=6)
    plt.xlim([-1, num_x])
    plt.xlabel('Art Piece Number')
    plt.ylabel('Importance')
    plt.show()


# %% [markdown]
# #### Data Cleaning

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

sns.heatmap(user_ratings.isnull(), ax=ax[0])
ax[0].set_title('User Ratings Dataset')
ax[0].set_xlabel('Response Information Column')
ax[0].set_ylabel('User ID')


sns.heatmap(art_ratings.isnull(), ax=ax[1])
ax[1].set_title('Art Information Dataset')
ax[1].set_xlabel('Art Information Column')
ax[1].set_xticklabels([i for i in range(art_ratings.shape[1])])
ax[1].set_ylabel('Art ID')

# %% [markdown]
# # 1. Is classical art more well liked than modern art?

# %% [markdown]
# ## Data Loading & Cleaning

# %%
# modern art indicies
modern_indicies = np.where(
    art_ratings['Source (1 = classical, 2 = modern, 3 = nonhuman)'] == 2)[0]


# classical art indicies
classical_indicies = np.where(
    art_ratings['Source (1 = classical, 2 = modern, 3 = nonhuman)'] == 1)[0]


# Get ratings for modern and classical art
modern_art_ratings, classical_art_ratings = join_drop_na_separate_ratings(
    user_ratings, modern_indicies, classical_indicies)

# %% [markdown]
# ## Data Visualization

# %%
# PLOTS
plot_ratings_histogram('Modern Art Ratings', 'Classical Art Ratings',
                       modern_art_ratings, classical_art_ratings)

# %%
plot_ratings_histogram_wl('Modern Art Ratings vs Classical Art Ratings', modern_art_ratings,
                          'Modern Art Ratings', classical_art_ratings, 'Classical Art Ratings')


# %% [markdown]
# ## Significance Testing

# %%
# Do a mann whitney test on modern art ratings vs classical art ratings (is classical more well liked than modern)

# H0: classical art is less or equally liked as modern art
# H1: classical art is more liked than modern art
# alpha = 0.05

u, p = stats.mannwhitneyu(classical_art_ratings.values.flatten(
), modern_art_ratings.values.flatten(), alternative='greater')

print('p-value = ' + str(p) + ' u = ' + str(u))
if p < 0.05:
    print('Reject H0, evidence suggests that classical art is more liked than modern art')
else:
    print('Fail to reject H0, evidence suggests that classical art is less or equally liked as modern art')


# %%
# Use KS test to see if modern art ratings and classical art ratings are from the same distribution

# H0: classical art is less or equally liked as modern art
# H1: classical art is more liked than modern art

# alpha = 0.05
ks, p = st.kstest(classical_art_ratings.values.flatten(),
                  modern_art_ratings.values.flatten())
# Less -> test modern < classical.


print('p-value = ' + str(p) + ' ks = ' + str(ks))
if p < 0.05:
    print('Reject H0, evidence suggests that classical art is more liked than modern art')
else:
    print('Fail to reject H0, evidence suggests that classical art is less or equally liked as modern art')


# %% [markdown]
# # 2. Is there a difference in the preference ratings for modern art vs. non-human (animals and computers) generated art?

# %% [markdown]
# ## Data Loading & Cleaning

# %%
# Modern Art indicies
modern_art_indicies = np.where(
    art_ratings['Source (1 = classical, 2 = modern, 3 = nonhuman)'] == 2)[0]

# Non-human art indicies
non_human_indicies = np.where(
    art_ratings['Source (1 = classical, 2 = modern, 3 = nonhuman)'] == 3)[0]


# Get ratings for modern and non-human art
ratings_for_modern_art, ratings_for_non_human_art = join_drop_na_separate_ratings(
    user_ratings, modern_art_indicies, non_human_indicies)

# %% [markdown]
# ## Data Visualization

# %%
plot_ratings_histogram('Modern Art Ratings', 'Non-Human Art Ratings',
                       ratings_for_modern_art, ratings_for_non_human_art)


# %%
plot_ratings_histogram_wl('Modern Art Ratings vs Non-Human Art Ratings', ratings_for_modern_art,
                          'Modern Art Ratings', ratings_for_non_human_art, 'Non-Human Art Ratings')

# %% [markdown]
# ## Significance testing

# %%
# We can see that the distributions look very different, and that the non-human art has < 35 art pieces, so we will use the KS test to test the null hypothesis

# H0: There is not difference in the preferences for modern art vs. non-human art

# H1: There is a difference in the preferences for modern art vs. non-human art

# Use the mann whitney test to test the null hypothesis

u, p = stats.mannwhitneyu(ratings_for_modern_art.values.flatten(
), ratings_for_non_human_art.values.flatten(), alternative='two-sided')

print('u = ' + str(u) + ', p = ' + str(p))
if p < 0.05:
    print('Reject the null hypothesis, evidence suggests that there is a difference in the preferences for modern art vs. non-human art')
else:
    print('Fail to reject the null hypothesis, evidence suggests that there is not a difference in the preferences for modern art vs. non-human art')


# %%
# Use the KS test to test the null hypothesis

# H0: There is not difference in the preferences for modern art vs. non-human art

# H1: There is a difference in the preferences for modern art vs. non-human art

ks, p = st.kstest(ratings_for_modern_art.values.flatten(),
                  ratings_for_non_human_art.values.flatten())

print('ks = ' + str(ks) + ', p = ' + str(p))
if p < 0.05:
    print('Reject the null hypothesis, evidence suggests that there is a difference in the preferences for modern art vs. non-human art')
else:
    print('Fail to reject the null hypothesis, evidence suggests that there is not a difference in the preferences for modern art vs. non-human art')


# %% [markdown]
# # 3. Do women give higher art preference ratings than men?

# %% [markdown]
# ## Data Loading & Cleaning

# %%
# Is there a positivie difference in the ratings between women and men (women-men) > 0

# Male ratings (user gender is at column 216, 1=male)
# get males (ratings from columns 0-90)
male_ratings = user_ratings[user_ratings[216]
                            == 1][user_ratings.columns[0:91]].dropna()

# Female ratings
female_ratings = user_ratings[user_ratings[216] ==
                              2][user_ratings.columns[0:91]].dropna()


# %% [markdown]
# #### Data Visualization

# %%
# Male vs female ratings
plot_ratings_histogram('Male Ratings', 'Female Ratings',
                       male_ratings, female_ratings)

# %%
plot_ratings_histogram_wl('Male Ratings vs Female Ratings for Art',
                          male_ratings, 'Male Ratings', female_ratings, 'Female Ratings')

# %% [markdown]
# ## Significance Testing

# %%
# H0: The difference in art preference ratings between women and men (women-men) is less than or equal to 0

# HA: The difference in art preference ratings between women and men (women-min) is greater than 0

# Use the Mann Whitney U test to test the null hypothesis

u, p = stats.mannwhitneyu(female_ratings.values.flatten(),
                          male_ratings.values.flatten(), alternative='greater')

print('u = ' + str(u) + ', p = ' + str(p))
if p < 0.05:
    print('Reject the null hypothesis, evidence suggests that the difference in art preference ratings between women and men (women-men) is greater than 0')
else:
    print('Fail to reject the null hypothesis, evidence suggests that the difference in art preference ratings between women and men (women-men) is less than or equal to 0')


# %%
# Use the KS test to test the null hypothesis

ks, p = stats.ks_2samp(female_ratings.values.flatten(),
                       male_ratings.values.flatten(), alternative='two-sided')

print('ks = ' + str(ks) + ', p = ' + str(p))
if p < 0.05:
    print('Reject the null hypothesis, evidence suggests that the difference in art preference ratings between women and men (women-men) is greater than 0')
else:
    print('Fail to reject the null hypothesis, evidence suggests that the difference in art preference ratings between women adn men (women-men) is less than or equal to 0')


# %% [markdown]
# # 4. Is there a difference in the preference ratings of users with some art background (some art education) vs. none?

# %% [markdown]
# ## Data Loading & Cleaning

# %%
no_art_edu_ratings = user_ratings[user_ratings[218] ==
                                  0][user_ratings.columns[0:91]].dropna()

# !=0 may include na values, so we use >
some_art_edu_ratings = user_ratings[user_ratings[218] >
                                    0][user_ratings.columns[0:91]].dropna()

# %% [markdown]
# ## Data Visualization

# %%
plot_ratings_histogram('No Art Education Ratings',
                       'Some Art Education Ratings', no_art_edu_ratings, some_art_edu_ratings)

# %%
plot_ratings_histogram_wl('No Art Education Ratings vs Some Art Education Ratings', no_art_edu_ratings,
                          'No Art Education Ratings', some_art_edu_ratings, 'Some Art Education Ratings')

# %% [markdown]
# ## Significance Testing

# %%
# H0: There is no difference in the preference ratings of users with some art background vs none

# HA: There is a difference in the preference ratings of users with some art background vs none

# Use mannwhitneyu test
u, p = stats.mannwhitneyu(no_art_edu_ratings.values.flatten(
), some_art_edu_ratings.values.flatten(), alternative='two-sided')
print('u = ' + str(u) + ', p = ' + str(p))

if p < 0.05:
    print('Reject the null hypothesis, evidence suggest that there is a difference in the preference ratings of users with some art background vs none')
else:
    print('Fail to reject the null hypothesis, evidence suggest that there is no difference in the preference ratings of users with some art background vs none')

# %%
ks, p = stats.ks_2samp(no_art_edu_ratings.values.flatten(
), some_art_edu_ratings.values.flatten(), alternative='two-sided')


print('ks = ' + str(ks) + ', p = ' + str(p))

if p < 0.05:
    print('Reject the null hypothesis, evidence suggest that there is a difference in the preference ratings of users with some art background vs none')
else:
    print('Fail to reject the null hypothesis, evidence suggest that there is no difference in the preference ratings of users with some art background vs none')


# %% [markdown]
# # 5. Build a regression model to predict art preference ratings from energy ratings only. Make sure to use cross-validation methods to avoid overfitting and characterize how well your model predicts art preference ratings.

# %% [markdown]
# ## Data Loading & Cleaning

# %%
# X -> energy rating of art piece
# Y -> preference rating of art piece

both_ratings = user_ratings[user_ratings.columns[0:182]].dropna()

# Get X (energy rating of each art piece for all users)
energy_ratings = both_ratings[both_ratings.columns[91:182]]
energy_ratings.index -= 91

# Get Y (preference rating of each art piece for all users)
preference_ratings = both_ratings[both_ratings.columns[0:91]]


x = energy_ratings.values
y = preference_ratings.values

# Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=N_NUMBER)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %% [markdown]
# ## Elastic-Net Regularized Regression

# %%
es_net = ElasticNet(alpha=0.002, normalize=True)

es_net.fit(x_train, y_train)

# Predict
y_pred = es_net.predict(x_test)
slope = es_net.coef_  # B1 (slope)
intercept = es_net.intercept_  # B0 (intercept)

# Print Summary Statistics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', rmse)

tmp_str = 'predicted preference = '
# show equation
for i in range(0, len(slope)):
    tmp_str += str(slope[i]) + ' * x' + str(i) + ' + '
tmp_str += str(intercept)
# print(tmp_str)

# %% [markdown]
# ### Visualizing the Results

# %%
plot_3d_prf(x_test, y_test, y_pred, rmse, 'Elastic-Net, (Energy Ratings')

# %% [markdown]
# ## Extreme Gradient Boosting

# %%
x = energy_ratings.values
y = preference_ratings.values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=N_NUMBER)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %%
xgbr_model = XGBRegressor(
    booster='gbtree', objective='reg:squarederror',
    seed=N_NUMBER, missing=0, n_estimators=200, max_depth=5)

xgbr_model = xgbr_model.fit(x_train, y_train, early_stopping_rounds=5,
                            eval_metric='rmse', eval_set=[(x_test, y_test)], verbose=True)

score = xgbr_model.score

y_hat = xgbr_model.predict(x_test)

# %%
k_fold = KFold(n_splits=5, shuffle=True, random_state=N_NUMBER)

cv_rmse = np.mean(cross_val_score(xgbr_model, x, y, cv=k_fold,
                  scoring='neg_root_mean_squared_error'))*-1
print('CV-RMSE:', cv_rmse)

# %% [markdown]
# ### Visualizing the Results

# %%
importances = xgbr_model.feature_importances_

plot_importances(
    importances, x.shape[1], 'Feature importances of Energy Ratings on predicted preferences')

# %%
plot_3d_prf(x_test, y_test, y_hat, rmse,
            'XGBoost Regression with Energy Ratings')

# %% [markdown]
# # 6. Build a regression model to predict art preference ratings from energy ratings and demographic information. Make sure to use cross-validation methods to avoid overfitting and comment on how well your model predicts relative to the “energy ratings only” model.

# %% [markdown]
# ## Data Loading & Cleaning

# %%
# X -> energy rating of art piece
# Y -> preference rating of art piece

all_columns = user_ratings[user_ratings.columns[0:219]]
# Drop columns between 182 and 215
all_columns = all_columns.drop(all_columns.columns[182:215], axis=1).dropna()


# Get X1 (energy rating of each art piece for all users)
energy_ratings = all_columns[all_columns.columns[91:182]]
#energy_ratings.index -= 91

# Get X2 demographic data
demographic_data = all_columns[all_columns.columns[182:]]


# Get Y (preference rating of each art piece for all users)
preference_ratings = all_columns[all_columns.columns[0:91]]

x = np.concatenate((energy_ratings, demographic_data), axis=1)
y = preference_ratings.values

# Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=N_NUMBER)


x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# ## Multi-Regression

# %%
es_net = ElasticNet(alpha=0.06, random_state=N_NUMBER)
es_net.fit(x_train, y_train)

y_pred = es_net.predict(x_test)

slope = es_net.coef_  # B1 (slope)
intercept = es_net.intercept_  # B0 (intercept)

# Print Summary Statistics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', rmse)

# %%
tmp_str = 'predicted preference = '

for i in range(0, len(slope) - 4):
    tmp_str += str(slope[i]) + ' * x' + str(i) + ' + '

tmp_h = {
    len(slope)-4: 'age',
    len(slope)-3: 'gender',
    len(slope)-2: 'political-orientation',
    len(slope)-1: 'art-education'
}

for i in range(len(slope) - 4, len(slope)):
    tmp_str += str(slope[i]) + ' * ' + tmp_h[i] + ' + '

tmp_str += str(intercept)
# print(tmp_str)

# %% [markdown]
# ### Visualizing the Results

# %%
plot_3d_prf(x_test, y_test, y_pred, rmse,
            'Elastic-Net Regression with Energy Ratings and Demographic Data', num_range=84)

# %% [markdown]
# ## Extreme Gradient Boosting Regression

# %%
x = np.concatenate((energy_ratings, demographic_data), axis=1)
y = preference_ratings .values

# Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=N_NUMBER)


x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %%
xgbr_model = XGBRegressor(
    booster='gbtree', objective='reg:squarederror',
    seed=N_NUMBER, missing=0, n_estimators=200, max_depth=5)


xgbr_model = xgbr_model.fit(x_train, y_train, early_stopping_rounds=5,
                            eval_metric='rmse', eval_set=[(x_test, y_test)], verbose=True)


score = xgbr_model.score

y_hat = xgbr_model.predict(x_test)

# %%
k_fold = KFold(n_splits=5, shuffle=True, random_state=N_NUMBER)

cv_rmse = np.mean(cross_val_score(xgbr_model, x, y, cv=k_fold,
                  scoring='neg_root_mean_squared_error'))*-1
print('CV-RMSE:', cv_rmse)

# %% [markdown]
# ### Visualize the results

# %%
importances = xgbr_model.feature_importances_

plot_importances(
    importances, x.shape[1], 'Feature importances of Energy Ratings and Demographic info. on predicted preferences')

# %%
plot_3d_prf(x_test, y_test, y_hat, rmse,
            'XGBoost Regression with Energy Ratings and Demographic Data', num_range=84)

# %% [markdown]
# # 7. Considering the 2D space of average preference ratings vs. average energy rating (that contains the 91 art pieces as elements), how many clusters can you – algorithmically - identify in this space? Make sure to comment on the identity of the clusters – do they correspond to particular types of art?

# %% [markdown]
# ## Data Preprocessing & Visualization

# %%
both_ratings = user_ratings[user_ratings.columns[0:182]].dropna()

# X1 (energy rating of each art piece for all users)
energy_ratings = both_ratings[both_ratings.columns[91:182]]
average_energy_ratings = energy_ratings.mean(axis=0)
average_energy_ratings.index -= 91

# Y (preference rating of each art piece for all users)
preference_ratings = both_ratings[both_ratings.columns[0:91]]
average_preference_ratings = preference_ratings.mean(axis=0)

# %%
# plot all the data inlcuding the outliers
plt.plot(average_preference_ratings, average_energy_ratings,
         color='r', marker='o', markersize=1, linestyle='None')
plt.xlabel('Average Preference Rating')
plt.xlim(0, 6)
plt.ylabel('Average Energy Rating Rating')
plt.ylim(0, 6)
plt.title(
    'Average Preference Rating vs. Average Preference Rating\nfor All Art Pieces')
plt.show()

# %%
# Remove painting 46 (outlier)
average_energy_ratings = average_energy_ratings.drop(45).T
average_preference_ratings = average_preference_ratings.drop(45).T

data = np.column_stack((average_preference_ratings, average_energy_ratings))

# Format data:
x = np.column_stack((data[:, 0], data[:, 1]))

# %% [markdown]
# ## Attempt with DBSCAN

# %%

# Fit model to our data:
dbscanModel = DBSCAN(min_samples=3, eps=0.18).fit(
    x)  # Default eps = 0.5, min_samples = 5

# # Get our labels for each data point:
labels = dbscanModel.labels_

# # Plot the color-coded data:
numClusters = len(set(labels)) - (1 if -1 in labels else 0)
for ii in range(numClusters):
    labelIndex = np.argwhere(labels == ii)
    plt.plot(data[labelIndex, 0], data[labelIndex, 1], 'o',
             markersize=2, label='Cluster {}'.format(ii+1))
    # plot the cluster number
plt.xlabel('Average Preference Ratings')
plt.ylabel('Average Energy Ratings')


plt.title('DBSCAN Clustering of Art Pieces \nAverage Energy and Preference Ratings\n(eps=0.16, min-samples=3)')
plt.legend()
plt.figure(figsize=(10, 10))
plt.show()

# # Looks like there are 4 clusters

# %%
# print art piece number for each cluster
for i in range(0, numClusters):
    print('Cluster {}:'.format(i+1))
    print(np.argwhere(labels == i) + 1)
    print('')

# %% [markdown]
# ## Attempt with K-Means

# %%
# Check with KMeans
# Use the silhouette coefficient to determine the best number of clusters

# Init:
numClusters = 9  # how many clusters are we looping over? (from 2 to 10)
sSum = np.empty([numClusters, 1])*np.NaN  # init container to store sums

# Compute kMeans for each k:
for i in range(2, numClusters+2):  # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters=int(i), random_state=N_NUMBER).fit(
        x)  # compute kmeans using scikit
    cId = kMeans.labels_  # vector of cluster IDs that the row belongs to
    # coordinate location for center of each cluster
    cCoords = kMeans.cluster_centers_
    # compute the mean silhouette coefficient of all samples
    s = silhouette_samples(x, cId)
    sSum[i-2] = sum(s)  # take the sum
    # Plot data:
    plt.subplot(3, 3, i-1)
    plt.hist(s, bins=20)
    plt.xlim(-0.2, 1)
    plt.ylim(0, 25)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}, k={}'.format(int(sSum[i-2]), i))
    plt.tight_layout()  # adjusts subplot
plt.show()

# %%

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2, 10, 9), sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.title('Sum of silhouette scores as a function of number of clusters')
plt.show()

# %%
# kMeans:
numClusters = 4
kMeans = KMeans(n_clusters=numClusters, random_state=N_NUMBER).fit(x)
cId = kMeans.labels_
cCoords = kMeans.cluster_centers_


# Plot the color-coded data:
for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(cCoords[int(ii-1), 0], cCoords[int(ii-1), 1],
             'o', markersize=5, color='black')
    plt.plot(x[plotIndex, 0], x[plotIndex, 1], 'o',
             markersize=1, label='Cluster {}'.format(ii+1))


plt.xlabel('Average Preference Ratings')
plt.ylabel('Average Energy Ratings')


plt.title('KMeans Clustering of Art Pieces Average Rating')
plt.legend()
plt.show()

# %%
for i in range(0, numClusters):
    print('Cluster {}:'.format(i+1))
    print(np.argwhere(cId == i) + 1)
    print('')

# %% [markdown]
# # 8. Considering only the first principal component of the self-image ratings as inputs to a regression model – how well can you predict art preference ratings from that factor alone?

# %% [markdown]
# ## Data Loading & Cleaning

# %%
# Self-image/self-esteem are columns 206-215
# Higher number higher esteem

# Combine prefrence and esteem ratings into one frame, drop na, then split
overall_df = pd.concat([user_ratings[user_ratings.columns[0:91]],
                       user_ratings[user_ratings.columns[205:215]]], axis=1).dropna()

preference_ratings = overall_df[overall_df.columns[0:91]]

esteem_responses = overall_df[overall_df.columns[91:]]

# %% [markdown]
# ## PCA

# %%
plt.figure(figsize=(50, 50))
ax = plt.gca()
plt.imshow(esteem_responses.T)
plt.ylabel('Self-Esteem Questions')
plt.xlabel('User')
plt.title('Self-Esteem Responses for Each User')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cax=cax)
plt.show()

# %%
labels = [
    "On the whole, I am satisfied with myself",
    "At times I think I am no good at all",
    "I feel that I have a number of good qualities",
    "I am able to do things as well as most other people",
    "I feel I do not have much to be proud of",
    "I certainly feel useless at times",
    "I feel that I'm a person of worth, at least on an equal plane with others",
    "I wish I could have more respect for myself",
    "All in all, I am inclined to feel that I am a failure",
    "I take a positive attitude toward myself"
]

corrMatrix = np.corrcoef(esteem_responses, rowvar=False)
plt.figure(figsize=(10, 10))
plt.imshow(corrMatrix)
plt.xlabel('Self-Esteem Questions')
plt.ylabel('Self-Esteem Questions')
plt.title('Self-Esteem Questions Correlation Matrix', )

# plot the question on each tick using the hashmap
plt.xticks(np.arange(10), labels, rotation=45, ha='right')
plt.yticks(np.arange(10), labels, rotation=0)
plt.colorbar()
plt.tight_layout()
plt.show()

# %%
zscoredData = stats.zscore(esteem_responses)
pca = PCA().fit(zscoredData)

eigVals = pca.explained_variance_

loadings = pca.components_

rotatedData = pca.fit_transform(zscoredData)

varExplained = eigVals/sum(eigVals)*100
numData = len(eigVals)
x = np.linspace(1, numData, numData)

plt.bar(x, eigVals, color='gray')
plt.plot([0, numData], [1, 1], color='orange')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Self-Esteem Principal Components')
plt.show()

# %%
print('The 1st principal component explains {:.2f}% of the variance'.format(
    varExplained[0]))

# %%
# Get the first principal component


# + 1. On the whole, I am satisfied with myself
# - 2. At times I think I am no good at all
# + 3. I feel that I have a number of good qualities
# + 4. I am able to do things as well as most other people
# - 5. I feel I do not have much to be proud of
# - 6. I certainly feel useless at times
# + 7. I feel that I'm a person of worth, at least on an equal plane with others
# - 8. I wish I could have more respect for myself
# - 9. All in all, I am inclined to feel that I am a failure
# + 10. I take a positive attitude toward myself


# Print plot for each principal component
for i in range(1):
    plt.figure(figsize=(7, 5))
    plt.bar(x, loadings[i, :]*-1)
    plt.xlabel('Self-Esteem Question')
    plt.xticks(np.arange(1, 11), labels, rotation=45, ha='right')
    plt.ylabel('Loading')
    plt.title('Principal Component ' + str(i+1))

    plt.show()

# PC1: Accounts for pretty much everything, so it will probably represent the overall self-esteem rating
# PC2: Accounts for the difference in positive and negative self-esteem questions


# %%
# Visualize the data in the first principal component
plt.figure(figsize=(10, 5))
plt.bar(np.linspace(1, len(rotatedData), len(
    rotatedData)), rotatedData[:, 0]*-1)
plt.xlabel('User')
plt.ylabel('Overall self-satisfaction rating')
plt.title('Self-Esteem Rating by User under 1st Principal Component')
plt.show()

# %% [markdown]
# ## Regression

# %%
x = ((rotatedData[:, 0]).reshape(-1, 1)*-1)
y = preference_ratings.values


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=N_NUMBER)


x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# ### Linear Regression

# %%
model = LinearRegression().fit(x_train, y_train)

slope = model.coef_  # B1 (slope)
intercept = model.intercept_  # B0 (intercept)

# Predict
y_pred = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', rmse)

print('R2: ', r2_score(y_test, y_pred))

# %%
tmp_str = 'predicted preference = '

for i in range(0, len(slope)):
    tmp_str += str(slope[i]) + ' * x' + str(i) + ' + '

tmp_str += str(intercept)

# print(tmp_str)

# %% [markdown]
# # 9. Consider the first 3 principal components of the “dark personality” traits – use these as inputs to a regression model to predict art preference ratings. Which of these components significantly predict art preference ratings? Comment on the likely identity of these factors (e.g. narcissism, manipulativeness, callousness, etc.).

# %% [markdown]
# ## Data Loading & Cleaning

# %%
# Self-image/self-esteem are columns 206-215 (205, 215)
# Dark personality are columns 183-194 (182, 195)

# Combine prefrence and esteem ratings into one frame, drop na, then split

# Combine user_ratings.columns[0:91] and user_ratings.columns[205:215]
overall_df = pd.concat([user_ratings[user_ratings.columns[0:91]],
                       user_ratings[user_ratings.columns[182:194]]], axis=1).dropna()

preference_ratings = overall_df[overall_df.columns[0:91]]

dark_responses = overall_df[overall_df.columns[91:]]

# Rated 1-5
labels = ['I tend to manipulate others to get my way',
          'I have used deceit or lied to get my way',
          'I have used flattery to get my way',
          'I tend to exploit others towards my own end',
          'I tend to lack remorse',
          'I tend to be unconcerned with the morality of my actions',
          'I can be callous or insensitive',
          'I tend to be cynical',
          'I tend to want others to admire me',
          'I tend to want others to pay attention to me',
          'I tend to seek prestige and status',
          'I tend to expect favors from others']


# %% [markdown]
# ## PCA

# %%
plt.figure(figsize=(50, 50))
ax = plt.gca()
im = ax.imshow(dark_responses.T)
plt.ylabel('Questions')
plt.xlabel('User')
plt.title('Dark Personality Responses by User')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()


# %%
corrMatrix = np.corrcoef(dark_responses, rowvar=False)
plt.figure(figsize=(6, 6))
plt.imshow(corrMatrix)
plt.xlabel('Dark-Response Questions')
plt.xticks(np.arange(12), labels, rotation=45, ha='right')
plt.ylabel('Dark-Response Questions')
plt.yticks(np.arange(12), labels, rotation=0)
plt.title('Correlation Matrix of Dark-Response Questions')
plt.colorbar()
plt.show()

# %%
pca = PCA().fit(scale(dark_responses))

eigVals = pca.explained_variance_
loadings = pca.components_

rotatedData = pca.fit_transform(scale(dark_responses))

varExplained = eigVals/sum(eigVals)*100

numData = dark_responses.shape[1]

x = np.linspace(1, numData, numData)

plt.bar(x, eigVals, color='gray')
plt.plot([0, numData], [1, 1], color='orange')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Dark Personality Principal Components')
plt.show()

# %%
print('The 1st principal component explains {:.2f}% of the variance'.format(
    varExplained[0]))

print('The 2nd principal component explains {:.2f}% of the variance'.format(
    varExplained[1]))
print('The 3rd principal component explains {:.2f}% of the variance'.format(
    varExplained[2]))

# %%
# Print plot for each principal component
fix, ax = plt.subplots(1, 3, figsize=(20, 5))

for i in range(3):
    ax[i].bar(x, loadings[i, :])
    ax[i].set_xlabel('Dark Personality Question')
    ax[i].set_xticks(np.arange(1, 13))
    ax[i].set_xticklabels(labels)
    plt.setp(ax[i].get_xticklabels(), rotation=45,
             ha='right', rotation_mode='anchor')
    ax[i].set_ylabel('Loading')
    ax[i].set_title('Principal Component ' + str(i+1))
plt.show()

# %%
# PC1:  Narcissism: having an inflated sense of self-importance

# PC2: Overall Insensitivity/Callousness: not understanding or caring about the feelings of others
# + 1. I tend to manipulate others to get my way
# - 2. I have used deceit or lied to get my way
# -- 3. I have used flattery to get my way
# ++ 4. I tend to exploit others towards my own end
# ++ 5. I tend to lack remorse
# ++ 6. I tend to be unconcerned with the morality of my actions
# ++ 7. I can be callous or insensitive
# + 8. I tend to be cynical
# --- 9. I tend to want others to admire me
# --- 10. I tend to want others to pay attention to me
# -- 11. I tend to seek prestige and status
# - 12. I tend to expect favors from others

# PC3: Opportunism/Manipulativeness: being deceitful and manipulative
# ++ 1. I tend to manipulate others to get my way
# - 2. I have used deceit or lied to get my way
# + 3. I have used flattery to get my way
# ++ 4. I tend to exploit others towards my own end
# - 5. I tend to lack remorse
# + 6. I tend to be unconcerned with the morality of my actions
# -- 7. I can be callous or insensitive
# --- 8. I tend to be cynical
# -- 9. I tend to want others to admire me
# - 10. I tend to want others to pay attention to me
# + 11. I tend to seek prestige and status
# ++ 12. I tend to expect favors from others


# %%
# 3d visualization of dark personality
# x -> narcissism
# y -> callousness
# z -> manipulativeness

overall_title = 'Dark Personality under $1^{st}$ 3 Principal Components'
views = [(13, 7, 'Front View'),
         (60, 0, 'Top View'), (0, 60, 'Side View')]

fig = plt.figure(figsize=(20, 6))

for i, (elev, azim, title) in enumerate(views):
    ax = fig.add_subplot(1, 3, i+1, projection='3d', elev=elev, azim=azim)
    ax.scatter(rotatedData[:, 0], rotatedData[:, 1],
               rotatedData[:, 2], c='blue', marker='o')
    ax.set_xlabel('Narcissism')
    ax.set_ylabel('Callousness')
    ax.set_zlabel('Manipulativeness')
    plt.title(title)

plt.suptitle(overall_title)
plt.show()

# %% [markdown]
# ## Regression

# %%
# Using 1st three principal components to predict preference ratings

narcissism = rotatedData[:, 0]
callousness = rotatedData[:, 1]
manipulativeness = rotatedData[:, 2]

print(narcissism.shape, callousness.shape, manipulativeness.shape)

# %%
x = np.column_stack((narcissism, callousness, manipulativeness))
y = preference_ratings.values

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=N_NUMBER)


x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# ## Elastic-Net Regularized Regression

# %%
es_model = ElasticNet(alpha=0.01, normalize=True)

es_model.fit(x_train, y_train)

# Predict
slope = es_model.coef_  # B1 (slope)
intercept = es_model.intercept_  # B0 (intercept)


y_pred = es_model.predict(x_test)
# Print Summary Statistics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', rmse)


tmp_str = 'predicted preference = '
# show equation
for i in range(0, len(slope)):
    tmp_str += str(slope[i]) + ' * x' + str(i) + ' + '
tmp_str += str(intercept)
# print(tmp_str)


# %%
def plot_3d_scatter_with_residuals(x_test, y_test, y_pred, rmse, x_columns, x_label, y_label):
    fig = plt.figure(figsize=(20, 5))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.scatter(x_test[:, x_columns[i][0]], x_test[:, x_columns[i]
                   [1]], y_test[:, 0], color='gray', label='Actual')
        ax.scatter(x_test[:, x_columns[i][0]], x_test[:, x_columns[i]
                   [1]], y_pred[:, 0], color='red', label='Predicted')

        for j in range(0, len(y_test)):
            ax.plot([x_test[j, x_columns[i][0]], x_test[j, x_columns[i][0]]], [x_test[j, x_columns[i][1]],
                    x_test[j, x_columns[i][1]]], [y_test[j, 0], y_pred[j, 0]], color='green', alpha=0.5)
        ax.set_xlabel(x_label[i])
        ax.set_ylabel(y_label[i])
        ax.set_zlabel('Preference Rating')

        ax.set_title(
            'Actual vs. Predicted Preference Ratings\nRMSE = {:.3f}'.format(rmse))
        ax.legend()

    plt.show()


plot_3d_scatter_with_residuals(x_test, y_test, y_pred, rmse, [[0, 1], [0, 2], [1, 2]], [
                               'Narcissism', 'Narcissism', 'Callousness'], ['Callousness', 'Manipulativeness', 'Manipulativeness'])


# %% [markdown]
# #### Use ANOVA to test significance
#
# Could we use  Benjamini-Hochberg procedure to correct for multiple testing?

# %% [markdown]
# #### 1st PCA component regression

# %%
x = narcissism.reshape(-1, 1)
y = preference_ratings.values

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=N_NUMBER)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %%
es_model = ElasticNet(alpha=0.01, normalize=True)

es_model.fit(x_train, y_train)

# Predict
slope = es_model.coef_  # B1 (slope)
intercept = es_model.intercept_  # B0 (intercept)


y_pred = es_model.predict(x_test)
# Print Summary Statistics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', rmse)


tmp_str = 'predicted preference = '
# show equation
for i in range(0, len(slope)):
    tmp_str += str(slope[i]) + ' * x' + str(i) + ' + '
tmp_str += str(intercept)
# print(tmp_str)
print('r2_score: ', r2_score(y_test, y_pred))
print('explained_variance_score: ', explained_variance_score(y_test, y_pred))

# %% [markdown]
# #### 2nd PCA component regression

# %%
x = callousness.reshape(-1, 1)
y = preference_ratings.values

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=N_NUMBER)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %%
es_model = ElasticNet(alpha=0.01, normalize=True)

es_model.fit(x_train, y_train)

# Predict
slope = es_model.coef
intercept = es_model.intercept_


y_pred = es_model.predict(x_test)
# Print Summary Statistics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', rmse)


tmp_str = 'predicted preference = '
# show equation
for i in range(0, len(slope)):
    tmp_str += str(slope[i]) + ' * x' + str(i) + ' + '
tmp_str += str(intercept)
# print(tmp_str)
print('r2_score: ', r2_score(y_test, y_pred))
print('explained_variance_score: ', explained_variance_score(y_test, y_pred))

# %% [markdown]
# #### 3rd PCA component regression

# %%
x = manipulativeness.reshape(-1, 1)
y = preference_ratings.values

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=N_NUMBER)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %%
es_model = ElasticNet(alpha=0.0012, normalize=True)
es_model.fit(x_train, y_train)

# Predict
slope = es_model.coef_
intercept = es_model.intercept_


y_pred = es_model.predict(x_test)
# Print Summary Statistics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', rmse)


tmp_str = 'predicted preference = '
# show equation
for i in range(0, len(slope)):
    tmp_str += str(slope[i]) + ' * x' + str(i) + ' + '
tmp_str += str(intercept)
# print(tmp_str)
print('r2_score: ', r2_score(y_test, y_pred))
print('explained_variance_score: ', explained_variance_score(y_test, y_pred))

# %% [markdown]
# # 10. Can you determine the political orientation of the users (to simplify things and avoid gross class imbalance issues, you can consider just 2 classes: “left” (progressive & liberal) vs. “non-left” (everyone else)) from all the other information available, using any classification model of your choice? Make sure to comment on the classification quality of this model.

# %% [markdown]
# ## Data Preprocessing

# %%
politics_index = user_ratings.columns.get_loc(217)
combined = user_ratings.dropna()


# ground truth
def left_or_right(x):
    if x == 0 or x == 1:
        return 0
    else:
        return 1


y = combined[217].apply(left_or_right).values.flatten()

x = combined.drop(217, axis=1)


print(x.shape, y.shape)

# %% [markdown]
# ## PCA

# %%
plt.figure(figsize=(50, 50))
ax = plt.gca()
im = ax.imshow(x.T)
plt.ylabel('Responses')
plt.xlabel('User')
plt.title('Dark Personality Responses by User')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()


# %%
corrMatrix = np.corrcoef(x, rowvar=False)
plt.figure(figsize=(6, 6))
plt.imshow(corrMatrix)
plt.xlabel('Responses')

plt.ylabel('Responses')
plt.title('Correlation Matrix of User Responses')
plt.colorbar()
plt.show()

# %%
# Run PCA on everything except political orientation
zscoredData = stats.zscore(x)

pca = PCA().fit(zscoredData)

eigVals = pca.explained_variance_

loadings = pca.components_

rotatedData = pca.fit_transform(zscoredData)

varExplained = eigVals/sum(eigVals)*100

numData = len(eigVals)

x_ = np.linspace(1, numData, numData)
num_components = np.count_nonzero(eigVals > 1)

plt.bar(x_, eigVals, color='gray')
plt.plot([0, numData], [1, 1], color='orange')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Scree plot of Eigenvalues for User Responses')
plt.show()

for i in range(num_components):
    print('Principal Component ' + str(i+1) +
          ': ' + str(varExplained[i]) + '%')


# %%

# Select the number of components to keep using the Kaiser criterion

# Create a subplot grid with the appropriate number of rows and columns
num_rows = 9 // 3
num_cols = 3
fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 5*num_rows))

# Iterate over the selected components and plot the loadings in each subplot
for i, ax in enumerate(axs.flat):
    if i < 9:
        ax.bar(x_, loadings[i, :]*-1)
        #ax.plot(x, loadings[i,:], color='gray')
        ax.set_title(
            f'Principal Component {i+1}\nVariance Explained: {varExplained[i]:.2f}%', fontsize=20)

# Adjust the spacing between subplots
plt.tight_layout()

# %%
# print comulative variance accounted for by kaiser criterion
print('Cumulative Variance Explained by Kaiser criterion: ' +
      str(sum(varExplained[:num_components])) + '%')

# %% [markdown]
# ## Classification

# %%
# x is loadings up to the number of components
x_train, x_test, y_train, y_test = train_test_split(
    x.values, y.reshape(-1, 1), test_size=0.3, random_state=N_NUMBER)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %%
combined[217].apply(left_or_right).value_counts()

# %% [markdown]
# ## Gradient Boosting Classification

# %%
clf = GradientBoostingClassifier(
    n_estimators=1000, learning_rate=0.01, max_depth=10, random_state=N_NUMBER)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=N_NUMBER)
for train_index, test_index in skf.split(x, y):
    x_train_, x_test_ = x.values[train_index], x.values[test_index]
    y_train_, y_test_ = y[train_index], y[test_index]
    clf.fit(x_train_, y_train_)
    y_pred_ = clf.predict(x_test_)
    print('Accuracy: ', accuracy_score(y_test_, y_pred_))


# %%
y_pred = clf.predict(x_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F1: ', f1_score(y_test, y_pred))
print('r2_score: ', r2_score(y_test, y_pred))

plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Gradient Boosting Classifier')

# %%
y_pred_prb = clf.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prb)
# Plot the ROC curve
plt.plot(fpr, tpr, 'b-', label='Gradient Boosting')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve, AUC = {:.4f}'.format(roc_auc_score(y_test, y_pred_prb)))
plt.legend(loc='best')
plt.show()


# %% [markdown]
# ### Visualization?

# %% [markdown]
# Ideas for visualization:
# -Voronoi diagram
# -plot the decision boundary
# - use t-distributed stochastic neighbor embedding (t-SNE) to visualize the data in 2D

# %% [markdown]
# # __Extra credit:__ Tell us something interesting about this dataset that is not trivial and not already part of an answer (implied or explicitly) to these enumerated questions.

# %%
# Based on K-Means, how many clusters can be identified in the first 3 principal components of the "dark personality traits"?

# %%
overall_df = pd.concat([user_ratings[user_ratings.columns[0:91]],
                       user_ratings[user_ratings.columns[182:194]]], axis=1).dropna()

preference_ratings = overall_df[overall_df.columns[0:91]]

dark_responses = overall_df[overall_df.columns[91:]]
pca = PCA().fit(scale(dark_responses))

eigVals = pca.explained_variance_
loadings = pca.components_

rotatedData = pca.fit_transform(scale(dark_responses))

varExplained = eigVals/sum(eigVals)*100

numData = dark_responses.shape[1]

x, y, z = rotatedData[:, 0], rotatedData[:, 1], rotatedData[:, 2]

# %%
pts = np.column_stack((x, y, z)).reshape(-1, 3)

# %%
num_clusters = 7
kMeans = KMeans(n_clusters=num_clusters).fit(np.column_stack(
    (rotatedData[:, 0], rotatedData[:, 1], rotatedData[:, 2])))


# %%
# K-Means using silhouette score
# Check with KMeans
# Use the silhouette coefficient to determine the best number of clusters

# Init:
numClusters = 9  # how many clusters are we looping over? (from 2 to 10)
sSum = np.empty([numClusters, 1])*np.NaN  # init container to store sums

# Compute kMeans for each k:
for i in range(2, numClusters+2):  # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters=int(i), random_state=N_NUMBER).fit(pts)
    cId = kMeans.labels_  # vector of cluster IDs that the row belongs to
    # coordinate location for center of each cluster
    cCoords = kMeans.cluster_centers_
    # compute the mean silhouette coefficient of all samples
    s = silhouette_samples(pts, cId)
    sSum[i-2] = sum(s)  # take the sum
    # Plot data:
    plt.subplot(3, 3, i-1)
    plt.hist(s, bins=20)
    plt.xlim(-0.2, 1)
    plt.ylim(0, 25)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}, k={}'.format(int(sSum[i-2]), i))
    plt.tight_layout()  # adjusts subplot
plt.show()


# %%

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2, 10, 9), sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.title('Sum of silhouette scores as a function of number of clusters')
plt.show()


# %%
num_clusters = 2
kMeans = KMeans(n_clusters=num_clusters, random_state=N_NUMBER).fit(scale(pts))
cId = kMeans.labels_
cCoords = kMeans.cluster_centers_

fig = plt.figure(figsize=(20, 6))
#ax = Axes3D(fig)
#fig.add_axes(ax, projection='3d')

views = [(13, 7, 'Front View'),
         (60, 0, 'Top View'), (0, 60, 'Side View')]
# Plot the color-coded data:

for j, (elev, azim, title) in enumerate(views):
    ax = fig.add_subplot(1, 3, j+1, projection='3d', elev=elev, azim=azim)
    ax.set_xlabel('PC1 - Narcissism')
    ax.set_ylabel('PC2 - Callousness')
    ax.set_zlabel('PC3 - Manipulativeness')
    plt.title(title)
    for i in range(num_clusters):
        plotIndex = np.argwhere(cId == i)
        sc = ax.scatter(x[plotIndex],
                        y[plotIndex],
                        z[plotIndex],
                        label=f'Cluster {i+1}',
                        s=30, marker='o', alpha=0.5)
        sc = ax.scatter(cCoords[int(i-1), 0],
                        cCoords[int(i-1), 1],
                        cCoords[int(i-1), 2],
                        color='black', s=100)
    ax.legend(loc='best')

plt.suptitle('K-Means Clustering of Dark Personality Traits')
