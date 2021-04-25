## Sentiment-Analysis-Logistic-Regression

* Sentiment Analysis is the Complex Task of Predicting whether a given Review is positive or negative in the cases where words are ambiguous most of the times.
* Sentiment Analysis is an application of Natural Language Processing which is a constituent of Artificial Intelligence.

### Run instantly:
* Install pandas, nltk, pickle
* Download the saved model from : https://rb.gy/hnjcy4
* Download the saved vectorizer from : https://rb.gy/256lnh
* Run it using the app.py file

**Language: Python**

### Package dependencies
*Use conda to install if you are using Anaconda Distribution*

* Scikit-learn (The ML Library): 
    * `pip install scikit-learn` or `conda install scikit-learn` 
    * Use as `import sklearn`
* NLTK - Natural Language Toolkit (For Processing Text data): 
    * `pip install nltk` or `conda install nltk`
    * Use as `import nltk`
* Regex - Regular Expressions: 
    * Installs automatically with Python
    * Use as `import re`
* Pickle - Saving computed model and vectorizer:
    * `pip install pickle` or `conda install pickle`
    * Use as `import pickle`
* Pandas - Dataset manipulation and preprocessing:
    * Install using `pip install pandas` or `conda install pandas`
    * Use as `import pandas as pd`
* Numpy - Numerical computations:
    * `pip install numpy` or `conda install numpy`
    * Use as `import numpy as np`
    

### Workflow

#### Dataset:

* Sentiment Analysis with Mobile review data of Amazon from kaggle : https://www.kaggle.com/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones

* This Dataset contains 5 fields
    * Product Name: The name of the Product. e.g. Sprint EPIC 4G Galaxy SPH-D7
    * Brand Name: Name of the parent company. e.g. Samsung
    * Price: Price of the product. (Max: 2598, Min: 1.73, Mean: 226.86)
    * Rating: Rating of the product ranging between 1-5
    * Reviews: Description of the user experience
    * Review Votes: Number of people voted the review (Min: 0, Max: 645, Mean: 1.50)
    
#### Process Flow

![Workflow Diagram](/images/sa_process_flow.PNG)

#### Specifications:

* The Dataset contains null values.
* We take only the Rating field and the Review field for our purpose

#### Data Preprocessing:

* Rating field: Omitting Neutral Reviews
* Reviews field: Removing Punctuation, stop words; Stemming, Lemmatizing
* Positivity field: (User defined Field) Reviews with rating 4,5 are considered positive; Reviews with rating 1,2 are considered negative.

#### Vectorizer: TF-IDF vectorizer

#### Classification Model: Logistic Regression Classification

![Workflow Diagram](/images/sa_auc_roc.PNG)

#### AUC ROC score: 95%

![Workflow Diagram](/images/sa_classification_accuracy.PNG)

#### Jacard score accuracy: 96.6%
