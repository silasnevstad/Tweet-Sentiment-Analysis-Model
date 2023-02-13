# Tweet-Sentiment-Analysis-Model
A machine learning model for sentiment analysis built in python, trained and tested on a dataset of Twitter tweets (1.6 million). The model uses NLTK to preprocess the text data and Scikit-learn to perform hyperparameter tuning using GridSearchCV. The best hyperparameters are then used to train a Naive Bayes classifier, which is used to perform sentiment analysis on new tweets.

<!-- TABLE OF CONTENTS -->
<div id="top"></div>
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#builtwith">Built With</a></li>
    <li><a href="#dependencies">Dependencies</a></li>
  </ol>
</details>

<!-- Overview -->
<div id="overview"></div>

## Overview
This code provides a machine-learning model for sentiment analysis on Twitter data. It first preprocesses the data by tokenizing, removing stopwords, stemming, and lemmatizing the text. Then, the code trains a Naive Bayes classifier on the preprocessed data using the best hyperparameters selected by grid search or random search. The model's performance is evaluated on a validation and test set, and the accuracy, precision, recall, and F1 scores are calculated for each class. Finally, the trained model is used to perform sentiment analysis on new tweets.

The model uses the NLTK library for text preprocessing and the scikit-learn library for the grid search, which allows for tuning hyperparameters to optimize the model's performance. Random search is also an option for tuning hyperparameters, which may be faster than grid search. The idea I had behind this sentiment analysis model was to provide a powerful tool for understanding public opinion on social media, which can be helpful for businesses, governments, and researchers.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Built With -->
<div id="builtwith"></div>

## Built With
* [Python](https://python.org)
* [Data](https://www.kaggle.com/datasets/kazanova/sentiment140)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Dependencies -->
<div id="dependencies"></div>

## Dependencies
- csv
- string
- nltk: Natural Language Toolkit for NLP tasks. Install with pip install nltk.
- random
- os 
- pickle
- time
- nltk.corpus (Pre-processed textual corpora for NLP tasks in NLTK)
- nltk.tokenize (Tokenizers for breaking text into words and sentences in NLTK)
- nltk.stem (Implementations of word stemming algorithms in NLTK)
- nltk.sentiment.vader (Pre-trained sentiment analysis model in NLTK)
- sklearn.model_selection
- sklearn.pipeline
- sklearn.naive_bayes
- sklearn.feature_extraction.text
- sklearn.metrics
