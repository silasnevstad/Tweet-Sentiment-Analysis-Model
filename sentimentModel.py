# [--- Imports ---]
import csv
import string
import nltk
import random
import os
import pickle
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,  roc_curve, auc
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# [--- RNN Model Building ---]
import numpy as np
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM
# from keras.utils import to_categorical

# Import the OpenAI API client
# import openai

# openai.api_key = 'API-KEY-HERE'

# start time 
start_time = time.time()


# Download the necessary NLTK data (such as stopwords, various tokenizers, etc.)
nltk.download('stopwords') # the, a, an, and, or, but, etc.
nltk.download('punkt') # word tokenizer
nltk.download('vader_lexicon') # exclamation marks, emojis, etc. 
nltk.download('averaged_perceptron_tagger') # part of speech tagger (nouns, verbs, adjectives, etc.)
nltk.download('wordnet') # lemmatizer (converts words to their root form)

# [--- Data Processiong ---]
# Preprocess the data
stop_words = set(stopwords.words('english') + list(string.punctuation))
ps = PorterStemmer()
wnl = nltk.WordNetLemmatizer()

preprocessed_data_filename = 'data/preprocessed_data.pkl'

if os.path.exists(preprocessed_data_filename):
    with open(preprocessed_data_filename, "rb") as f:
        preprocessed_data = pickle.load(f)
    print("Loaded preprocessed data from file.")
else:
    # Load the dataset
    filename = 'data/training_processed.csv'
    dataset = []
    with open(filename, "r", encoding="ISO-8859-1") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            dataset.append(row)

    print("Loaded {} tweets.".format(len(dataset)))
    # shuffle the data
    dataset = random.sample(dataset, len(dataset))
    preprocessed_data = []
    print("Preprocessing data...")
    for i, row in enumerate(dataset):
        tweet = row[5] # the tweet is the 6th column in the dataset
        tweet = ''.join([i for i in tweet if not i.isdigit()]) # remove numbers
        tweet = ' '.join([word for word in tweet.split() if 'http' not in word and not word.startswith('@') and word != 'RT']) # remove URLs, @mentions, and retweets
        tweet = tweet.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        tweet = tweet.encode('ascii', 'ignore').decode('ascii')
        tokens = word_tokenize(tweet.lower()) # convert to lowercase and tokenize
        tokens_filtered = [word for word in tokens if not word in stop_words] # remove stopwords
        # tokens_stemmed = [ps.stem(word) for word in tokens_filtered] # stem the words
        tokens_snowball = [SnowballStemmer('english').stem(word) for word in tokens_filtered] # stem the words
        tokens_pos = nltk.pos_tag(tokens_snowball) # get the part of speech for each word
        tokens_pos_lemmatized = [] # lemmatize the words
        for word, pos in tokens_pos:
            if pos.startswith('NN'):
                pos = 'n' # noun
            elif pos.startswith('VB'):
                pos = 'v' # verb
            elif pos.startswith('JJ'):
                pos = 'a' # adjective
            elif pos.startswith('R'):
                pos = 'r' # adverb
            else:
                pos = None
            if pos:
                tokens_pos_lemmatized.append((wnl.lemmatize(word, pos), pos))
        # print(f"Original tweet: {tweet}")
        # print(f"Preprocessed tweet: {tokens_pos_lemmatized}")
        preprocessed_data.append((tokens_pos_lemmatized, row[0]))
        
        if (i+1) % 8000 == 0:
            print("Preprocessed {} / {} tweets ({}%)".format(i+1, len(dataset), round((i+1)/len(dataset)*100, 2)))
    print("Finished preprocessing data.")

    # Save the preprocessed data to a file
    with open(preprocessed_data_filename, "wb") as f:
        pickle.dump(preprocessed_data, f)
    print("Saved preprocessed data to file.")


def visual_pos_words():
    # get negative words from preprocessed data
    print("Visualizing positive words...")
    pos_data = []
    for i in preprocessed_data:
        if i[1] == '4':
            pos_data.append(i[0])
    # print(pos_data)
    plt.figure(figsize = (10,10))
    string_of_words = ""
    for i in pos_data:
        for j in i:
            string_of_words += j[0] + " "
    wc = WordCloud(background_color="white", max_words=2000, width=1000, height=600).generate(string_of_words)
    plt.imshow(wc) 
    plt.show()

def visual_neg_words():
    print("Visualizing negative words...")
    # get negative words from preprocessed data
    neg_data = []
    for i in preprocessed_data:
        if i[1] == '0':
            neg_data.append(i[0])
    plt.figure(figsize = (10,10))
    string_of_words = ""
    for i in neg_data:
        for j in i:
            string_of_words += j[0] + " "
    wc = WordCloud(background_color="white", max_words=2000, width=1000, height=600).generate(string_of_words)
    plt.imshow(wc)
    plt.show()

# visual_pos_words() # visualizes the positive words from the preprocessed data
# visual_neg_words() # visualizes the negative words from the preprocessed data

def model_Evaluate(model, x_test, y_test):
    import seaborn as sns
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','True Pos','False Pos','False Neg']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    sns.heatmap(cf_matrix, annot = np.asarray([f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]).reshape(2,2), cmap = 'Blues',fmt = '',xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    y_pred = [0 if i == 0 else 1 for i in y_pred]
    plt.show()

# [--- Logistic Regression ---]
def logistic_regression():
    X = [i[0] for i in preprocessed_data] # the tweets
    y = [str(i[1]) for i in preprocessed_data] # the labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% test
    X_words = [[word for word, pos in sentence] for sentence in X_train]
    X_sentences = [" ".join(sentence) for sentence in X_words]
    X_test = [" ".join([word for word, pos in sentence]) for sentence in X_test]
    vectoriser = TfidfVectorizer(ngram_range=(1,2))
    vectoriser.fit(X_sentences)
    print('No. of feature_words: ', len(vectoriser.get_feature_names_out()))

    x_vectorised = vectoriser.transform(X_sentences)
    X_test = vectoriser.transform(X_test)
    
    print("Training Logistic Regression model...")
    logreg = LogisticRegression(n_jobs=1, C=1e5, solver='lbfgs', multi_class='multinomial', max_iter = 100000)
    logreg.fit(x_vectorised, y_train)
    model_Evaluate(logreg, X_test, y_test)
    
# logistic_regression()

# [--- Naive Bayes ---]
# Feature engineering
def get_word_features(words):
    words = [word for word, pos in words]
    word_features = {}
    for word in words:
        word_features[word] = True
    return word_features

def preprocess(tokens):
    return " ".join(tokens)

# Split the data into training and testing sets
train_data = preprocessed_data[:int(len(preprocessed_data)*0.6)]
val_data = preprocessed_data[int(len(preprocessed_data)*0.6):int(len(preprocessed_data)*0.8)]
test_data = preprocessed_data[int(len(preprocessed_data)*0.8):]

# Define the pipeline and grid search parameters
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(preprocessor=preprocess, max_features=10000)),
    ('classifier', MultinomialNB())
])

parameters = {
    'vectorizer__ngram_range': [(1,1), (1,2)],
    'vectorizer__max_df': [0.2, 0.4, 0.6],
    'classifier__alpha': [0.4, 0.5, 0.6]
}

# Fit the grid search to the training data and tune hyperparameters using the validation set
train_features = [get_word_features(tokens) for tokens, label in train_data]
train_labels = [label for tokens, label in train_data]

val_features = [get_word_features(tokens) for tokens, label in val_data]
val_labels = [label for tokens, label in val_data]

scores = cross_val_score(pipeline, train_features, train_labels, cv=10)
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

# Fit the grid search to the training data and tune hyperparameters using the validation set
print("Fitting grid search...")
grid_search = GridSearchCV(pipeline, parameters, cv=10, n_jobs=6, verbose=1)
grid_search.fit(train_features, train_labels)

# Print the best parameters and score
print("Best score: {}".format(grid_search.best_score_))
print("Best parameters: {}".format(grid_search.best_params_))

# Evaluate the model on the test data
test_features = [get_word_features(tokens) for tokens, label in test_data]
test_labels = [label for tokens, label in test_data]

test_score = grid_search.score(test_features, test_labels)
print("Test score: {}".format(test_score))

predicted_labels = grid_search.predict(test_features) # Predict labels for the test data

for label in grid_search.classes_: # Calculate precision, recall, and F1 score for each class
    y_true = [int(x == label) for x in test_labels]
    y_pred = [int(x == label) for x in predicted_labels]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("Label: {}".format(label))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 score: {}".format(f1))

# Train the sentiment analysis model using the best hyperparameters
print("Training model...")
sid = SentimentIntensityAnalyzer()
train_set = [(get_word_features(tokens_pos_lemmatized), sentiment) for tokens_pos_lemmatized, sentiment in train_data]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Finished training model.")

# Evaluate the performance of the model on the validation set
val_set = [(get_word_features(tokens_pos_lemmatized), sentiment) for tokens_pos_lemmatized, sentiment in val_data]
accuracy = nltk.classify.accuracy(classifier, val_set)
print("Accuracy on validation set:", accuracy)

# Evaluate the performance of the model on the test set
test_set = [(get_word_features(tokens_pos_lemmatized), sentiment) for tokens_pos_lemmatized, sentiment in test_data]
accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy on test set:", accuracy)

# Use the model to perform sentiment analysis on new tweets
new_tweet = "I hate python!"

tokens = word_tokenize(new_tweet.lower())
tokens_filtered = [word for word in tokens if not word in stop_words]
tokens_stemmed = [ps.stem(word) for word in tokens_filtered]
tokens_pos = nltk.pos_tag(tokens_filtered)
tokens_pos_lemmatized = []
for word, pos in tokens_pos:
    if pos.startswith('NN'):
        pos = 'n'
    elif pos.startswith('VB'):
        pos = 'v'
    elif pos.startswith('JJ'):
        pos = 'a'
    elif pos.startswith('R'):
        pos = 'r'
    else:
        pos = None
    if pos:
        tokens_pos_lemmatized.append((wnl.lemmatize(word, pos), pos))
        
new_features = get_word_features(tokens_pos_lemmatized)
prediction = classifier.classify(new_features)
print("Prediction for \"{}\": {}".format(new_tweet, prediction))

print("Code completed in {} seconds.".format(round(time.time() - start_time, 2)))
