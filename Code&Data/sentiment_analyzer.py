from textblob import TextBlob, Word, Blobber
import preprocessor as p
from scipy import stats
import pandas as pd
import numpy as np
import tweepy
import nltk
import csv
import re
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

#--------------------Reading in From Data CSV-----------------------------------

stoplist = stopwords.words('english')
stoplist.append("RT") #Retweet doesn't effect our experiment
stoplist.append("rt")

f = pd.read_csv('ThankGod.csv', lineterminator='\n')

games = f['Tweet'].astype(str)
y = f['Outcome']

won_games = f[(f['Outcome']==1)]
won_games = won_games['Tweet'].astype(str)

lost_games = f[(f['Outcome']==0)]
lost_games = lost_games['Tweet'].astype(str)

#-------------------------------Functions----------------------------------------

#Pre-Processing Text: Removed punctuation and hyperlinks
def cleaning_tweets(won_lost_game_list):
    cleaned_won_lost = []
    for tweet in won_lost_game_list:
        tweet = re.sub(r'http\S+', '', tweet)
        tweet = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", tweet)
        cleaned_won_lost.append(re.sub("([^0-9A-Za-z \t])|(\w+:@\/\/\S+)", " ", tweet))
    return cleaned_won_lost

#Word Tokenizing
def tokenizing_tweets(cleaned_won_lost):
    won_lost_tokens = []
    for tweet in cleaned_won_lost:
        won_lost_tokens.append(nltk.word_tokenize(tweet))
    return won_lost_tokens

#Remove StopWords
def remove_stop_words(won_lost_tokens, stoplist):
    won_lost_filtered_tokens = []
    for list in won_lost_tokens:
        temp = []
        for w in list:
            if w not in stoplist:
                temp.append(w)
        won_lost_filtered_tokens.append(temp)
    return won_lost_filtered_tokens

#Find frequency distributions for words in tweets
def frequency_distribution(won_lost_filtered_tokens):
    total_won_lost_words = []
    for list in won_lost_filtered_tokens:
        for tweet in list:
            total_won_lost_words.append(tweet)
    freqlist_won_lost = nltk.FreqDist(total_won_lost_words)
    return freqlist_won_lost

#generate word cloud
def generate_word_cloud(freqlist_won_lost):
    wordcloud = WordCloud(width=800,height=300, max_words=500,relative_scaling=1,normalize_plurals=False,background_color='rgb(236, 237, 238)').generate_from_frequencies(freqlist_won_lost)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

#generate list of sentiments for all tweets
def sentiment_analyzer(cleaned_won_lost):
    wins_losses_polarities = []
    for tweet in cleaned_won_lost:
        analysis = TextBlob(tweet)
        wins_losses_polarities.append(analysis.sentiment.polarity)
    return wins_losses_polarities

#generate average sentiment overall
def average_sentiment_analyzer(wins_losses_polarities):
    average_win_loss_polarity = 0
    win_loss_num_total = len(wins_losses_polarities)
    for game in wins_losses_polarities:
        average_win_loss_polarity += game
    average_win_loss_polarity = average_win_loss_polarity / win_loss_num_total
    return average_win_loss_polarity

def downcase_tweets(games):
    lowercase_all_games = []
    lowercase_all_games = [tweet.lower() for tweet in games]
    return lowercase_all_games

def bagOfwords(filtered_tokens):
    detokenized = []
    for tweet in filtered_tokens:
        detokenized.append(TreebankWordDetokenizer().detokenize(tweet))
    bag_of_words = CountVectorizer()
    text_counts = bag_of_words.fit_transform(detokenized)
    return text_counts

#-------------------------------ANALYZER----------------------------------------
#----------------WON GAMES-------------

cleaned_won = cleaning_tweets(won_games)
won_tokens = tokenizing_tweets(cleaned_won) #only need to do this for word cloud
won_filtered_tokens = remove_stop_words(won_tokens, stoplist) #only need to do this for word cloud
freqlist_won = frequency_distribution(won_filtered_tokens)  #only need to do this for word cloud

wins_polarities = sentiment_analyzer(cleaned_won)
average_win_polarity = average_sentiment_analyzer(wins_polarities)
print("The average polarity of all games won:", average_win_polarity)

#----------------LOST GAMES-----------

cleaned_lost = cleaning_tweets(lost_games)
lost_tokens = tokenizing_tweets(cleaned_lost) #only need to do this for word cloud
lost_filtered_tokens = remove_stop_words(lost_tokens, stoplist) #only need to do this for word cloud
freqlist_lost = frequency_distribution(lost_filtered_tokens) #only need to do this for word cloud

losses_polarities = sentiment_analyzer(cleaned_lost)
average_loss_polarity = average_sentiment_analyzer(losses_polarities)
print("The average polarity of all games won:", average_loss_polarity)

#----------------Word Clouds----------------

"""
generate_word_cloud(freqlist_won)
generate_word_cloud(freqlist_lost)
"""

#----------------T-Test----------------

# compare the polarity of the wins and losses of both teams to see if significant
stat, p = stats.ttest_ind(wins_polarities, losses_polarities)
print("T-Test for Games Won vs. Lost:")
print('t=%.3f, p=%.3f' % (stat, p))

#-------------------------------ClASSIFIER----------------------------------------

lowercase_all_games = downcase_tweets(games)
cleaned = cleaning_tweets(lowercase_all_games)
alltokens = tokenizing_tweets(cleaned)
filtered_tokens = remove_stop_words(alltokens, stoplist)
freqlist_all = frequency_distribution(filtered_tokens)
text_counts = bagOfwords(filtered_tokens)

"""
#plotting frequencies of words for all tweets
freqlist_all.plot(30,cumulative=False)
plt.show()

generate_word_cloud(freqlist_all)
"""

#Splitting test-train sets
X_train, X_test, y_train, y_test = train_test_split(text_counts, y, test_size=0.3, random_state=1)

#Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))
print(confusion_matrix(y_test,predicted))
print(classification_report(y_test,predicted))

#Linear SVM
svm = LinearSVC().fit(X_train, y_train)
predicted = svm.predict(X_test)
print("SVM Accuracy:", metrics.accuracy_score(y_test, predicted))
print(confusion_matrix(y_test,predicted))
print(classification_report(y_test,predicted))
