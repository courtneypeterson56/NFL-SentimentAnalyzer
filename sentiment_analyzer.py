from textblob import TextBlob, Word, Blobber
import preprocessor as p
from scipy import stats
import nltk
import re
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import tweepy
from tweepy import OAuthHandler
from sklearn import metrics
import csv
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

stoplist = stopwords.words('english')

f = pd.read_csv('ThankGod.csv', lineterminator='\n')
#print(f.shape)
#print(f.head())

games = f['Tweet'].astype(str)
y = f['Outcome']

won_games = f[(f['Outcome']==1)]
won_games = won_games['Tweet'].astype(str)
won_games = won_games.head(37647)

lost_games = f[(f['Outcome']==0)]
lost_games = lost_games['Tweet'].astype(str)

# print(len(won_games))
# print(len(lost_games))

#-------------------------------Word Cloud----------------------------------------
#Cleaning Won Tweets
cleaned_won = []
for tweet in won_games:
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", tweet)
    cleaned_won.append(re.sub("([^0-9A-Za-z \t])|(\w+:@\/\/\S+)", " ", tweet))
#print("Won Cleaned:", cleaned_won[0:8])

won_tokens = []
total_won_words = []
won_filtered_tokens = []

#Tokenizing Won Tweets
for tweet in cleaned_won:
    won_tokens.append(nltk.word_tokenize(tweet))

#Take Stop Words Out
for list in won_tokens:
    temp = []
    for w in list:
        if w not in stoplist:
            temp.append(w)
    won_filtered_tokens.append(temp)
#print("Filtered Tokens:", filtered_tokens)

for list in won_filtered_tokens:
    for tweet in list:
        total_won_words.append(tweet)
freqlist_won = nltk.FreqDist(total_won_words)

#Cleaning Lost Tweets
cleaned_lost = []
for tweet in lost_games:
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", tweet)
    cleaned_lost.append(re.sub("([^0-9A-Za-z \t])|(\w+:@\/\/\S+)", " ", tweet))
#print("Lost Cleaned:", cleaned_lost[0:4])

lost_tokens = []
total_lost_words = []
lost_filtered_tokens = []

#Tokenizing Lost Tweets
for tweet in cleaned_lost:
    lost_tokens.append(nltk.word_tokenize(tweet))

#Take Stop Words Out
for list in lost_tokens:
    temp = []
    for w in list:
        if w not in stoplist:
            temp.append(w)
    lost_filtered_tokens.append(temp)
#print("Filtered Tokens:", filtered_tokens)

for list in lost_filtered_tokens:
    for tweet in list:
        total_lost_words.append(tweet)
freqlist_lost = nltk.FreqDist(total_lost_words)

"""
#wordcloud of word frequencies for won games
wordcloud = WordCloud(width=900,height=500, max_words=1628,relative_scaling=1,normalize_plurals=False,background_color='white').generate_from_frequencies(freqlist_won)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#wordcloud of word frequencies for lost games
wordcloud = WordCloud(width=900,height=500, max_words=1628,relative_scaling=1,normalize_plurals=False,background_color='white').generate_from_frequencies(freqlist_lost)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
"""

#-------------------------------ANALYZER----------------------------------------

#-----------------------------WON GAMES-----------------------------------------

test = TextBlob('And really Dbacks Goldschmidt is mad enough as it is since Giants Tim Lincecum is not pitching in this series.')
print(test.sentiment.polarity)

wins_polarities = []
average_win_polarity = 0
for tweet in cleaned_won:
    analysis = TextBlob(tweet)
    wins_polarities.append(analysis.sentiment.polarity)
    #print(tweet, analysis.sentiment.polarity)

win_num_total = len(wins_polarities)
for game in wins_polarities:
    average_win_polarity += game
average_win_polarity = average_win_polarity / win_num_total
print("The average polarity of all games won:", average_win_polarity)

#--------------------------------LOST GAMES------------------------------------

losses_polarities = []
average_loss_polarity = 0
for tweet in cleaned_lost:
    analysis = TextBlob(tweet)
    losses_polarities.append(analysis.sentiment.polarity)
    #print(tweet, analysis.sentiment.polarity)

loss_num_total = len(losses_polarities)
for game in losses_polarities:
    average_loss_polarity += game
average_loss_polarity = average_loss_polarity / loss_num_total
print("The average polarity of all games lost:", average_loss_polarity)

#-----------------------------------T-Test--------------------------------------

# compare the polarity of the wins and losses of both teams

stat, p = stats.ttest_ind(wins_polarities, losses_polarities)
print("T-Test for Games Won vs. Lost")
print('t=%.3f, p=%.3f' % (stat, p))

#-------------------------------ClASSIFIER----------------------------------------

alltokens = []
lowercase_all_games = []
filtered_tokens = []
total_words = []

#Making All Tweets Lowercase
lowercase_all_games = [tweet.lower() for tweet in games]
#print("Lowercase:", lowercase_all_games[0:2])

#Cleaning Tweets
cleaned = []
for tweet in lowercase_all_games:
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", tweet)
    cleaned.append(re.sub("([^0-9A-Za-z \t])|(\w+:@\/\/\S+)", " ", tweet))
#print("Cleaned:", cleaned[0:3])

#Tokenizing Tweets
for tweet in cleaned:
    alltokens.append(nltk.word_tokenize(tweet))
#print("All tokens:", alltokens)

#Take Stop Words Out
for list in alltokens:
    temp = []
    for w in list:
        if w not in stoplist:
            temp.append(w)
    filtered_tokens.append(temp)
#print("Filtered Tokens:", filtered_tokens)

for list in filtered_tokens:
    for tweet in list:
        total_words.append(tweet)
freqlist_all = nltk.FreqDist(total_words)

"""
#plotting frequencies
freqlist_all.plot(30,cumulative=False)
plt.show()
"""

#Bag of words
detokenized = []
for tweet in filtered_tokens:
    detokenized.append(TreebankWordDetokenizer().detokenize(tweet))
#print("Detokened:", detokenized[0:3])

bag_of_words = CountVectorizer()
text_counts = bag_of_words.fit_transform(detokenized)

X_train, X_test, y_train, y_test = train_test_split(text_counts, y, test_size=0.3, random_state=1)

#Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))
print(confusion_matrix(y_test,predicted))
print(classification_report(y_test,predicted))

#Multinomial SVM
svm = LinearSVC().fit(X_train, y_train)
predicted = svm.predict(X_test)
print("SVM Accuracy:", metrics.accuracy_score(y_test, predicted))
print(confusion_matrix(y_test,predicted))
print(classification_report(y_test,predicted))
