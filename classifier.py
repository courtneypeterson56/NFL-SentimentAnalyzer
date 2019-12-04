import gensim
from nltk.corpus import stopwords
import numpy as np
import scipy as sp
import re
from sklearn.cluster import KMeans


## Here I am just customizing the nltk English stop list
stoplist = stopwords.words('english')
stoplist.extend(["ever", "one", "do","does","make", "go", "us", "to", "get", "about", "may", "s", ".", ",", "!", "i", "I", '\"', "?", ";", "--", "--", "would", "could", "”", "Mr.", "Miss", "Mrs.", "don’t", "said", "can't", "didn't", "aren't", "I'm", "you're", "they're", "'s"])
stoplist.remove("which")


## Here I am reading in the news (non-clickbait headlines)
newsheadlines = []     # this will store the original headline strings
newsheadlinetoks = []  # this will store the lists of tokens in those headlines

f = open("train_non_clickbait.txt")
for line in f:
    line = line.rstrip()
    newsheadlines.append(line)
    line = re.sub(r"(^| )[0-9]+($| )", r" ", line)  # remove digits
    addme = [t.lower() for t in line.split() if t.lower() not in stoplist]
    newsheadlinetoks.append(addme)
f.close()

## Now, just printing out an example line from the original headline strings
print(newsheadlines[50])

## And printing out the normalized list of tokens for that string
print(newsheadlinetoks[50])

########################################################

bigmodel = gensim.models.KeyedVectors.load_word2vec_format("../lab5/GoogleNews-vectors-negative300-SLIM.bin", binary=True)
newsvectors = []   # this list will contain one 300-dimensional vector per headline

for h in newsheadlinetoks:
    totvec = np.zeros(300)
    for w in h:
        if w.lower() in bigmodel:
            totvec = totvec + bigmodel[w.lower()]
    newsvectors.append(totvec)

print(len(newsvectors))
print(len(newsheadlines))
print(len(newsvectors[10]))

#this takes a while
kmnews = KMeans(n_clusters=50, random_state=0)
newsclusters = kmnews.fit_predict(newsvectors

for i in range(len(newsclusters)):
    if newsclusters[i] == 33:
        print(newsheadlines[i])

########################################################
#KNN Classification
from scipy.spatial.distance import cdist
from sklearn import metrics

## WRITE YOUR K-NEAREST NEIGHBORS CODE HERE
## COMMENT YOUR CODE CLEARLY

testtargets = []  # where to store whether a test headline is 0 or 1
testvectors = []  # where to store the vector for each headline

# while you read in test.txt...
# ...keep track of whether each headline is 1 (clickbait) or 0 (news) in the list testtargets[]
# ...AND get the summed word embedding vector for each headline and append it to list testvectors[]
headlines = []
headlinetoks = []
f = open("test.txt")
for line in f:
    line = line.rstrip()
    headlines.append(line)
    line = re.sub(r"(^| )[0-9]+($| )", r" ", line)  # remove digits
    addme = [t.lower() for t in line.split() if t.lower() not in stoplist]
    headlinetoks.append(addme)
    testtargets = [int(headlinetoks[i][0]) for i in range(len(headlinetoks))]
f.close()
## CREATE YOUR VECTORS
for h in headlinetoks:
    testvec = np.zeros(300)
    for w in h:
        if w.lower() in bigmodel:
            testvec = testvec + bigmodel[w.lower()]
    testvectors.append(testvec)

## SANITY CHECKING
# len(testvectors) should equal 2000 and should be a list of lists
# len(testvectors[100]) should equal 300
# len(testtargets) should equal 2000 and should be a list of 1s and 0s
#print(testvectors)

## GET THE COSINE DISTANCES
# get the cosine distance between the each test vector and each of the clickbait vectors
# use scipy.spatial.distance.cdist(testvectors, clickvectors)
# save the output of cdist to a 2D array called clickdistances
# each row will correspond to one test vector
# each value in the row will correspond to the distance between that vector
# and one of the new vectors
clickdistances = sp.spatial.distance.cdist(testvectors, clickvectors)

# get the cosine distance between the each test vector and each of the news vectors
# use scipy.spatial.distance.cdist(testvectors, newsvectors)
# save the output of cdist to a 2D array called newsdistances
# each row will correspond to one test vector
# each value in the row will correspond to the distance between that vector
# and one of the new vectors
newsdistances = sp.spatial.distance.cdist(testvectors, newsvectors)

## GET THE MIN COSINE DISTANCES
# get the min of of each row in clickdistances using clickdistances.min(axis=1)
# save out to a list or vector called clickmins
clickmins = clickdistances.min(axis=1)
# get the min of of each row in newsdistances using newsdistances.min(axis=1)
# save out to a list or vector called newsmins
newsmins = newsdistances.min(axis=1)

## GET YOUR PREDICTIONS
predictedknn = []  # where to store your KNN predictions

# loop through the mins in newsmins and clickmins
# if the news min is smaller than the click min, append 0 to predictedknn
# otherwise append 1 to knnpredicted
for i in range(len(newsmins)):
    if newsmins[i] > clickmins[i]:
        predictedknn.append(1)
    else:
        predictedknn.append(0)

## EVALUATE YOUR PREDICTIONS
# print the classification report
print(metrics.classification_report(testtargets, predictedknn))
