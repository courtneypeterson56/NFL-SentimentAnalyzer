import csv
import tweepy

tweetIDs = []

consumer_key = 'MC6IhlxZY5qys2V8iQKGQHPyM'
consumer_secret = 'exTsFcPRU8mWNemMoLi9vQNkgeEb9SmESwaAvGxg3w5ssTUnyR'
access_token = '1186688342424879109-efDNkOCRME6DHgd3Egy3QVo4ANVXbd'
access_token_secret = 'JWTAeql3lmSW5u8zHfusdy6Ak7pCaMrpr6Q6aivgod1eM'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

with open('tweets.nfl.2012.postgame.csv','r') as csvinput:
    reader = csv.reader(csvinput)

    for row in reader:
        tweetIDs.append(row[0])

csvinput.close()

api = tweepy.API(auth)
tweets = []

i = 0
j = 100
while j<65535:
    new_tweets = api.statuses_lookup(tweetIDs[i:j])
    tweets.extend(new_tweets)
    i = i+100
    j = j+100

with open('tweets.nfl.2012.postgame.csv','r') as csvinput:
    with open('tweets.nfl.2012.postgame.outputALL.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)
            
        allTweets = []
        for row, tweet in zip(reader, tweets):
            if row[6] > row[7]:
                if tweet.text:
                    row.append(tweet.text)
                    row.append('1')
                    allTweets.append(row)
                else:
                    pass
            elif row[7] > row[6]:
                if tweet.text:
                    row.append(tweet.text)
                    row.append('0')
                    allTweets.append(row)
                else:
                    pass 
            else:
                pass

        writer.writerows(allTweets)



                
        

        

